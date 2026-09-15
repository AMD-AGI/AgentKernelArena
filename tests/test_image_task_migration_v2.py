"""CPU regression evidence for the 21 image tasks' v2 migration.

Real image/GPU qualification is separate; these tests never represent simulated
latencies as measured hardware results.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from src.task_protocol import CaseManifest, parse_command_result
from src.task_session import TaskSession
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks/image_kernel"
DIRECTORIES = sorted(p.parent for p in TASKS.glob("*/config.yaml"))
CPU_REFERENCE_TASKS = [d for d in DIRECTORIES if d.name not in {
    "mi300x_sglang_hip_pa_decode", "mi300x_sglang_hip_pa_ragged",
    "mi355x_vllm_ck_moe_2stage", "mi355x_vllm_ck_cktile_moe_2stage",
    "mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3", "mi355x_vllm_tilelang_mhc_fused_post_pre",
}]

# Immutable evidence collected from the pre-migration harness at 5c9f8ef2.
ORIGINAL_EVIDENCE = {'mi300x_sglang_hip_mha_batch_prefill': {'cases': 'a00c0df97e2eaf67455f794dccc265c9fd77773ca91e9d7c974adcefa60ba8e4',
                                         'phases': {'run_compile': '753fa096b05b188fdf6fc87e3169116f3f18e4e2ae855b32d9381c924362785c',
                                                    'run_correctness': '8a2c9aa3fe4e49ba7ab86c32072e4215f9f47808d95301be3708d710ce329905',
                                                    'run_performance': 'cee333ba31775b913b237ef2510be426e02bdc6faf4bb37c2f8708481219ad10'},
                                         'numbers': {'_make_case': [0,
                                                                    0,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    0,
                                                                    0,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    1,
                                                                    0,
                                                                    256,
                                                                    0,
                                                                    1,
                                                                    1.0]},
                                         'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9'},
 'mi300x_sglang_hip_pa_decode': {'cases': '767b17d152975c05f5af5370978665a53f19a455b07f403c4f2fb8fed3809775',
                                 'phases': {'run_compile': '753fa096b05b188fdf6fc87e3169116f3f18e4e2ae855b32d9381c924362785c',
                                            'run_correctness': '8a2c9aa3fe4e49ba7ab86c32072e4215f9f47808d95301be3708d710ce329905',
                                            'run_performance': '4d663917bafbb0c1f9561cbb3f7f18ff33eca71f3d96e8d195496c184eb497be'},
                                 'numbers': {'_make_case': [256,
                                                            0,
                                                            0,
                                                            1,
                                                            1,
                                                            0,
                                                            0,
                                                            8,
                                                            1.0,
                                                            1.0,
                                                            1,
                                                            1,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            1,
                                                            1,
                                                            0.5,
                                                            1,
                                                            0,
                                                            1,
                                                            1,
                                                            4,
                                                            1,
                                                            1,
                                                            2,
                                                            0]},
                                 'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9'},
 'mi300x_sglang_hip_pa_ragged': {'cases': 'c62df2a3a968735fccf145d60769d98a92ec4914763c6e9cb3d4e890430c608c',
                                 'phases': {'run_compile': '753fa096b05b188fdf6fc87e3169116f3f18e4e2ae855b32d9381c924362785c',
                                            'run_correctness': '8a2c9aa3fe4e49ba7ab86c32072e4215f9f47808d95301be3708d710ce329905',
                                            'run_performance': '4d663917bafbb0c1f9561cbb3f7f18ff33eca71f3d96e8d195496c184eb497be'},
                                 'numbers': {'_make_case': [256,
                                                            0,
                                                            0,
                                                            1,
                                                            1,
                                                            0,
                                                            0,
                                                            8,
                                                            1.0,
                                                            1.0,
                                                            1,
                                                            1,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            1,
                                                            1,
                                                            0.5,
                                                            1,
                                                            0,
                                                            1,
                                                            1,
                                                            4,
                                                            1,
                                                            1,
                                                            2,
                                                            0]},
                                 'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9'},
 'mi300x_sglang_triton_fp8_gemm': {'cases': 'b995d6abd0196199dbb6f088e1ea26ccaa74228a9e8ad78fe89ecf5fc6011b36',
                                   'phases': {'run_compile': '753fa096b05b188fdf6fc87e3169116f3f18e4e2ae855b32d9381c924362785c',
                                              'run_correctness': '561e04027ae83bfb5b65b593ed7dd7a154305d166055ea6a1adb9937424b2ec5',
                                              'run_performance': '07933100c8ea2a468f77a2502f04dc5bb546a8f14f0da75cee920b3ae207d3ee'},
                                   'numbers': {'_make_case': [0, 10, 1, 1, 1]},
                                   'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9'},
 'mi300x_sglang_triton_gemm': {'cases': '613c00c6d96b7bceec196cecd7d085856d077d3d79485920ae5368d24f7b9596',
                               'phases': {'run_compile': '753fa096b05b188fdf6fc87e3169116f3f18e4e2ae855b32d9381c924362785c',
                                          'run_correctness': 'b39a14d0eeb8c656a680f95cb5ad2e643434c0dbe9156252c116ee684a3cfc78',
                                          'run_performance': '07933100c8ea2a468f77a2502f04dc5bb546a8f14f0da75cee920b3ae207d3ee'},
                               'numbers': {'_make_case': [0]},
                               'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9'},
 'mi355x_sglang_triton_mxfp8_grouped_gemm': {'cases': '6f196129c4e5d6eac03293346014d82e4cc35868412983679b63db3502ad6f96',
                                             'phases': {'run_compile': '517c12ccc1003dd4fbc36a92d8b6bf521e9e15a4fae7439bf54192646d76075d',
                                                        'run_correctness': '661adefe5b32114607b858261f37645d883535bab96ee329e787159af549024b',
                                                        'run_performance': '77f2aabe0b8f3e278399e417a34c90bb5f2c21c9c0d1881d7adc21055be68d57'},
                                             'numbers': {'_relerr': [1e-08],
                                                         '_make': [0.5, 0.1, 0.1, 0, 1, 2, 1, 1, 1],
                                                         '_assert_timed_outputs': [0.08, 2]},
                                             'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                             'session': '8ba375388d80ce2fe2a221aca062c0616937c6ee1a554a0951b09e97c6508c0d'},
 'mi355x_sglang_triton_mxfp8_linear': {'cases': '17dcb80e1121993f7ba09c686a5b53d5d4166216a4ebb4309d62838d43a5ec68',
                                       'phases': {'run_compile': 'd1473d33dd308b3c9bcbe5bcce56d672faa8ef9a060f688e42a4950edaac7410',
                                                  'run_correctness': 'bd35cf0823839e2c63b905071365781e57621263921f546487a200f22acc4975',
                                                  'run_performance': '55db6b4a9ed0b125edaf3fe5497710f4b2f448bffd1d4c6fafef4b758d71c15c'},
                                       'numbers': {'_relerr': [1e-08],
                                                   '_make': [0.1, 0.5, 0],
                                                   '_assert_close': [0.06],
                                                   '_perturb_inputs': [59, 0.5],
                                                   '_assert_timed_outputs': []},
                                       'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                       'session': 'f27ca1eb4012059ef8ae83cb10705b93cacb420b1cd2c8e6a06daa5998a924ae'},
 'mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3': {'cases': 'c5b3565a69c605d7cddceef2e875c83c627408d5fd38bf4cef850df7280b1eef',
                                                'phases': {'run_compile': '147d7503a8f7f6e3ca22c090e5a16135c437d0d2f6dca4b0688910c33f269e7b',
                                                           'run_correctness': '445116c7ea4b21c535026c70fe9e569363eafdf51eaedac465e50be2e3be3a97',
                                                           'run_performance': 'c93262ab32d0c0bcaf4867fe9af1b8a42f86fc7c05eb1331971464c91015f667'},
                                                'numbers': {'_assert_tuned_dispatch': [2000],
                                                            '_assert_vllm_shuffle_contract': [16],
                                                            '_prepare': [0.1,
                                                                         0.03,
                                                                         0.03,
                                                                         16,
                                                                         19,
                                                                         2],
                                                            '_moe_deviation': [0, 1e-08],
                                                            '_assert_moe_within_tolerance': [0.97,
                                                                                             0.25],
                                                            '_perturb_moe_inputs': [43, 0.0, 0.1],
                                                            '_assert_timed_outputs': []},
                                                'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                                'session': '9712aacc0d0e177425df75b32a4e86255cf89873499a8e612f9eae4589eb8629'},
 'mi355x_vllm_ck_a8w8_blockscale_gemm': {'cases': '8f1dc33577c4a5cfaa54dc10b72aa54bc1ae4ffb85172710efe4cd480bde77e0',
                                         'phases': {'run_compile': '2b38d369ef9b289de2fae70c23b8d582a8b6570f72800847d47ba6e7a30d969e',
                                                    'run_correctness': '96eb935cbc5d50a8defaaffbd5eb8058c56ebe5b42a00cf3af7e925a6e61a737',
                                                    'run_performance': '19d1f27379fea10f9a5f6342646a743a83bf4a9a3a37b8c7440fa57f98e9ab49'},
                                         'numbers': {'_make_attention': [7,
                                                                         1,
                                                                         128,
                                                                         1,
                                                                         0.7,
                                                                         1,
                                                                         0.5,
                                                                         0.7],
                                                     '_make_gemm': [9,
                                                                    0.01,
                                                                    0.01,
                                                                    64,
                                                                    0.1,
                                                                    0.1,
                                                                    0.1,
                                                                    0.1,
                                                                    0.2,
                                                                    0.2,
                                                                    128,
                                                                    128,
                                                                    128],
                                                     '_make_quant': [11, 1],
                                                     '_make_mhc': [13, 0.001, 3, 64, 1, 2, 1],
                                                     '_make_mla': [17, 0, 64, 128, 1, 1024, 1],
                                                     '_prepare_moe': [19,
                                                                      0.1,
                                                                      0.03,
                                                                      0.03,
                                                                      64,
                                                                      16,
                                                                      16,
                                                                      1,
                                                                      2,
                                                                      1,
                                                                      2,
                                                                      2,
                                                                      1,
                                                                      1,
                                                                      2,
                                                                      2,
                                                                      16,
                                                                      16,
                                                                      16,
                                                                      16],
                                                     '_make': []},
                                         'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                         'session': '320fb13807210c99a47e99a802a3558800d1deb41847b07f0ab78d573da289d9'},
 'mi355x_vllm_ck_cktile_moe_2stage': {'cases': 'fa18fe5f310333a08ba905a9e80c42acd280f0be88907f13c0e57fa57fdca480',
                                      'phases': {'run_compile': '2b38d369ef9b289de2fae70c23b8d582a8b6570f72800847d47ba6e7a30d969e',
                                                 'run_correctness': '96eb935cbc5d50a8defaaffbd5eb8058c56ebe5b42a00cf3af7e925a6e61a737',
                                                 'run_performance': '19d1f27379fea10f9a5f6342646a743a83bf4a9a3a37b8c7440fa57f98e9ab49'},
                                      'numbers': {'_make_attention': [7,
                                                                      1,
                                                                      128,
                                                                      1,
                                                                      0.7,
                                                                      1,
                                                                      0.5,
                                                                      0.7],
                                                  '_make_gemm': [9,
                                                                 0.01,
                                                                 0.01,
                                                                 64,
                                                                 0.1,
                                                                 0.1,
                                                                 0.1,
                                                                 0.1,
                                                                 0.2,
                                                                 0.2,
                                                                 128,
                                                                 128,
                                                                 128],
                                                  '_make_quant': [11, 1],
                                                  '_make_mhc': [13, 0.001, 3, 64, 1, 2, 1],
                                                  '_make_mla': [17, 0, 64, 128, 1, 1024, 1],
                                                  '_prepare_moe': [19,
                                                                   0.1,
                                                                   0.03,
                                                                   0.03,
                                                                   64,
                                                                   16,
                                                                   16,
                                                                   1,
                                                                   2,
                                                                   1,
                                                                   2,
                                                                   2,
                                                                   1,
                                                                   1,
                                                                   2,
                                                                   2,
                                                                   16,
                                                                   16,
                                                                   16,
                                                                   16],
                                                  '_make': []},
                                      'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                      'session': 'e06b373166510ccf6a0235fc7c765712fbb8dfe861f8f73b4ea8ee7fd7941aeb'},
 'mi355x_vllm_ck_moe_2stage': {'cases': 'f421e5e31be970cdf2ee36438cda25f42764f8a0bd1b25efd683788f8b8a1d83',
                               'phases': {'run_compile': '2b38d369ef9b289de2fae70c23b8d582a8b6570f72800847d47ba6e7a30d969e',
                                          'run_correctness': '96eb935cbc5d50a8defaaffbd5eb8058c56ebe5b42a00cf3af7e925a6e61a737',
                                          'run_performance': '19d1f27379fea10f9a5f6342646a743a83bf4a9a3a37b8c7440fa57f98e9ab49'},
                               'numbers': {'_make_attention': [7, 1, 128, 1, 0.7, 1, 0.5, 0.7],
                                           '_make_gemm': [9,
                                                          0.01,
                                                          0.01,
                                                          64,
                                                          0.1,
                                                          0.1,
                                                          0.1,
                                                          0.1,
                                                          0.2,
                                                          0.2,
                                                          128,
                                                          128,
                                                          128],
                                           '_make_quant': [11, 1],
                                           '_make_mhc': [13, 0.001, 3, 64, 1, 2, 1],
                                           '_make_mla': [17, 0, 64, 128, 1, 1024, 1],
                                           '_prepare_moe': [19,
                                                            0.1,
                                                            0.03,
                                                            0.03,
                                                            64,
                                                            16,
                                                            16,
                                                            1,
                                                            2,
                                                            1,
                                                            2,
                                                            2,
                                                            1,
                                                            1,
                                                            2,
                                                            2,
                                                            16,
                                                            16,
                                                            16,
                                                            16],
                                           '_make': []},
                               'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                               'session': 'a566c3946fd38e59e15dd7b823988d2ea1dfadd30a77dc6a0003f606727811bb'},
 'mi355x_vllm_hip_dynamic_per_tensor_quant': {'cases': '6853933842248e459f4a68180923ac4dac19e1a35ae39cb2e36dd4a6644cae67',
                                              'phases': {'run_compile': 'c845fd98673b8a55f66f53cd75f8d44182dc46396037d1fc0d42878c48b60617',
                                                         'run_correctness': 'bc7163cb7701b02dcf2262e5aa7124cf1ff345773c0010a5db89e69b3ca97e53',
                                                         'run_performance': 'e34ccfe6f54d3b680fca91894685fa21ab8b747800ba47f1a69794bbf48371b6'},
                                              'numbers': {'_make_quant': [11, 1],
                                                          '_assert_quant_correct': [1,
                                                                                    1e-05,
                                                                                    0.02,
                                                                                    0.25,
                                                                                    0.15],
                                                          '_perturb_quant_inputs': [31],
                                                          '_assert_operator': [],
                                                          '_make': [],
                                                          '_assert_timed_outputs': []},
                                              'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                              'session': 'aa545a4269da356539bfa699eb87044a0a80b8c60ad63fbdc9fc87b70a58249a'},
 'mi355x_vllm_hip_paged_attention_decode': {'cases': '808e77ced5dacf90723042c4ed0eba77599fbdac4f9b01865fad4565499bdcd4',
                                            'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                       'run_correctness': 'fd8b471cefdfebecf76f08e75b32e51a44973d5ad34c3aff94125affb60dc926',
                                                       'run_performance': '80e057d960c3e0153a843eb0002c91ea442561771acaed5fba5b4ab195ecc933'},
                                            'numbers': {'_make': [29,
                                                                  1,
                                                                  1,
                                                                  0.5,
                                                                  1,
                                                                  2,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1],
                                                        '_perturb_inputs': [71],
                                                        '_assert_close': [0.02, 0.02],
                                                        '_assert_timed_outputs': []},
                                            'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                            'session': '34c8c807c463fb09a9c6bdfce1d5bb7269fcd95ef398042abada3ba89667c3fb'},
 'mi355x_vllm_tilelang_mhc_fused_post_pre': {'cases': '665db7fbc0af7458feb1bc422918fea750cee3e24b8088f9cd9da660364ec52e',
                                             'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                        'run_correctness': 'dfcc4178f00911b1190f996b8f21d3ca9b1c95434215107a8f2fdb658a7937b4',
                                                        'run_performance': 'e34ccfe6f54d3b680fca91894685fa21ab8b747800ba47f1a69794bbf48371b6'},
                                             'numbers': {'_assert_operator': [],
                                                         '_make_mhc': [13, 0.001, 3, 1, 2, 1],
                                                         '_assert_mhc_close': [0.08, 0.08],
                                                         '_perturb_mhc_inputs': [29, 1],
                                                         '_make': [],
                                                         '_assert_timed_outputs': []},
                                             'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                             'session': '9ac2a8ec7a2feab833c8dc2965dd2d61a3f36259d1c5d43a124f13941f040348'},
 'mi355x_vllm_triton_fused_moe_gemma4': {'cases': '5bcd34e9e4371a170527b202a9e9b6d45bdf2d1c3158144f939ec76002db50cc',
                                         'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                    'run_correctness': 'be8cf07a799302a123645510bb2fc01a4f7918ba5598eeb0a8660a23b6a157cd',
                                                    'run_performance': '89f7c1078f2fb2953180bb99de20121253655b251b371f654d8731f8954e8410'},
                                         'numbers': {'_make': [31, 0.5, 0.5, 1, 2, 1],
                                                     '_assert_close': [0.03, 0.03],
                                                     '_perturb_inputs': [61],
                                                     '_assert_timed_outputs': []},
                                         'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                         'session': '7b41cd3388b2c8074e8c1ae6e44d36c30d1153d56630526e454f8e9a10ee7207'},
 'mi355x_vllm_triton_fused_moe_gptq_awq': {'cases': 'edc9b685433bbe82df976fc3a244915dfdee04fae7d9ae655b86c2d3ce5d1445',
                                           'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                      'run_correctness': '9cc08d47222168a2ad7f296e6e3450ac4a47ca777627cd3dbcd5322830139bf7',
                                                      'run_performance': '80e057d960c3e0153a843eb0002c91ea442561771acaed5fba5b4ab195ecc933'},
                                           'numbers': {'_make': [31, 0, 0, 0.5, 0.5, 1, 2, 1],
                                                       '_assert_close': [0.02, 0.02],
                                                       '_perturb_inputs': [53],
                                                       '_assert_timed_outputs': []},
                                           'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                           'session': '824e092df75644e5ce5de0c6454e33e409c30903d512b32059a9c66365fc8f70'},
 'mi355x_vllm_triton_kda_linear_attn_kimi_k3': {'cases': '558f39d15191479e60cba7ef775c0ad6177d6743b807151e63c779da2c6efcbe',
                                                'phases': {'run_compile': '887228816472403d8e9c0459366a05aeade3c558cc5998f8a13c92ebfa965afe',
                                                           'run_correctness': '724c843f4197f956f733eae801003bc659c6fdd629fa610ead4d46a166aeea68',
                                                           'run_performance': '1290d1826cc5e2f4eef7755190016a3352f017c1fd894d227904c6d1e7964658'},
                                                'numbers': {'_prepare': [1.0,
                                                                         4.0,
                                                                         0.1,
                                                                         1,
                                                                         1,
                                                                         2.0,
                                                                         1,
                                                                         1,
                                                                         1,
                                                                         0,
                                                                         1,
                                                                         23,
                                                                         0.5,
                                                                         0.5,
                                                                         0.5,
                                                                         0.5,
                                                                         1,
                                                                         1,
                                                                         1,
                                                                         0.1,
                                                                         1,
                                                                         0.5,
                                                                         1,
                                                                         0.1,
                                                                         3,
                                                                         1]},
                                                'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                                'session': '2fede60f9eb1c88d9e635bf1bd37bcca2bfc2a8b84f4e70fe0531f408956923f'},
 'mi355x_vllm_triton_paged_attention_2d': {'cases': 'd6516a40b59a7eb60404178883085085bd4f305c26135da73942a9976bb26704',
                                           'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                      'run_correctness': '27ef45de2d13aad48c4ea65b18d316ccdf04d7da40bee878b0d859093c2c1069',
                                                      'run_performance': '80e057d960c3e0153a843eb0002c91ea442561771acaed5fba5b4ab195ecc933'},
                                           'numbers': {'_make': [23,
                                                                 1,
                                                                 1,
                                                                 0.5,
                                                                 1,
                                                                 1,
                                                                 2,
                                                                 1,
                                                                 1,
                                                                 1,
                                                                 1,
                                                                 1,
                                                                 1,
                                                                 1],
                                                       '_perturb_inputs': [37],
                                                       '_assert_close': [0.08, 0.08],
                                                       '_assert_timed_outputs': []},
                                           'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                           'session': '5c8e419c1e2dea7f18db3ec6fc0158802c2631ad5787f8b3a385b3cc3c65b257'},
 'mi355x_vllm_triton_sparse_attn_prefill_ragged': {'cases': '62043fa503aa1ce6acec453b07f5f0cc55fea79c6208cd4b6f511c535eb41bdd',
                                                   'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                              'run_correctness': '27ef45de2d13aad48c4ea65b18d316ccdf04d7da40bee878b0d859093c2c1069',
                                                              'run_performance': '80e057d960c3e0153a843eb0002c91ea442561771acaed5fba5b4ab195ecc933'},
                                                   'numbers': {'_make': [29, 0, 0, 0.5, 1, 1, 1],
                                                               '_assert_close': [0.08, 0.08],
                                                               '_perturb_inputs': [47],
                                                               '_assert_timed_outputs': []},
                                                   'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                                   'session': '3fffa30992afd56ba70ee725282825d6480593fe7b57b5a8cd8af286bee10517'},
 'mi355x_vllm_triton_unified_attention': {'cases': '6ad488de6b221b93a7142a8523eff9732c7dc57a0aab689629a0ee330b6cead6',
                                          'phases': {'run_compile': '6a3368288b8e648e32bb0a74887de84140b1aac42ff47560191d3a5913274cb7',
                                                     'run_correctness': '69651ab26caf0acc3d55ae0d7d9f15a31c6903a604a51377efc3299f7a7cae86',
                                                     'run_performance': 'e34ccfe6f54d3b680fca91894685fa21ab8b747800ba47f1a69794bbf48371b6'},
                                          'numbers': {'_make_attention': [7,
                                                                          1,
                                                                          1,
                                                                          0.7,
                                                                          1,
                                                                          0.5,
                                                                          0.7],
                                                      '_assert_operator': [],
                                                      '_make': [],
                                                      '_assert_attention_close': [0.08, 0.08],
                                                      '_perturb_attention_inputs': [41, 0.7],
                                                      '_assert_timed_outputs': []},
                                          'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                          'session': '6c6d249224627747314de00e73d46b5413d348a88e28320564481fd5bd97b33d'},
 'mi355x_vllm_triton_unified_attention_gemma4': {'cases': '3f78038b35a508c9971d07318e0aba5d98c33ab9f16c8483b77cefa3e62b3ea4',
                                                 'phases': {'run_compile': '29d22a3d115d85b583589e50d67ac91463433b0564832dec079c2bbf4480833b',
                                                            'run_correctness': 'fd8b471cefdfebecf76f08e75b32e51a44973d5ad34c3aff94125affb60dc926',
                                                            'run_performance': 'ed80c02942154b5e1ed1c1e72308a04fd5cc198e44434358e2ea2c002ee13f5c'},
                                                 'numbers': {'_make': [31,
                                                                       1,
                                                                       1,
                                                                       0.5,
                                                                       0,
                                                                       1,
                                                                       2,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1,
                                                                       1],
                                                             '_perturb_inputs': [67],
                                                             '_assert_close': [0.02, 0.02],
                                                             '_assert_timed_outputs': []},
                                                 'generated': 'acd723f83091e920a2fb0b731441a1aa8ca65d97cfd70460c0af50e1b49fb8f9',
                                                 'session': '632eaf47fc9b9f2bfc65fa2ad210261bf646ea4d729563415646abb484503306'}}


def normalized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(normalized(value).encode()).hexdigest()


def load_module(path):
    name = "image_test_" + hashlib.sha256(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def local_modules(directory):
    names = ("task_adapter", "setup_task", "reference_controls", "reference_mxfp8", "source_build")
    saved = {name: sys.modules.pop(name, None) for name in names}
    sys.path.insert(0, str(directory / "scripts"))
    try:
        yield
    finally:
        sys.path.pop(0)
        for name, value in saved.items():
            sys.modules.pop(name, None)
            if value is not None:
                sys.modules[name] = value


def test_inventory_and_declared_roles():
    assert len(DIRECTORIES) == 21
    total_c = total_p = 0
    for directory in DIRECTORIES:
        spec = load_task_spec(directory / "config.yaml", task_id="image_kernel/" + directory.name)
        assert spec.candidate.initial_state == "implemented"
        assert spec.baseline.kind == "initial_candidate"
        assert spec.baseline.correctness_policy == "required"
        assert all(edit.path.count("/") >= 1 for edit in spec.candidate.editable)
        assert all(action.commands[0][:2] == ("python3", "scripts/evaluate.py") for action in spec.actions)
        data = json.loads((directory / "workloads.json").read_text())
        total_c += sum("correctness" in c["checks"] for c in data["cases"])
        total_p += sum("performance" in c["checks"] for c in data["cases"])
        for script in (directory / "scripts").glob("*.py"):
            for node in ast.walk(ast.parse(script.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert (node.module or "").split(".")[0] not in {"src", "agents"}
                elif isinstance(node, ast.Import):
                    assert not any(alias.name.split(".")[0] in {"src", "agents"} for alias in node.names)
    # Preserve all 98 migrated correctness cases and 68 measured cases; add two
    # unscored ragged-CSR checks after the GPU quality review found uniform-only
    # coverage of the sparse attention operator.
    assert (total_c, total_p) == (100, 68)


@pytest.mark.parametrize("directory", DIRECTORIES, ids=lambda d: d.name)
def test_task_execution_has_no_retired_agent_driver_dependency(directory):
    retired = {"forge_driver", "standalone_driver"}
    assert not any((directory / "scripts" / f"{name}.py").exists() for name in retired)
    # Include setup and reference code, not just the configured CLI. Direct
    # imports and dynamic file-loader arguments must not retain deleted modules.
    for script in (directory / "scripts").glob("*.py"):
        for node in ast.walk(ast.parse(script.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[-1] not in retired for alias in node.names), script
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[-1] not in retired, script
                assert all(alias.name not in retired for alias in node.names), script
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not any(f"{name}.py" in node.value for name in retired), script
    spec = load_task_spec(directory / "config.yaml", task_id="image_kernel/" + directory.name)
    assert len(spec.actions) == 7
    for action in spec.actions:
        assert action.commands[0][:2] == ("python3", "scripts/evaluate.py")


@pytest.mark.parametrize("task_name,case_id", [
    ("mi355x_vllm_hip_paged_attention_decode", "llama3_1-8b-decode-m64-ctx1024"),
    ("mi355x_vllm_triton_unified_attention_gemma4", "gemma4-sliding-decode-m64-ctx1024"),
    ("mi355x_vllm_triton_fused_moe_gemma4", "gemma4-moe-decode-m64"),
])
def test_public_profiling_case_selection_survives_driver_removal(task_name, case_id):
    directory = TASKS / task_name
    runner = load_module(directory / "scripts/task_runner.py")
    selected = runner.profile_case()
    assert selected["id"] == case_id
    assert any(selected is case for case in runner.CASES)
    manifest = json.loads((directory / "workloads.json").read_text())
    assert any(case["test_case_id"] == case_id and "performance" in case["checks"]
               for case in manifest["cases"])


@pytest.mark.parametrize("name,activation,expected", [
    ("mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3", "situv2",
     [[4.0, -0.50390625], [-1.390625, 0.2177734375]]),
    ("mi355x_vllm_ck_moe_2stage", "silu",
     [[4.3125, -0.5859375], [-1.421875, 0.22265625]]),
    ("mi355x_vllm_ck_cktile_moe_2stage", "silu",
     [[4.3125, -0.5859375], [-1.421875, 0.22265625]]),
])
def test_two_stage_moe_known_answer_includes_bf16_intermediate(name, activation, expected):
    torch = pytest.importorskip("torch")
    controls = load_module(TASKS / name / "scripts/reference_controls.py")
    _, answer = controls.moe_data(activation)
    # Independent two-token/two-expert scalar answer, rounded to BF16 between
    # gate/up and down projections. The unrounded Kimi control returned 4.03125.
    assert answer.dtype == torch.bfloat16
    torch.testing.assert_close(answer.float(), torch.tensor(expected), rtol=0, atol=0)
    controls.rejects(lambda bad: controls.equal(bad, answer), torch.zeros_like(answer))


@pytest.mark.parametrize("directory", DIRECTORIES, ids=lambda d: d.name)
def test_original_cases_numerical_policy_and_benchmark_unchanged(directory):
    harness = load_module(directory / "scripts/task_runner.py")
    original = {"correctness": harness.CASES, "performance": getattr(harness, "PERF_CASES", None)}
    evidence = ORIGINAL_EVIDENCE[directory.name]
    assert digest(original) == evidence["cases"]
    data = json.loads((directory / "workloads.json").read_text())
    assert normalized(original) == normalized(data["original_cases"])
    tree = ast.parse((directory / "scripts/task_runner.py").read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, expected in evidence["phases"].items():
        function = deepcopy(functions[name])
        if name == "run_performance":
            assert isinstance(function.body[-1], ast.Return)
            function.body.pop()  # Return the fresh rows to v2.
            if (directory.name.startswith("mi300x_") or directory.name.startswith("mi355x_vllm_ck_")
                    or directory.name == "mi355x_vllm_triton_kda_linear_attn_kimi_k3"):
                # Post-migration fix: captured-output correctness after timing.
                # Remove only the three reviewed additions before comparing the
                # original measurement body, preserving all timing constants.
                loop = next(n for n in function.body if isinstance(n, ast.For))
                collector = [n for n in loop.body if isinstance(n, ast.Assign)
                             and isinstance(n.value, ast.Call)
                             and isinstance(n.value.func, ast.Name) and n.value.func.id == "_TimedRun"]
                check = [n for n in loop.body if isinstance(n, ast.Expr)
                         and isinstance(n.value, ast.Call)
                         and isinstance(n.value.func, ast.Name) and n.value.func.id == "_assert_timed_outputs"]
                assert len(collector) == len(check) == 1
                loop.body = [n for n in loop.body if n not in collector + check]
                if (directory.name.startswith("mi355x_vllm_ck_")
                        or directory.name == "mi355x_vllm_triton_kda_linear_attn_kimi_k3"):
                    prep = [n for n in loop.body if isinstance(n, ast.Assign)
                            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                            and n.value.func.id == "_prepare_timed_check"]
                    assert len(prep) == 1
                    loop.body.remove(prep[0])
                if directory.name == "mi355x_vllm_triton_kda_linear_attn_kimi_k3":
                    observer = [n for n in loop.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Attribute) and t.attr == "before_sample" for t in n.targets)]
                    assert len(observer) == 1
                    loop.body.remove(observer[0])
                calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "_benchmark_cuda_graph_or_events"]
                assert len(calls) == 1
                assert [k.arg for k in calls[0].keywords].count("timed_run") == 1
                calls[0].keywords = [k for k in calls[0].keywords if k.arg != "timed_run"]
        if name == "run_correctness" and directory.name.startswith("mi355x_vllm_ck_"):
            loop = next(n for n in function.body if isinstance(n, ast.For))
            checks = [n for n in loop.body if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == "_assert_output_contract"]
            assert len(checks) == 1
            loop.body.remove(checks[0])
        if name == "run_correctness" and directory.name == "mi355x_vllm_triton_kda_linear_attn_kimi_k3":
            loop = next(n for n in function.body if isinstance(n, ast.For))
            snapshots = [n for n in loop.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "before" for t in n.targets)]
            checks = [n for n in loop.body if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == "_assert_state_result"]
            assert len(snapshots) == len(checks) == 1
            loop.body = [n for n in loop.body if n not in snapshots + checks]
            extended = [n for n in loop.body if isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                        and n.value.func.id == "_golden"]
            assert len(extended) == 1
            index = loop.body.index(extended[0])
            loop.body[index] = ast.parse("ref = _golden(inp)").body[0]
        assert digest(ast.dump(function, include_attributes=False)) == expected
    for name, expected in evidence["numbers"].items():
        actual = [n.value for n in ast.walk(functions[name])
                  if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
        assert actual == expected, f"Numeric gate or input/timing constant changed in {name}"
    text = (directory / "scripts/task_runner.py").read_text()
    generated = text[text.index("# >>> AKA-GENERATED:"):text.index("# <<< AKA-GENERATED <<<")]
    assert hashlib.sha256(generated.encode()).hexdigest() == evidence["generated"]
    if evidence.get("session"):
        assert hashlib.sha256((directory / "session_cases.json").read_bytes()).hexdigest() == evidence["session"]


@pytest.mark.parametrize("directory", DIRECTORIES, ids=lambda d: d.name)
@pytest.mark.parametrize("argv", [["validate-task"], ["baseline", "compile"], ["baseline", "correctness"],
                                  ["baseline", "performance"], ["candidate", "compile"],
                                  ["candidate", "correctness"], ["candidate", "performance"]])
def test_unmaterialized_image_sources_fail_with_valid_envelope(directory, argv):
    # Launch from another cwd with no repository imports on the child path.
    proc = subprocess.run([sys.executable, str(directory / "scripts/evaluate.py"), *argv],
                          cwd=directory.parent, capture_output=True, text=True, timeout=20)
    role, action = ("task", "validate-task") if len(argv) == 1 else argv
    result = parse_command_result(proc.stdout, role=role, action=action, returncode=proc.returncode)
    assert not result.passed
    assert result.failure_kind == "evaluation_error"
    assert "numerical_mismatch" not in proc.stdout
    assert result.reason


@pytest.mark.parametrize("directory", CPU_REFERENCE_TASKS, ids=lambda d: d.name)
def test_real_reference_known_answers_and_negative_controls(directory):
    torch = pytest.importorskip("torch")
    with local_modules(directory):
        harness = load_module(directory / "scripts/task_runner.py")
        # Only remove the GPU availability gate; exercise the real mathematical
        # reference and original comparison on explicitly constructed CPU inputs.
        harness._torch = lambda: torch
        controls = load_module(directory / "scripts/reference_controls.py")
        result = controls.check_reference(harness)
        assert result == {"known_answer": "PASS", "negative_control": "PASS", "scored": False}


def test_manifest_includes_non_scored_buckets_and_attention_paths():
    kimi = json.loads((TASKS / "mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3/workloads.json").read_text())
    assert len(kimi["cases"]) == 14
    assert sum(c["checks"] == ["correctness"] for c in kimi["cases"]) == 12
    attention = json.loads((TASKS / "mi355x_vllm_triton_unified_attention/workloads.json").read_text())
    small = [c for c in attention["cases"] if c["test_case_id"].endswith(":2d")]
    assert len(small) == 5
    assert all(c["params"]["ctx_len"] == 128 and c["checks"] == ["correctness"] for c in small)


def test_sparse_ragged_manifest_preserves_all_scored_cases():
    directory = TASKS / "mi355x_vllm_triton_sparse_attn_prefill_ragged"
    harness = load_module(directory / "scripts/task_runner.py")
    adapter = load_module(directory / "scripts/task_adapter.py")
    adapter.validate_workloads(harness)
    data = json.loads((directory / "workloads.json").read_text())
    scored = [c for c in data["cases"] if "performance" in c["checks"]]
    assert [c["test_case_id"] for c in scored] == [c["id"] for c in harness.CASES]
    extra = [c for c in data["cases"] if c["checks"] == ["correctness"]]
    assert len(extra) == 2
    assert [c["params"]["num_queries"] for c in extra] == [64, 1073]
    assert all(c["params"]["row_lengths"] == [0, 1, 15, 16, 17, 31, 32, 33, 127, 255, 511, 512]
               for c in extra)
    harness.RAGGED_CASES = harness.RAGGED_CASES[:-1]
    with pytest.raises(ValueError, match="ragged manifest"):
        adapter.validate_workloads(harness)


@pytest.mark.parametrize("chunk", [1, 3, 512])
def test_sparse_ragged_reference_matches_independent_row_oracle(monkeypatch, chunk):
    torch = pytest.importorskip("torch")
    directory = TASKS / "mi355x_vllm_triton_sparse_attn_prefill_ragged"
    h = load_module(directory / "scripts/task_runner.py")
    h._torch = lambda: torch
    h._load_kernel_module = lambda: None
    h._REFERENCE_QUERY_CHUNK = chunk
    # Exercise the actual input generator on CPU, including its seeds and CSR
    # construction. Only redirect its explicit device placement.
    for name in ("randn", "randint", "tensor", "arange"):
        original = getattr(torch, name)
        def cpu_factory(*args, _original=original, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu_factory)
    case = deepcopy(h.RAGGED_CASES[0])
    case["params"].update(num_queries=24, num_heads=2, head_dim=4,
                          nope_head_dim=2, rope_head_dim=2)
    inputs = h._make_ragged(case)
    lengths = inputs["indptr"].diff().tolist()
    assert lengths == case["params"]["row_lengths"] * 2
    assert inputs["indices"].numel() == sum(lengths)
    assert inputs["indices"].dtype == inputs["indptr"].dtype == torch.int32
    expected = torch.zeros_like(inputs["q"])
    for i, width in enumerate(lengths):
        if not width:
            continue
        begin, end = map(int, inputs["indptr"][i:i+2])
        values = inputs["kv"][inputs["indices"][begin:end].long()].double()
        for head in range(inputs["q"].shape[1]):
            scores = values @ inputs["q"][i, head].double() * inputs["scale"]
            expected[i, head] = (torch.softmax(scores, dim=0) @ values).to(expected.dtype)
    torch.testing.assert_close(h._reference(inputs), expected, atol=0.004, rtol=0.004)
    h._assert_close(inputs, expected)
    with pytest.raises(AssertionError):
        h._assert_close(inputs, torch.zeros_like(expected))
    bad = dict(inputs, indptr=inputs["indptr"].clone())
    bad["indptr"][-1] += 1
    with pytest.raises(ValueError, match="CSR offsets"):
        h._reference(bad)


def test_sparse_additional_checks_reject_incorrect_captured_path():
    directory = TASKS / "mi355x_vllm_triton_sparse_attn_prefill_ragged"
    adapter = load_module(directory / "scripts/task_adapter.py")
    h = load_module(directory / "scripts/task_runner.py")
    events = []
    h.run_correctness = lambda: events.append("original checks")
    h._make_ragged = lambda case: {"case": case}
    h._run = lambda inputs: "correct immediate output"
    h._torch = lambda: SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    h._assert_close = lambda inputs, got: events.append(got)
    h._TimedRun = lambda: SimpleNamespace(bound=False)
    def capture(call, **options):
        assert call() == "correct immediate output"
        options["timed_run"].bound = True
        events.append("captured")
    h._benchmark_cuda_graph_or_events = capture
    def reject(inputs, timed):
        assert timed.bound
        raise AssertionError("wrong captured result")
    h._assert_timed_outputs = reject
    with pytest.raises(AssertionError, match="wrong captured result"):
        adapter.run_correctness(h)
    assert events == ["original checks", "correct immediate output", "captured"]


@pytest.mark.parametrize("protected", ["_mxfp8_e4m3_quantize_torch", "_mxfp8_quant_kernel",
                                      "_mxfp8_e4m3_quantize_triton", "mxfp8_e4m3_quantize",
                                      "dequant_mxfp8_to_bf16"])
def test_mxfp8_linear_guard_preserves_input_quantizers(tmp_path, protected):
    from src.harness_guard import (describe_workspace_harness, snapshot_workspace_harness,
                                   verify_workspace_harness)
    directory = TASKS / "mi355x_sglang_triton_mxfp8_linear"
    config = yaml.safe_load((directory / "config.yaml").read_text())
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    edit = config["candidate"]["editable"][0]
    assert edit["scope"] == "symbols" and edit["allow_new_helpers"] is True
    assert edit["symbols"] == ["_mxfp8_linear_kernel", "_run_mxfp8_linear_kernel"]
    source = tmp_path / edit["path"]
    source.parent.mkdir(parents=True)
    source.write_text(f"def {protected}(x): return x\n"
                      "def _mxfp8_linear_kernel(x): return x\n"
                      "def _run_mxfp8_linear_kernel(x): return _mxfp8_linear_kernel(x)\n")
    snapshot = snapshot_workspace_harness(tmp_path)
    assert describe_workspace_harness(tmp_path)["editable_entrypoint_targets"] == {
        edit["path"]: edit["symbols"]}
    source.write_text(source.read_text().replace("def _mxfp8_linear_kernel(x): return x",
                                                "def _mxfp8_linear_kernel(x): return helper(x)")
                      + "def helper(x): return x + 1\n")
    verify_workspace_harness(snapshot)
    source.write_text(source.read_text().replace(f"def {protected}(x): return x",
                                                f"def {protected}(x): return x * 0"))
    with pytest.raises(RuntimeError, match="mxfp8_amd_gfx95.py"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("violation", ["dtype", "shape", "device"])
def test_kimi_moe_comparison_rejects_wrong_output_contract(violation):
    torch = pytest.importorskip("torch")
    directory = TASKS / "mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3"
    h = load_module(directory / "scripts/task_runner.py")
    h._torch = lambda: torch
    expected = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    cos, error = h._moe_deviation(expected, expected)
    assert cos > 0.999 and error == 0
    bad = {"dtype": lambda: expected.float(), "shape": lambda: expected.flatten(),
           "device": lambda: torch.empty_like(expected, device="meta")}[violation]()
    with pytest.raises(AssertionError):
        h._moe_deviation(bad, expected)
    # Captured replay invokes the same contract before comparing numeric error.
    h._perturb_moe_inputs = lambda inputs: None
    h._reference = lambda inputs: expected
    if violation != "device":
        timed = SimpleNamespace(bound=True, outputs=None, rerun=lambda: bad)
        with pytest.raises(AssertionError):
            h._assert_timed_outputs({"id": "contract", "params": {}}, {}, timed)


def test_reduced_ck_checks_do_not_claim_scored_shape_coverage(monkeypatch):
    directory = TASKS / "mi355x_vllm_ck_a8w8_blockscale_gemm"
    adapter = load_module(directory / "scripts/task_adapter.py")
    seen = []
    class Torch:
        cuda = SimpleNamespace(synchronize=lambda: None)
        testing = SimpleNamespace(assert_close=lambda a, b, **kw: seen.append((a, b, kw)))
    harness = SimpleNamespace(
        CASES=[{"id":"full", "params":{"m":7211}}],
        run_correctness=lambda:seen.append("original small checks"),
        _make=lambda case, correctness: (case["params"]["m"], correctness),
        _run=lambda inputs: inputs, _gemm_reference=lambda inputs: inputs,
        _torch=lambda:Torch,
    )
    adapter.run_correctness(harness)
    assert seen == ["original small checks", ((7211, False), (7211, False), {"atol":0.15,"rtol":0.12})]


def test_performance_rejects_missing_stale_or_conflicting_rows():
    evaluator = load_module(DIRECTORIES[0] / "scripts/evaluate.py")
    expected = [{"test_case_id":"x", "params":{"m":8}, "shape":[8],
                 "checks":["correctness","performance"]}]
    valid = {"test_case_id":"x", "shape":[8], "execution_time_ms":0.5,
             "metadata":{"m":8,"benchmark_method":"cuda_graph"}}
    assert evaluator.validate_performance([valid], expected)[0]["execution_time_ms"] == 0.5
    for rows in (None, [], [valid, valid], [dict(valid, test_case_id="other")],
                 [dict(valid, shape=[1])], [dict(valid, execution_time_ms=float("nan"))],
                 [dict(valid, execution_time_ms=0)], [dict(valid, metadata={"m":1,"benchmark_method":"cuda_graph"})],
                 [dict(valid, metadata={"benchmark_method":"host_wall_time"})]):
        with pytest.raises((ValueError, KeyError)):
            evaluator.validate_performance(rows, expected)


def test_multifile_kimi_paths_and_frozen_reference(tmp_path, monkeypatch):
    directory = TASKS / "mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3"
    spec = load_task_spec(directory / "config.yaml", task_id="image_kernel/kimi")
    assert len(spec.candidate.editable) == 5
    assert all(e.path.startswith("aiter/aiter/") for e in spec.candidate.editable)
    assert any(e.path.endswith("kimik3_fp4_tuned_fmoe.csv") for e in spec.candidate.editable)
    adapter = load_module(directory / "scripts/task_adapter.py")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    source = tmp_path / "aiter/aiter/fused_moe.py"
    source.parent.mkdir(parents=True)
    source.write_text("def torch_moe_stage1(): return 42\n")
    asset = tmp_path / "aiter/csrc/include/aiter_enum.h"
    asset.parent.mkdir(parents=True);asset.write_text("header")
    adapter.setup()
    frozen = tmp_path / "scripts/_reference_fused_moe.py"
    source.write_text("def torch_moe_stage1(): return 0\n")
    adapter.setup()
    assert frozen.read_text() == "def torch_moe_stage1(): return 42\n"


def test_module_origin_must_be_the_declared_candidate(tmp_path, monkeypatch):
    adapter = load_module(TASKS / "mi355x_sglang_triton_mxfp8_linear/scripts/task_adapter.py")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "MODULE_BINDINGS", [("fixture_kernel", "candidate.py")])
    (tmp_path / "candidate.py").write_text("def kernel(): return 1\n")
    monkeypatch.setitem(sys.modules, "fixture_kernel", SimpleNamespace(__file__=str(tmp_path / "installed.py")))
    with pytest.raises(RuntimeError, match="outside the declared implementation"):
        adapter.prepare(SimpleNamespace(_configure=lambda:None))


def test_empty_placeholder_and_escaping_candidates_fail(tmp_path, monkeypatch):
    setup = load_module(DIRECTORIES[0] / "scripts/setup_task.py")
    monkeypatch.setattr(setup, "ROOT", tmp_path)
    cfg={"candidate":{"editable":["source/kernel.py"],"entrypoints":[{"file":"source/kernel.py","symbol":"kernel"}]}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    source = tmp_path / "source/kernel.py";source.parent.mkdir()
    for body in ("", "def kernel(): pass\n", "def kernel(): return None\n", "def kernel(): raise NotImplementedError\n"):
        source.write_text(body)
        with pytest.raises(ValueError):setup.verify_sources()
    source.write_text("def kernel(x): return x + 1\n")
    assert setup.verify_sources()
    source.unlink();source.symlink_to(Path(__file__).resolve())
    with pytest.raises(ValueError, match="escapes"):
        setup.verify_sources()


def test_mhc_rejects_missing_output_even_when_prefix_matches(monkeypatch):
    torch = pytest.importorskip("torch")
    h = load_module(TASKS / "mi355x_vllm_tilelang_mhc_fused_post_pre/scripts/task_runner.py")
    monkeypatch.setattr(h,"_torch",lambda:torch)
    output=tuple(torch.ones(1) for _ in range(4))
    monkeypatch.setattr(h,"_mhc_reference",lambda _:output)
    h._assert_mhc_close({},output)
    with pytest.raises(AssertionError,match="all four"):
        h._assert_mhc_close({},output[:2])


def test_mxfp8_rejects_broadcast_output_and_wrong_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    h = load_module(TASKS / "mi355x_sglang_triton_mxfp8_linear/scripts/task_runner.py")
    monkeypatch.setattr(h,"_torch",lambda:torch)
    expected=torch.ones(2,2,dtype=torch.bfloat16)
    monkeypatch.setattr(h,"_reference",lambda _:expected)
    case={"id":"x","params":{"max_relerr":0.06}}
    for wrong in (torch.ones(1,2,dtype=torch.bfloat16),expected.float()):
        with pytest.raises(AssertionError):h._assert_close(case,{},wrong)


def test_real_runner_session_freezes_baseline_and_rejects_wrong_candidate(tmp_path):
    # CPU lifecycle fixture: real migrated envelope runner + setup verification
    # and real TaskSession. The latency is a protocol fixture, not a GPU result.
    workspace = tmp_path / "candidate"
    scripts = workspace / "scripts"
    scripts.mkdir(parents=True)
    for name in ("evaluate.py", "setup_task.py"):
        shutil.copyfile(DIRECTORIES[0] / "scripts" / name, scripts / name)
    cfg = {
        "schema_version": 2,
        "candidate": {"language": "python", "initial_state": "implemented",
                      "editable": ["source/kernel.py"],
                      "entrypoints": [{"file": "source/kernel.py", "kind": "function", "symbol": "kernel"}]},
        "baseline": {"kind": "initial_candidate", "correctness_policy": "required"},
        "evaluation": {"runner": [sys.executable, "scripts/evaluate.py"],
                       "workloads": "workloads.json", "timeout_s": 30},
    }
    (workspace / "config.yaml").write_text(yaml.safe_dump(cfg))
    source = workspace / "source/kernel.py"
    source.parent.mkdir()
    source.write_text("def kernel(x): return x + 1\n")
    cases = [{"test_case_id": "fixture", "params": {"x": 6}, "shape": [1],
              "checks": ["correctness", "performance"]}]
    manifest = {"original_cases": {"correctness": [{"x": 6}], "performance": None}, "cases": cases}
    (workspace / "workloads.json").write_text(json.dumps(manifest))
    (scripts / "task_adapter.py").write_text(
        "def prepare(h): return None\n"
        "def validate_workloads(h): assert h.CASES == [{'x': 6}]\n"
        "def run_correctness(h): h.run_correctness()\n")
    (scripts / "reference_controls.py").write_text(
        "def check_reference(h):\n"
        "    assert 6 + 1 == 7 and 0 != 7\n"
        "    return {'known_answer': 'PASS', 'negative_control': 'PASS'}\n")
    (scripts / "task_runner.py").write_text(
        "from pathlib import Path\nimport runpy\n"
        "ROOT = Path(__file__).resolve().parents[1]\nCASES = [{'x': 6}]\n"
        "def run_compile(): compile((ROOT / 'source/kernel.py').read_text(), 'kernel.py', 'exec')\n"
        "def run_correctness():\n"
        "    assert runpy.run_path(str(ROOT / 'source/kernel.py'))['kernel'](6) == 7\n"
        "def run_performance():\n"
        "    return [{'test_case_id':'fixture', 'shape':[1], 'execution_time_ms':0.01,\n"
        "             'metadata':{'benchmark_method':'cuda_graph','test_fixture_only':True}}]\n")
    spec = load_task_spec(workspace / "config.yaml", task_id="image_kernel/cpu_protocol_fixture")
    session = TaskSession.create(spec, workspace, tmp_path / "state")
    initial = session.validate_initial()
    assert initial.accepted, initial.errors
    for action in ("compile", "correctness", "performance"):
        assert session.candidate_action(action).result.passed
    assert len(list((tmp_path / "state").glob("action-*.json"))) == 7
    # Same-length change avoids relying on Python timestamp/size cache invalidation.
    source.write_text("def kernel(x): return x - 1\n")
    assert session.candidate_action("compile").result.passed
    failed = session.candidate_action("correctness").result
    assert not failed.passed and failed.failure_kind == "evaluation_error"
    assert session._execute("baseline", "correctness", "task_validation").result.passed
    session.verify_baseline_sources()
    assert (session.baseline_workspace / "source/kernel.py").read_text() == "def kernel(x): return x + 1\n"


def test_build_evidence_rejects_unrelated_or_failed_compilation(tmp_path):
    module = load_module(TASKS / "mi355x_vllm_ck_moe_2stage/scripts/source_build.py")
    candidate = tmp_path / "aiter_meta/csrc/ck_moe/target.cu"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("// candidate translation unit\n")
    evidence = module.BuildEvidence(tmp_path, ["aiter_meta/csrc/ck_moe/target.cu"])
    calls = []

    def build(md_name, srcs):
        calls.append(md_name)
        if md_name == "failed":
            raise RuntimeError("compiler failed")
        srcs.clear()  # Real helpers may mutate source lists.
        return "compiled"

    compiler = SimpleNamespace(build_module=build)
    evidence.wrap(compiler, "build_module", ("srcs",), "md_name")
    compiler.build_module("unrelated", [str(tmp_path / "aiter_meta/csrc/flydsl_wrapper.cu")])
    with pytest.raises(RuntimeError, match="did not compile"):
        evidence.finish()
    with pytest.raises(RuntimeError, match="compiler failed"):
        compiler.build_module("failed", [str(candidate)])
    with pytest.raises(RuntimeError, match="did not compile"):
        evidence.finish()
    assert compiler.build_module("ck_actual", [str(candidate)]) == "compiled"
    assert evidence.finish()["compiled_candidate_paths"] == ["aiter_meta/csrc/ck_moe/target.cu"]
    assert evidence.finish()["build_modules"] == ["ck_actual"]


def test_template_build_evidence_accepts_declared_headers_only(tmp_path):
    module = load_module(TASKS / "mi300x_sglang_hip_pa_ragged/scripts/source_build.py")
    candidate = tmp_path / "aiter/csrc/pa.cuh"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("// header\n")
    evidence = module.BuildEvidence(tmp_path, ["aiter/csrc/pa.cuh"])
    compiler = SimpleNamespace(compile_lib=lambda src_file, folder, includes=None, sources=None: "ok")
    evidence.wrap(compiler, "compile_lib", ("includes", "sources"), "folder")
    compiler.compile_lib("rendered C++", "case-1", includes=[str(candidate)])
    assert evidence.finish()["compiled_candidate_paths"] == ["aiter/csrc/pa.cuh"]


@pytest.mark.parametrize("failure", [RuntimeError("missing compiler"), SystemExit(7)])
def test_action_errors_still_emit_failure_envelope(tmp_path, monkeypatch, failure):
    evaluator = load_module(DIRECTORIES[0] / "scripts/evaluate.py")
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    case = {"test_case_id": "x", "checks": ["correctness", "performance"]}
    (tmp_path / "workloads.json").write_text(json.dumps({"cases": [case]}))
    monkeypatch.setitem(sys.modules, "setup_task", SimpleNamespace(verify_sources=lambda: {"kernel.py": "digest"}))
    monkeypatch.setitem(sys.modules, "task_adapter", SimpleNamespace(
        prepare=lambda h: None, validate_workloads=lambda h: None, run_correctness=lambda h: h.run_correctness()))

    def fail():
        raise failure

    report, code = evaluator.invoke(["candidate", "correctness"],
                                     harness_loader=lambda: SimpleNamespace(run_correctness=fail))
    parsed = parse_command_result("ARENA_EVAL_RESULT=" + json.dumps(report),
                                  role="candidate", action="correctness", returncode=code)
    assert not parsed.passed and code != 0
    assert parsed.failure_kind == "evaluation_error"
    assert parsed.cases[0]["status"] == "FAIL"


@pytest.mark.parametrize("directory", [d for d in DIRECTORIES if d.name.startswith("mi300x_")], ids=lambda d:d.name)
def test_old_image_harness_checks_actual_replay_output(directory, monkeypatch):
    torch = pytest.importorskip("torch")
    h = load_module(directory / "scripts/task_runner.py")
    key = "x" if "triton" in directory.name else "query"
    data = torch.tensor([[1., -2.]], dtype=torch.bfloat16)
    case = {key: data, "params": {"dtype": "bfloat16"}}
    monkeypatch.setattr(h, "_run_torch", lambda c: (c[key].float() * 3 + 2).to(torch.bfloat16))
    stale = h._run_torch(case).clone()
    output = stale.clone()
    timed = SimpleNamespace(bound=True, outputs=output)

    def correct_replay():
        assert torch.isnan(output).all()
        output.copy_(h._run_torch(case))
        return output

    timed.rerun = correct_replay
    h._assert_timed_outputs(case, timed)
    torch.testing.assert_close(data, torch.tensor([[-1., 2.]], dtype=torch.bfloat16))
    # It rejects returning a prior answer, even though that tensor is finite.
    data.neg_()
    timed.rerun = lambda: stale
    with pytest.raises(AssertionError):
        h._assert_timed_outputs(case, timed)
    timed.rerun = lambda: output  # Fails if the invocation did not overwrite NaNs.
    with pytest.raises(AssertionError):
        h._assert_timed_outputs(case, timed)


def test_aiter_image_sources_use_qualified_repository_layout():
    selected = [d for d in DIRECTORIES if d.name.startswith("mi355x_vllm_ck_")
                or d.name in {"mi355x_vllm_hip_dynamic_per_tensor_quant",
                              "mi355x_vllm_hip_paged_attention_decode", "mi355x_vllm_triton_unified_attention"}]
    assert len(selected) == 6
    for directory in selected:
        cfg = yaml.safe_load((directory / "config.yaml").read_text())
        by_dest = {source["destination"]: source for source in cfg["workspace"]["sources"]}
        assert by_dest["aiter_meta"]["image_path"] == "/sgl-workspace/aiter"
        assert by_dest["aiter"]["image_path"] == "/sgl-workspace/aiter/aiter"
        if not directory.name.endswith("triton_unified_attention"):
            assert all(p.startswith("aiter_meta/csrc/") for p in cfg["candidate"]["editable"])


@pytest.mark.parametrize("name", ["mi355x_sglang_triton_mxfp8_linear", "mi355x_sglang_triton_mxfp8_grouped_gemm"])
def test_mxfp8_pinned_git_staging_preserves_candidate_on_reentry(name, tmp_path):
    directory = TASKS / name
    shutil.copyfile(directory / "config.yaml", tmp_path / "config.yaml")
    spec = load_task_spec(directory / "config.yaml", task_id="image_kernel/" + name)
    source = spec.to_mapping()["workspace"]["sources"][0]
    assert source == {"kind": "git", "url": "https://github.com/sgl-project/sglang.git",
                      "revision": "3ea875fef48f6f01fa3bddd9e2197ad190cef29d", "destination": "upstream/sglang"}
    package = tmp_path / "upstream/sglang/python/sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# source package\n")
    target = spec.candidate.editable[0].path
    upstream = package / Path(target).relative_to("sglang")
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text("def compute(): return 3\n")
    # The actual pinned package links .clang-format to sgl-kernel in the same
    # declared checkout. Copy its content so the staged package stands alone.
    style = tmp_path / "upstream/sglang/sgl-kernel/.clang-format"
    style.parent.mkdir()
    style.write_text("BasedOnStyle: LLVM\n")
    (package / ".clang-format").symlink_to("../../sgl-kernel/.clang-format")
    stage = load_module(directory / "scripts/materialize_source.py")
    stage.materialize(tmp_path)
    candidate = tmp_path / target
    assert candidate.read_bytes() == upstream.read_bytes()
    staged_style = tmp_path / "sglang/.clang-format"
    assert not staged_style.is_symlink() and staged_style.read_bytes() == style.read_bytes()
    candidate.write_text("def compute(): return 4\n")
    before = {str(p): p.read_bytes() for p in (tmp_path / "sglang").rglob("*") if p.is_file()}
    stage.materialize(tmp_path)
    after = {str(p): p.read_bytes() for p in (tmp_path / "sglang").rglob("*") if p.is_file()}
    assert after == before
    assert candidate.read_text() == "def compute(): return 4\n"
    assert upstream.read_text() == "def compute(): return 3\n"
    # A receipt contains no checkout-local absolute paths or inode identities:
    # a separately copied baseline can safely repeat its own setup as well.
    copied = tmp_path.with_name(tmp_path.name + "-copied")
    shutil.copytree(tmp_path, copied, symlinks=True)
    stage.materialize(copied)
    assert (copied / target).read_bytes() == candidate.read_bytes()


@pytest.mark.parametrize("name", ["mi355x_sglang_triton_mxfp8_linear", "mi355x_sglang_triton_mxfp8_grouped_gemm"])
def test_mxfp8_staging_rejects_external_package_symlink(name, tmp_path):
    directory = TASKS / name
    shutil.copyfile(directory / "config.yaml", tmp_path / "config.yaml")
    package = tmp_path / "upstream/sglang/python/sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# package\n")
    external = tmp_path / "outside.py"
    external.write_text("# outside package\n")
    (package / "escape.py").symlink_to(external)
    stage = load_module(directory / "scripts/materialize_source.py")
    with pytest.raises(ValueError, match="external source symlink"):
        stage.materialize(tmp_path)
    assert not (tmp_path / "sglang").exists()


@pytest.mark.parametrize("name", ["mi355x_sglang_triton_mxfp8_linear", "mi355x_sglang_triton_mxfp8_grouped_gemm"])
@pytest.mark.parametrize("collision", ["unknown", "invalid_receipt", "changed_source", "changed_revision", "symlink", "upstream_escape"])
def test_mxfp8_staging_rejects_unknown_or_changed_source_on_repeat(name, collision, tmp_path):
    directory = TASKS / name
    config_path = tmp_path / "config.yaml"
    shutil.copyfile(directory / "config.yaml", config_path)
    package = tmp_path / "upstream/sglang/python/sglang"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# pinned package\n")
    stage = load_module(directory / "scripts/materialize_source.py")
    destination = tmp_path / "sglang"
    if collision == "unknown":
        destination.mkdir()
        (destination / "__init__.py").write_text("# unrelated existing package\n")
    elif collision == "symlink":
        destination.symlink_to(package, target_is_directory=True)
    else:
        stage.materialize(tmp_path)
        if collision == "invalid_receipt":
            (destination / stage.RECEIPT).write_text("not-json")
        elif collision == "changed_source":
            (package / "__init__.py").write_text("# different upstream bytes\n")
        elif collision == "changed_revision":
            cfg = yaml.safe_load(config_path.read_text())
            cfg["workspace"]["sources"][0]["revision"] = "f" * 40
            config_path.write_text(yaml.safe_dump(cfg))
        elif collision == "upstream_escape":
            external = tmp_path / "outside.py"
            external.write_text("# external\n")
            (package / "escape.py").symlink_to(external)
    before = (destination / "__init__.py").read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        stage.materialize(tmp_path)
    assert (destination / "__init__.py").read_bytes() == before


@pytest.mark.parametrize("name", [
    "mi355x_vllm_ck_a8w8_blockscale_gemm", "mi355x_vllm_ck_cktile_moe_2stage",
    "mi355x_vllm_ck_moe_2stage", "mi355x_vllm_hip_dynamic_per_tensor_quant",
    "mi355x_vllm_hip_paged_attention_decode",
])
def test_hip_package_binding_rejects_installed_dispatch(name, tmp_path, monkeypatch):
    adapter = load_module(TASKS / name / "scripts/task_adapter.py")
    monkeypatch.setattr(adapter, "os", SimpleNamespace(environ=dict(os.environ)))
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    (tmp_path / "aiter_meta/csrc").mkdir(parents=True)
    (tmp_path / "aiter").mkdir()
    shutil.copyfile(TASKS / name / "config.yaml", tmp_path / "config.yaml")
    imported = SimpleNamespace(__file__=str(tmp_path.parent / "installed_aiter/__init__.py"))
    monkeypatch.setattr(adapter, "importlib", SimpleNamespace(import_module=lambda name: imported))
    marker = object()
    monkeypatch.setitem(sys.modules, "source_build", SimpleNamespace(watch=lambda *a, **k: marker))
    harness = SimpleNamespace(_configure=lambda: None)
    with pytest.raises(RuntimeError, match="outside the declared implementation"):
        adapter.prepare(harness)
    imported.__file__ = str(tmp_path / "aiter/__init__.py")
    assert adapter.prepare(harness) is marker


@pytest.mark.parametrize("name", [
    "mi355x_vllm_ck_a8w8_blockscale_gemm", "mi355x_vllm_ck_cktile_moe_2stage",
    "mi355x_vllm_ck_moe_2stage",
])
def test_ck_replay_checks_original_inputs_and_restores_on_failures(name, monkeypatch):
    torch = pytest.importorskip("torch")
    h = load_module(TASKS / name / "scripts/task_runner.py")
    monkeypatch.setattr(h, "_torch", lambda: torch)
    if h.OPERATOR == "a8w8_blockscale_gemm":
        inputs = {"shape": [2, 2, 128], "x": torch.ones(2,128),
                  "weight": torch.full((2,128), 2.),
                  "x_scale": torch.full((2,1), .1), "w_scale": torch.full((1,1), .1)}
        reference = h._gemm_reference
        mutate = "weight"
    else:
        inputs = {"hidden": torch.tensor([[1.,2.],[-3.,4.]], dtype=torch.bfloat16)}
        reference = lambda inp: (inp["hidden"].float() * 2).to(torch.bfloat16)
        monkeypatch.setattr(h, "_moe_reference", reference)
        mutate = "hidden"
    initial = {key: value.clone() for key,value in inputs.items() if isinstance(value, torch.Tensor)}
    check = h._prepare_timed_check(inputs)
    stale = reference(inputs).clone()
    output = stale.clone()
    timed = SimpleNamespace(bound=True, outputs=output)
    def replay():
        assert torch.isnan(output).all()
        output.copy_(reference(inputs))
        return output
    def restored():
        for key,value in initial.items(): torch.testing.assert_close(inputs[key],value,rtol=0,atol=0)
    timed.rerun = replay
    h._assert_timed_outputs(inputs, timed, check)
    restored()
    for bad in (stale, torch.zeros_like(output), torch.full_like(output, float("nan"))):
        output.copy_(stale)
        timed.rerun = lambda bad=bad: bad
        with pytest.raises(AssertionError): h._assert_timed_outputs(inputs, timed, check)
        restored()
    # A wrong original measured output cannot be rescued by a correct rerun.
    output.zero_();timed.rerun = replay
    with pytest.raises(AssertionError): h._assert_timed_outputs(inputs, timed, check)
    restored()
    # Neither original-timing nor replay input writes can contaminate the oracle.
    inputs[mutate].zero_();output.copy_(stale)
    with pytest.raises(AssertionError, match="Readonly input"):
        h._assert_timed_outputs(inputs, timed, check)
    restored()
    def corrupt_replay():
        result = replay();inputs[mutate].zero_();return result
    output.copy_(stale);timed.rerun = corrupt_replay
    with pytest.raises(AssertionError, match="Readonly input"):
        h._assert_timed_outputs(inputs, timed, check)
    restored()
    def fail_replay(): raise RuntimeError("replay failed")
    output.copy_(stale);timed.rerun = fail_replay
    with pytest.raises(RuntimeError, match="replay failed"):
        h._assert_timed_outputs(inputs, timed, check)
    restored()
    with pytest.raises(AssertionError, match="BF16"):
        h._assert_output_contract(inputs, reference(inputs).float())
    with pytest.raises(AssertionError, match="shape"):
        h._assert_output_contract(inputs, reference(inputs).flatten())


KDA_TASK = TASKS / "mi355x_vllm_triton_kda_linear_attn_kimi_k3"


def test_kda_public_source_staging_preserves_edits_and_rejects_collision(tmp_path):
    shutil.copyfile(KDA_TASK / "config.yaml", tmp_path / "config.yaml")
    spec = load_task_spec(tmp_path / "config.yaml", task_id="image_kernel/kda")
    source = spec.to_mapping()["workspace"]["sources"][0]
    assert source == {"kind":"git", "url":"https://github.com/vllm-project/vllm.git",
                      "revision":"000c7df9ffd3e470980fd4cd6b8ec1b0585500ff",
                      "destination":"upstream/vllm"}
    upstream = tmp_path / "upstream/vllm/vllm"
    upstream.mkdir(parents=True)
    (upstream / "__init__.py").write_text("# public package\n")
    target = Path(spec.candidate.editable[0].path).relative_to("vllm")
    (upstream / target).parent.mkdir(parents=True)
    (upstream / target).write_text("def kernel(): return 1\n")
    stage = load_module(KDA_TASK / "scripts/materialize_source.py")
    stage.materialize(tmp_path)
    candidate = tmp_path / "vllm" / target
    assert candidate.read_bytes() == (upstream / target).read_bytes()
    candidate.write_text("def kernel(): return 2\n")
    stage.materialize(tmp_path)
    assert candidate.read_text() == "def kernel(): return 2\n"
    (upstream / target).write_text("def kernel(): return 3\n")
    with pytest.raises(FileExistsError, match="source identity"):
        stage.materialize(tmp_path)


def test_kda_operator_loader_uses_real_leaves_without_model_initializers(tmp_path, monkeypatch):
    import importlib
    loader = load_module(KDA_TASK / "scripts/operator_imports.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in (*loader.PACKAGES, "vllm", "vllm.third_party", "vllm.third_party.flash_linear_attention"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        monkeypatch.delitem(sys.modules, name, raising=False)
        path = tmp_path.joinpath(*name.split("."));path.mkdir(parents=True,exist_ok=True)
        (path / "__init__.py").write_text("raise RuntimeError('unrelated model initializer')\n" if name in loader.PACKAGES else "# namespace root\n")
    leaf = tmp_path / "vllm/models/kimi_k3/amd/ops/third_party/kda"
    leaf.mkdir();(leaf / "__init__.py").write_text("from .chunk import entry\n")
    (leaf / "chunk.py").write_text("def entry(): return 'public operator'\n")
    for name in ("vllm.models.kimi_k3.amd.ops.third_party.kda", "vllm.models.kimi_k3.amd.ops.third_party.kda.chunk"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        monkeypatch.delitem(sys.modules,name,raising=False)
    loader.configure_operator_packages(tmp_path)
    module = importlib.import_module("vllm.models.kimi_k3.amd.ops.third_party.kda")
    assert module.entry() == 'public operator'
    assert Path(module.entry.__code__.co_filename) == leaf / "chunk.py"
    outside = tmp_path.parent / "external-kda";outside.mkdir()
    monkeypatch.setitem(sys.modules,loader.PACKAGES[0],SimpleNamespace(__path__=[str(outside)]))
    with pytest.raises(RuntimeError,match="another source"):
        loader.configure_operator_packages(tmp_path)


def _kda_scalar_fixture(torch, mode):
    state = torch.tensor([[[[7.]]],[[[3.]]]],dtype=torch.float32) if mode == 'packed_decode' else torch.tensor([[[[3.]]]])
    return {"cfg":{"params":{"min_cosine":.999,"max_rel_err":.03}},
            "mode":mode,"H":1,"D":1,"scale":1.,"num_seqs":1,"seq_len":1,"total_t":1,
            "segments":[(0,1)],"seg_state0":[torch.tensor([[[3.]]],dtype=torch.float64)],
            "q":torch.ones(1,1,1,1,dtype=torch.bfloat16),"k":torch.ones(1,1,1,1,dtype=torch.bfloat16),
            "v":torch.tensor([[[[2.]]]],dtype=torch.bfloat16),
            "mixed_qkv":torch.tensor([[1.,1.,2.]],dtype=torch.bfloat16),
            "raw_g":torch.zeros(1,1,1,1),"raw_beta":torch.zeros(1,1,1),
            "A_log":torch.zeros(1),"dt_bias":torch.zeros(1),"state":state,
            "state_indices":torch.tensor([1],dtype=torch.int32)}


@pytest.mark.parametrize('mode',['chunk','packed_decode'])
def test_kda_state_reference_and_output_contract(mode, monkeypatch):
    import math
    torch = pytest.importorskip('torch')
    h = load_module(KDA_TASK / 'scripts/task_runner.py');monkeypatch.setattr(h,'_torch',lambda:torch)
    inp = _kda_scalar_fixture(torch,mode);before = inp['state'].clone()
    out, states = h._golden(inp,return_state=True)
    n=math.sqrt(1.+1e-6);decayed=3.*math.exp(-2.5)
    expected_state=decayed+(2.-decayed/n)*.5/n
    torch.testing.assert_close(states[0],torch.tensor([[[expected_state]]],dtype=torch.float64))
    torch.testing.assert_close(out,torch.tensor([[[[expected_state/n]]]],dtype=torch.float64))
    if mode=='packed_decode':
        inp['state'][1].copy_(states[0]);result_state=inp['state']
    else:result_state=torch.stack(states).float()
    result=(out.bfloat16(),result_state)
    h._assert_state_result(inp,result,states,before)
    with pytest.raises(AssertionError,match='BF16'):
        h._assert_state_result(inp,(out.float(),result_state),states,before)
    result_state.zero_()
    with pytest.raises(AssertionError):h._assert_state_result(inp,result,states,before)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_hip_paged_cache_layout_matches_scalar_slot_mapping(dtype_name):
    import torch
    h = load_module(TASKS / "mi355x_vllm_hip_paged_attention_decode/scripts/task_runner.py")
    dtype = getattr(torch, dtype_name)
    heads, dim, block_size, blocks = 2, 16, 4, 5
    x = 16 // torch.empty((), dtype=dtype).element_size()
    slots = torch.tensor([9, 2, 15, 4, 0])
    keys = torch.arange(5 * heads * dim).reshape(5, heads, dim).to(dtype)
    values = -keys - 1
    key_cache = torch.full((blocks, heads, dim // x, block_size, x), -99., dtype=dtype)
    value_cache = torch.full((blocks, heads, dim, block_size), -99., dtype=dtype)
    inputs = dict(num_kv_heads=heads, head_size=dim, block_size=block_size,
                  slot_mapping=slots, key=keys, value=values,
                  key_cache=key_cache, value_cache=value_cache)
    expected_key, expected_value = key_cache.clone(), value_cache.clone()
    for token, slot in enumerate(slots.tolist()):
        for head in range(heads):
            for d in range(dim):
                expected_key[slot // block_size, head, d // x, slot % block_size, d % x] = keys[token, head, d]
                expected_value[slot // block_size, head, d, slot % block_size] = values[token, head, d]
    h._fill_kv_cache(inputs)
    torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)
    # Refresh really replaces populated slots; untouched padding stays untouched.
    inputs["key"] = keys + 5
    h._fill_kv_cache(inputs)
    assert not torch.equal(key_cache, expected_key)
    assert torch.equal(value_cache, expected_value)
    assert torch.equal(key_cache[4], expected_key[4])


@pytest.mark.parametrize("quant,expected_block,expected_split", [(1,64,3),(2,32,3),(3,32,0)])
def test_ck_dispatch_binds_both_stages_and_keeps_quant_contract(monkeypatch, quant, expected_block, expected_split):
    from types import SimpleNamespace
    h = load_module(TASKS / "mi355x_vllm_ck_moe_2stage/scripts/ck_dispatch.py")
    qtype = SimpleNamespace(per_1x128=1, per_1x32=2)
    class EnumValue(int):
        @property
        def value(self): return int(self)
    stage1, stage2 = lambda: None, lambda: None
    original_meta = SimpleNamespace(stage1="flydsl",stage2="asm",run_1stage=True)
    calls = []
    def impl(*args, **kw):
        selected = kw["_metadata_transform"](original_meta)
        assert selected.stage1.func is stage1 and selected.stage2.func is stage2
        assert selected.run_1stage is False
        assert selected.block_m == expected_block and selected.ksplit == expected_split
        assert selected.stage1.keywords["quant_type"] == quant
        assert selected.stage2.keywords["kernelName"] == ""
        assert kw["_q_dtype_a"] == "original-activation-dtype"
        assert kw["w1_scale"] is w1_scale and kw["w2_scale"] is w2_scale
        assert args == (hidden,w1,w2,weights,ids)
        calls.append(selected)
        return "ck-result"
    aiter = SimpleNamespace(QuantType=qtype,ck_moe_stage2_fwd=stage2)
    moe = SimpleNamespace(get_padded_M=lambda m:m,get_inter_dim=lambda *a:(8,4096,1792),
        get_block_size_M=lambda *a:32,get_ksplit=lambda *a:3,use_nt=lambda *a:False,
        MOEMetadata=SimpleNamespace,ck_moe_stage1=stage1,_fused_moe_impl=impl)
    modules={"aiter":aiter,"aiter.fused_moe":moe}
    monkeypatch.setattr(h.importlib,"import_module",lambda name:modules[name])
    hidden, w1, w2, ids = [SimpleNamespace(shape=s) for s in ((64,4096),(8,3584,4096),(8,4096,1792),(64,2))]
    weights,w1_scale,w2_scale = object(),object(),object()
    assert h.run_ck_moe(hidden,w1,w2,weights,ids,w1_scale=w1_scale,w2_scale=w2_scale,
        quant_type=EnumValue(quant),activation=EnumValue(0),dtype="bf16",
        activation_dtype="original-activation-dtype") == "ck-result"
    assert len(calls)==1 and original_meta.stage1=="flydsl"
    # Missing runtime hook must fail, never execute the broad dispatcher.
    del moe._fused_moe_impl
    with pytest.raises(AttributeError):
        h.run_ck_moe(hidden,w1,w2,weights,ids,w1_scale=w1_scale,w2_scale=w2_scale,
            quant_type=EnumValue(quant),activation=EnumValue(0),dtype="bf16",activation_dtype="a")


@pytest.mark.parametrize('mode', ['chunk', 'packed_decode'])
def test_kda_measured_state_requires_observation_and_rejects_corruption(mode, monkeypatch):
    torch = pytest.importorskip('torch')
    h = load_module(KDA_TASK / 'scripts/task_runner.py')
    monkeypatch.setattr(h, '_torch', lambda: torch)
    inp = _kda_scalar_fixture(torch, mode)
    check = h._prepare_timed_check(inp)
    initial = {k: v.clone() for k, v in check['snapshots'].items()}
    repeats = 3
    metadata = {'benchmark_effective_repeats': repeats}

    def run_math():
        out, states = h._repeated_reference(inp, inp['state'].clone(), repeats)
        if mode == 'packed_decode':
            inp['state'][inp['state_indices'].long()] = torch.stack(states).float()
            state = inp['state']
        else:
            state = torch.stack(states).float()
        return (out.bfloat16(), state)

    def measured():
        for k, value in initial.items(): inp[k].copy_(value)
        h._observe_timed_sample(inp, check, repeats)
        return SimpleNamespace(bound=True, outputs=run_math(), rerun=run_math)

    def restored():
        for key, value in initial.items():
            assert torch.equal(inp[key], value), key

    timed = measured()
    h._assert_timed_outputs(inp, timed, metadata, check)
    restored()
    timed = measured()
    timed.outputs[0].zero_()
    with pytest.raises(AssertionError): h._assert_timed_outputs(inp, timed, metadata, check)
    restored()
    timed = measured()
    inp['raw_beta'].fill_(3)
    with pytest.raises(AssertionError, match='Readonly'):
        h._assert_timed_outputs(inp, timed, metadata, check)
    restored()
    timed = measured()
    def corrupt_replay():
        result = run_math(); inp['raw_g'].fill_(9); return result
    timed.rerun = corrupt_replay
    with pytest.raises(AssertionError, match='Readonly'):
        h._assert_timed_outputs(inp, timed, metadata, check)
    restored()
    timed = measured()
    def fail_replay(): raise RuntimeError('GPU replay failed')
    timed.rerun = fail_replay
    with pytest.raises(RuntimeError, match='replay failed'):
        h._assert_timed_outputs(inp, timed, metadata, check)
    restored()
    if mode == 'packed_decode':
        timed = measured(); check['sample'] = None
        with pytest.raises(AssertionError, match='did not observe'):
            h._assert_timed_outputs(inp, timed, metadata, check)
        restored()
        timed = measured()
        with pytest.raises(AssertionError, match='replay count'):
            h._assert_timed_outputs(inp, timed, {'benchmark_effective_repeats': 2}, check)
        restored()
