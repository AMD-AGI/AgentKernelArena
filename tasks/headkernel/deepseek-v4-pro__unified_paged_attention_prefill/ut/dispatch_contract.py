"""Only captured native dispatch branches are exposed as editable GPU bodies."""
OPUS_NAME = "opus_moe1_afp8_wfp4_bf16_t64x384_gs_qgb3_aff64_k1lead_fp8"


def validate_dispatch(manifest):
    for case in manifest['cases']:
        if manifest['seam'] == 'moe1_prefill':
            controls = case['scalars']['arguments']
            if controls['kernelName'] != OPUS_NAME or controls['block_m'] != 64:
                raise ValueError('Case dispatches outside the captured Opus group-split specialization')
        elif manifest['seam'] == 'mla_prefill':
            shape = case['tensors']['arg.q']['shape']
            if len(shape) != 3 or shape[1] != 16:
                raise ValueError('Case dispatches outside the captured H=16 prefill specialization')
        else:
            raise ValueError('Unsupported captured dispatch seam')
    return True
