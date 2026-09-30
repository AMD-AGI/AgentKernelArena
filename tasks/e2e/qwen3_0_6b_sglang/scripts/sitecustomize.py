"""Protected server integration; activated only for the serving process."""
import os

if os.environ.get("AKA_INSTALL_KERNEL") == "1":
    import functools
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    import importlib.abc
    import importlib.machinery
    import sys

    def wrap(name, candidate, source, digest):
        function = getattr(candidate, name)
        seen = set()

        @functools.wraps(function)
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            # First actual invocation in each rank, outside timed steady state.
            key = (os.getpid(), name)
            if key not in seen:
                seen.add(key)
                import torch
                import dataclasses
                from sglang.srt.server_args import get_global_server_args
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                evidence = dict(pid=os.getpid(), rank=rank, function=name,
                                source=str(source), sha256=digest,
                                shape=list(args[0].shape), device=str(args[0].device),
                                server_settings=dataclasses.asdict(get_global_server_args()),
                                runtime_settings={k:v for k,v in sorted(os.environ.items())
                                    if k.startswith(('SGLANG_', 'VLLM_', 'HIP_', 'ROCR_', 'HSA_',
                                                     'RCCL_', 'NCCL_', 'CUDA_', 'AITER_', 'TORCH_',
                                                     'PYTORCH_', 'FLYDSL_', 'AMD_'))})
                with open('/task/rank_calls.jsonl', 'a') as stream:
                    stream.write(json.dumps(evidence) + '\n')
            return result
        return call

    def install(layernorm):
        source = Path('/task/source/rmsnorm.py')
        spec = importlib.util.spec_from_file_location('aka_candidate', source)
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = candidate
        spec.loader.exec_module(candidate)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for name in ('rms_norm', 'fused_add_rms_norm'):
            setattr(layernorm, name, wrap(name, candidate, source, digest))

    class KernelFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'sglang.srt.layers.layernorm':
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                raise ImportError('SGLang layernorm module is unavailable')
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, module_spec):
                    return original.create_module(module_spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    install(module)

            spec.loader = Loader()
            return spec

    # Registration does not import the framework. Compiler/helper subprocesses
    # inherit this hook safely without starting another dependency build.
    sys.meta_path.insert(0, KernelFinder())
