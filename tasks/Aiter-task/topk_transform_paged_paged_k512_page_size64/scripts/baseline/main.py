"""Build the task-owned production HIP kernel once, before GPU timing."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import os
import tempfile


@lru_cache(maxsize=1)
def _module():
    import torch
    from torch.utils.cpp_extension import get_default_build_root, load

    if not torch.version.hip:
        raise RuntimeError("The production Top-k implementation requires ROCm")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise RuntimeError(f"The production Top-k task requires gfx950, got {arch}")
    source = Path(__file__).with_name("topk_kernel.cu").read_bytes()
    digest = sha256(source).hexdigest()[:16]
    runtime = sha256(f"{torch.__version__}:{torch.version.hip}:{arch}".encode()).hexdigest()[:8]
    name = f"aka_paged_topk_{digest}_{runtime}"
    build = Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root())) / name
    build.mkdir(parents=True, exist_ok=True)
    # hipify writes next to the input source. Stage an atomic cache copy so it
    # never modifies protected task files or a read-only source snapshot.
    staged = build / "topk_kernel.cu"
    if not staged.is_file() or staged.read_bytes() != source:
        with tempfile.NamedTemporaryFile(dir=build, delete=False) as temporary:
            temporary.write(source)
            temporary_path = Path(temporary.name)
        temporary_path.replace(staged)
    return load(name=name, sources=[str(staged)], build_directory=str(build),
                extra_cuda_cflags=["-DUSE_ROCM", "--offload-arch=gfx950"],
                with_cuda=True, verbose=False)


def run(scores, seq_lens, metadata, page_size, page_tables, out_page_indices):
    # metadata stays read-only in the task ABI; this HIP kernel reads lengths directly.
    _module().run(scores, seq_lens, page_tables, out_page_indices, page_size, None)
