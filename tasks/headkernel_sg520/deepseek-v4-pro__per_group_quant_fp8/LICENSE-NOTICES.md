Native source and support headers were copied from the AITER source bundled in the pinned runtime image. AITER's MIT license is reproduced in `LICENSE`; original copyright and SPDX notices remain in the copied files.

Some bundled headers, including `ut/native/include/quant_utils.cuh`, retain Apache License 2.0 notices for AMD and vLLM-derived code. The complete license is included at `LICENSES/Apache-2.0.txt`. File-level notices govern those components.

Packaging does not change the native kernel or support-header bodies. The narrow `quant_entry_pybind.cu` binding preserves the production native export under a separate module name, and its attribution remains in the file. Runtime compiler/toolkit dependencies are supplied by the pinned image and are not redistributed as binary artifacts in this package.
