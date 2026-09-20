# Extract head-kernel archive contracts locally

The shipped generated-input tasks use their task-local compact JSON, source
provenance, and protected generator and reference helpers. They do not run
archive extraction as part of correctness or performance evaluation.

For development, `src/tools/extract_head_kernel_contract.py` can extract a
structural contract from a local original tensor archive. Select the existing
task, archive, output file, and model family explicitly:

```bash
python3 src/tools/extract_head_kernel_contract.py \
  --family minimax \
  --task tasks/head_kernels/<model>/<workload>/<image>/<kernel> \
  --archive local_archives/reference_io.pt \
  --output extracted/generated_cases.json
```

Supported families are `minimax`, `kimi`, `deepseek`, `qwen`, and `glm`. The command uses
the original archive SHA-256 recorded in `ut/meta.json`, including its
`archival_capture` section. It verifies the file before loading it with
`weights_only=True`, `map_location="cpu"`, and memory mapping. The inode remains
open throughout extraction, and changes during the read cause an error. This
requires a PyTorch version supporting the archive's dtypes and a Linux system
with `/proc/self/fd`; no SSH host, cluster mount, external repository, or download
endpoint is assumed.

When a capture has positional arguments, the command looks for the declared
callable's argument names in the task source. Use `--positional-names` followed
by the names in declaration order if that callable is provided by a runtime
package instead. The family-specific extractors live under
`src/tools/head_kernel_archives/` and can also be imported by developer tooling.

The output is a new local JSON file. Existing outputs and task contracts are
never overwritten. Extraction retains structural details such as shapes,
strides, dtype, storage sharing, scalar arguments, and routing or page indices;
numeric arrays are represented by generator recipes. Review an extracted
contract and its source provenance before integrating it into a task. Updating
a task contract still requires its correctness, benchmark, and validator checks.
Extraction success is not GPU qualification.

Historical conversion reports and machine-specific transport scripts are
research artifacts, outside the shipping tree. The authoritative runtime
contract and archival hashes remain inside each task. CPU regressions load the
same task-local helper files that run in isolated workspaces. Third-party API
test excerpts and their pinned provenance belong in `tests/fixtures/`.
