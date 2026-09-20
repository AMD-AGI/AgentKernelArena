# Public runtime manifest evidence

These are the exact manifest response bytes retrieved from the public
`registry-1.docker.io/v2/rocm/hyperloom/manifests/sha256:<filename>` endpoint.
Do not reformat the JSON: its SHA-256 must equal its filename. Each manifest's
`config.digest` binds the pinned registry manifest to the pinned config digest
declared by the task runtime profiles. No image layer payload is included.

The host identity verifier uses this evidence only when Docker reports the
manifest digest as its engine image ID. Docker's containerd image store does
this in [its image inspection implementation](https://github.com/moby/moby/blob/master/daemon/containerd/image_inspect.go).
Acceptance also requires the selected repository's pinned RepoDigest and a
matching local Descriptor, including media type and manifest byte size. These
records do not establish GPU qualification or equivalence to historical images.
