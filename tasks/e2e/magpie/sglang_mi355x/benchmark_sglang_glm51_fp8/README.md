# Whole-model kernel optimization from a Magpie workload

Optimize any compute operation used by this model, including fusion across
operations. Add Python, Triton, HIP/C++ or supported GPU implementation sources
under `source/`. Declare replacements in `source/patches.json` using module,
symbol, file and callable fields. Available kernel libraries must come from the
pinned image; scoring does not install new packages. The original empty manifest runs the pinned
production model and earns no optimization credit.

Weights, precision, numerical semantics, TP/EP/PP, engine flags, environment,
speculative-decoding policy, request corpus, image inputs, concurrency and
measurement settings are fixed by the protected task and runtime lock. Changing
any of them cannot count as kernel optimization. Use general implementations;
benchmark detection, cached/precomputed answers, reduced work and input-specific
answers are forbidden. Dispatch on legitimate tensor shapes/layouts is allowed.

The whole model is the optimization scope. There is no fixed inventory of
eligible operators and no requirement to preserve their existing boundaries.
This task does not enumerate a unit test for each kernel or require the agent
to supply a new oracle for every fusion. Any historical operator-specific
examples are starting points, not a restriction to selected kernels.
Engine configuration and the evaluator's configuration observation APIs are
protected control-plane modules; they are not compute replacement targets.
Profiling actual kernel costs and workloads is recommended; the agent chooses
its tools and no profiling report is required.

The framework reconstructs each action from trusted files, installs only the
declared compute bundle, verifies actual execution and runtime settings, checks
the task-declared model outputs/numerics and quality against an independent reference, and measures
at least three fresh baseline/candidate pairs. A source-bound independent
`candidate_verifier` agent also checks implementation changes, preserved
semantics, generality and benchmark integrity against frozen source and
execution evidence. The optimizing agent cannot approve its own submission;
failed or inconclusive review blocks acceptance. The verifier cannot waive the
framework's numerical or measurement gates.
Incomplete evidence rejects the candidate. These finite checks do not prove
correctness for every possible input or establish a hostile-code sandbox.
The protected lock states the reference policy. Native-token-only tasks declare either exact equality against a fresh original
server or an independent FP32 teacher-forced greedy-token policy with a frozen
regret bound. Both check every probed generated token; neither claims engine
logprob equivalence. Failed checks never switch policies. The FP32 policy
requires the complete reference model to fit the declared evaluation hardware.

For native trace workloads, read `request_manifest.input_token_policy` in the
protected lock. Under `native_reconstruction`, nominal input lengths count the
loader's token budget before decoded text is re-encoded and chat-formatted;
actual wire counts can be larger or smaller. The fixed native client controls
request construction, actual usage counters must agree, and output lengths
remain exact. Counts alone do not independently verify every HTTP payload.
Trace throughput retains the native phase observation window, including idle
and terminal drain time; cancelled work never counts as a completed request.

Run task-owned checks with:

```bash
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

These commands require the trusted serving executor supplied by the Arena
Docker runner. Configuration and all files outside `source/` are protected.
Generated task packages remain self-contained; their evaluator helpers are
materialized from the maintained shared implementations.
