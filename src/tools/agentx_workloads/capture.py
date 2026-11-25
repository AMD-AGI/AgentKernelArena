# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Capture one model's AgentX serving workload on a GPU node (SGLang in Docker).

Each *leg* starts a fresh SGLang server container, replays the corpus from a
client container (``replay.py``), takes torch-profiler windows in steady state
through SGLang's HTTP profiler and stops both containers. Two legs are run by
default:

``graph``
    The production configuration (decode CUDA graphs on). Its scheduler log is
    the observed workload: every prefill batch and decode iteration with batch
    size and token counts. The ROCm torch profiler does not record the device
    kernels of most HIP-graph replays, so this leg's traces measure prefill but
    not decode.
``eager``
    The same server with ``--disable-cuda-graph``. Every decode kernel is then
    an ordinary launch that the profiler records with the shapes of its CPU op,
    so the census measures decode shapes and GPU times instead of rescaling a
    few recorded replays. Kernel device times do not depend on graph replay;
    eager decode runs the unpadded batch, so ``propose`` pads decode shapes to
    the graph leg's captured batch sizes.

Shape hooks (``--shape-hooks``, ``shape_hooks.py``) give kernels that Python wrappers launch without a
recorded op (Triton, Gluon, FlyDSL) a recorded ``pyhook::<wrapper>`` scope with the call's input shapes, so
the census can read their shapes; the server container loads them through ``PYTHONPATH``.

The orchestrator itself needs only the standard library and ``docker``. Every
container it starts is named ``aka-agentx-<capture>-<leg>-{server,client}``
and is removed when the leg ends. Outputs are written under
``<out>/<capture>/<leg>/``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path

from . import shape_hooks

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_DIR = Path(__file__).resolve().parent
HOOKS_MOUNT = "/opt/aka-shape-hooks"  # the tool directory, read-only in the server container
CONTAINER_PREFIX = "aka-agentx-"
LEGS = {"graph": [], "eager": ["--disable-cuda-graph"]}
REPLAY_MAX_OUT = 16384  # output tokens per request are capped at this in the replay
# A server that died or hung: refused or dropped connections, timeouts, malformed replies.
HTTP_ERRORS = (urllib.error.URLError, OSError, HTTPException)
GRAPH_BS_RE = re.compile(r"Capture (?:target decode )?cuda graph begin.*?bs=\[([0-9, ]+)\]", re.I)


def utc(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[capture {utc(time.time())}] {msg}", flush=True)


def corpus_traces(corpus: str) -> tuple[Path, str, str]:
    """(traces.jsonl, corpus id, revision) of a corpus given as the file, a snapshot or an HF dataset cache directory.

    The id and revision come from the HF cache layout (datasets--<org>--<name>/snapshots/<revision>); the
    returned path is resolved, because snapshot files are symlinks into the cache's blob store.
    """
    path = Path(corpus).absolute()
    if path.is_file():
        traces = path
    else:
        found = sorted(glob.glob(str(path / "traces.jsonl"))) or sorted(
            glob.glob(str(path / "snapshots" / "*" / "traces.jsonl")))
        if len(found) != 1:
            raise SystemExit(f"capture: expected one traces.jsonl under {corpus}, found {found}")
        traces = Path(found[0])
    revision = traces.parent.name if traces.parent.parent.name == "snapshots" else "unknown"
    name = next(("/".join(part.split("--")[1:3]) for part in traces.parts if part.startswith("datasets--")),
                traces.parent.name)
    return traces.resolve(), name, revision


def model_mount(model_path: str) -> str:
    """Directory to mount for a model: the HF repo directory, so snapshot symlinks into blobs/ resolve."""
    if "/snapshots/" in model_path:
        return model_path.split("/snapshots/")[0]
    return model_path


DEFAULT_VOCAB_HI = 150000


def prompt_vocab_hi(model_path: str) -> int:
    """Exclusive upper token id for synthetic prompts: below the model's vocabulary and its first added
    (special or media) token above the ordinary range, at most ``DEFAULT_VOCAB_HI``."""
    bound = DEFAULT_VOCAB_HI
    root = Path(model_path)
    try:
        config = json.loads((root / "config.json").read_text())
        vocab = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
        if isinstance(vocab, int) and vocab > 1000:
            bound = min(bound, vocab)
    except (OSError, ValueError):
        pass
    try:
        added = json.loads((root / "tokenizer_config.json").read_text()).get("added_tokens_decoder") or {}
        specials = [int(k) for k in added if str(k).isdigit() and int(k) >= 2000]
        if specials:
            bound = min(bound, min(specials))
    except (OSError, ValueError):
        pass
    return bound


def model_mounts(model_path: str) -> list[str]:
    """Directories the server must see for a model: its HF repo directory, plus the shared blob store that
    the snapshot's files resolve into when it lies outside the repo (a deduplicated cache links
    ``blobs/<hash>`` on to a common store such as ``<cache root>/blobs/<xx>/<hash>``)."""
    repo = model_mount(model_path)
    dirs = [repo]
    repo_real = os.path.realpath(repo)
    outside = set()
    snapshot = Path(model_path)
    if snapshot.is_dir():
        for f in snapshot.iterdir():
            if f.is_symlink():
                target = os.path.realpath(f)
                if not target.startswith(repo_real + os.sep):
                    outside.add(os.path.dirname(target))
    if outside:
        root = os.path.commonpath(sorted(outside))
        dirs += [root] if len(Path(root).parts) > 3 else sorted(outside)
    return dirs


def mount_args(model_path: str, out_root: str, extra: list[str]) -> list[str]:
    """``-v`` arguments of the server container; an explicit ``--mount`` wins over an automatic one."""
    explicit = {m.split(":")[1] if ":" in m else m for m in extra}
    args = []
    for d in model_mounts(model_path):
        if d not in explicit:
            args += ["-v", f"{d}:{d}:ro"]
    if out_root not in explicit:
        args += ["-v", f"{out_root}:{out_root}"]
    for m in extra:
        args += ["-v", m]
    return args


def docker(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def running(name: str) -> bool:
    out = docker("inspect", "-f", "{{.State.Running}}", name, check=False, capture=True)
    return out.returncode == 0 and out.stdout.strip() == "true"


def http(url: str, body: dict | None = None, timeout: float = 30) -> bytes:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def port_answers(base_url: str) -> bool:
    """Whether something already serves ``base_url`` (another server on the port would answer our checks)."""
    try:
        http(base_url + "/health", timeout=5)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError):
        return False


def wait_healthy(base_url: str, server: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not running(server):
            raise RuntimeError(f"server container {server} exited before it became healthy")
        try:
            http(base_url + "/health", timeout=10)
            return
        except HTTP_ERRORS:
            time.sleep(10)
    raise RuntimeError(f"server not healthy after {timeout:.0f} s")


def finalized_ranks(directory: Path, tp: int) -> set[int]:
    """TP ranks whose trace in ``directory`` is a complete gzip stream (``-TP-<r>[-DP-<d>...].trace.json.gz``)."""
    ranks = set()
    for path in directory.glob("*-TP-*.trace.json.gz"):
        m = re.search(r"-TP-(\d+)(?:-[A-Z]+-\d+)*\.trace\.json\.gz$", path.name)
        if m and subprocess.run(["gzip", "-t", str(path)], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0:
            ranks.add(int(m.group(1)))
    return ranks


def profile_window(base_url: str, directory: Path, tp: int, steps: int, server: str,
                   finalize_timeout: float, exit_wait: float = 60) -> dict:
    """One SGLang profiler window of ``steps`` forward steps; waits for every rank's trace.

    A server that cannot be reached is not an error of the capture: the window gets status ``server exited``
    (the container stopped within ``exit_wait`` seconds; ``run_leg`` records the crash and restarts) or
    ``server unreachable`` (it still runs but does not answer)."""
    directory.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        http(base_url + "/start_profile", {"output_dir": str(directory), "num_steps": steps,
                                           "record_shapes": True, "with_stack": False,
                                           "activities": ["CPU", "GPU"]}, timeout=60)
    except HTTP_ERRORS as exc:
        deadline = time.time() + exit_wait
        while running(server) and time.time() < deadline:
            time.sleep(5)
        return {"dir": str(directory), "start": utc(t0), "error": f"start_profile: {exc}"[:300],
                "status": "server unreachable" if running(server) else "server exited"}
    deadline = t0 + finalize_timeout
    while time.time() < deadline:
        if not running(server):
            return {"dir": str(directory), "start": utc(t0), "status": "server exited"}
        if len(finalized_ranks(directory, tp)) >= tp:
            return {"dir": str(directory), "start": utc(t0), "end": utc(time.time()), "status": "ok"}
        time.sleep(5)
    try:
        http(base_url + "/stop_profile", {}, timeout=60)
    except HTTP_ERRORS:
        pass
    return {"dir": str(directory), "start": utc(t0), "status": "timeout"}


def server_script(a: argparse.Namespace, leg_dir: Path, extra: list[str], log_name: str) -> str:
    args = ["python3", "-m", "sglang.launch_server", "--model-path", a.model_path,
            "--host", "127.0.0.1", "--port", str(a.port), "--tp-size", str(a.tp),
            "--context-length", str(a.context_length), "--mem-fraction-static", str(a.mem_fraction),
            "--chunked-prefill-size", str(a.chunked_prefill_size),
            *(["--page-size", str(a.page_size)] if a.page_size else []),
            *(["--kv-cache-dtype", a.kv_cache_dtype] if a.kv_cache_dtype else []),
            "--watchdog-timeout", "600", "--enable-metrics", "--decode-log-interval", "1",
            "--log-level", "info", "--log-requests-level", "0"]
    if a.served_model_name:
        args += ["--served-model-name", a.served_model_name]
    if a.model_override_args:
        args += ["--json-model-override-args", a.model_override_args]
    args += extra + shlex.split(a.server_args)
    cmd = " ".join(shlex.quote(x) for x in args)
    out = shlex.quote(str(leg_dir))
    if getattr(a, "shape_hooks", ""):  # every server process imports the hooks' sitecustomize
        site = shlex.quote(f"{HOOKS_MOUNT}/{shape_hooks.SITE_DIR}")
        cmd = f"export PYTHONPATH={site}${{PYTHONPATH:+:$PYTHONPATH}}; {cmd}"
    # The server runs as root (the image's default); give the outputs back to the caller on exit.
    return (f"cleanup() {{ chown -R \"$HOST_UID:$HOST_GID\" {out}; }}; trap cleanup EXIT TERM INT; "
            f"cd /sgl-workspace 2>/dev/null || true; {cmd} > {out}/{shlex.quote(log_name)} 2>&1 & wait $!")


FAULT_RE = re.compile(r"Memory access fault|Segmentation fault|Traceback|RuntimeError|HIP error|core dumped")
BATCH_RE = re.compile(r"(Prefill|Decode) batch")


def crash_excerpt(text: str, lines: int = 6) -> tuple[str, list[str]]:
    """(first fault line, the scheduler batch lines that precede it) of a server log."""
    rows = text.splitlines()
    for i, line in enumerate(rows):
        if FAULT_RE.search(line):
            before = [r[:240] for r in rows[:i] if BATCH_RE.search(r)][-lines:]
            return line.strip()[:300], before
    return "server exited during the replay", [r[:240] for r in rows if BATCH_RE.search(r)][-lines:]


def plan_windows(window_at: list[float], t0: float, segment_start: float, done: set[int]) -> list[tuple[int, float]]:
    """Profiler windows still to take in a segment: (index, wall time); earlier ones are skipped."""
    return [(i, t0 + off) for i, off in enumerate(window_at, 1) if i not in done and t0 + off >= segment_start]


def profiled_legs(choice: str, legs: list[str]) -> set[str]:
    """Legs that take profiler windows. ``auto``: the eager leg when there is one, else the graph leg.

    Profiling a CUDA-graph server has hung servers (no step after /start_profile); the graph leg is only
    needed for its scheduler log and CUDA-graph sizes, so by default only the eager leg is profiled.
    """
    if choice == "auto":
        return {"eager"} if "eager" in legs else {"graph"}
    if choice == "none":  # e.g. re-running only the graph leg of a capture whose eager leg was profiled
        return set()
    chosen = {x.strip() for x in choice.split(",") if x.strip()}
    if not chosen or not chosen <= set(legs):
        raise SystemExit(f"capture: --profile-legs must be auto or a subset of --legs {legs}")
    return chosen


def run_leg(a: argparse.Namespace, leg: str, traces: Path) -> dict:
    """One leg: serve, replay, profile; on a server crash restart and resume the replay for the remaining time."""
    window_at = list(a.window_at) if leg in a.profile_set else []
    cap_dir = Path(a.out) / a.capture_id
    leg_dir = cap_dir / leg
    if leg_dir.exists() and any(leg_dir.iterdir()):
        raise SystemExit(f"capture: {leg_dir} exists and is not empty; choose another --capture-id")
    (leg_dir / "logs").mkdir(parents=True)
    (leg_dir / "traces").mkdir()
    (leg_dir / "home").mkdir()
    base_url = f"http://127.0.0.1:{a.port}"
    stem = f"{CONTAINER_PREFIX}{a.capture_id}-{leg}"
    server, client = stem + "-server", stem + "-client"
    out_root = str(Path(a.out).resolve())
    mounts = mount_args(a.model_path, out_root, a.mount)
    env = ["-e", "HF_HUB_OFFLINE=1", "-e", f"HIP_VISIBLE_DEVICES={a.gpus}", "-e", f"CUDA_VISIBLE_DEVICES={a.gpus}",
           "-e", f"SGLANG_TORCH_PROFILER_DIR={leg_dir / 'traces'}",
           "-e", f"HOST_UID={os.getuid()}", "-e", f"HOST_GID={os.getgid()}"]
    for item in a.env:  # the production recipe's server environment (kernel selection, backends)
        env += ["-e", item]
    hooks = getattr(a, "shape_hooks", "")
    if hooks:
        env += ["-e", f"{shape_hooks.ENV}={hooks}"]
        mounts = mounts + ["-v", f"{TOOL_DIR}:{HOOKS_MOUNT}:ro"]
    record = {"leg": leg, "cuda_graph": leg == "graph", "server_log": "logs/server.log",
              "server_extra_args": LEGS[leg] + shlex.split(a.server_args), "dir": str(leg_dir), "windows": [],
              "segments": [], "crashes": [], "restarts": 0, "shape_hooks": bool(hooks)}
    corpus_dir = str(traces.parent)
    t0, done, start = None, set(), 0
    while True:
        log_name = "logs/server.log" if start == 0 else f"logs/server.{start}.log"
        log(f"leg {leg}: starting {server} ({log_name})")
        docker("rm", "-f", server, check=False, capture=True)
        if port_answers(base_url):
            raise SystemExit(f"capture: {base_url} already answers before {server} started; another server "
                             f"holds port {a.port} (use --port or stop it)")
        docker("run", "-d", "--rm", "--name", server, "--device", "/dev/kfd", "--device", "/dev/dri",
               "--group-add", "video", "--ipc", "host", "--network", "host", "--shm-size", "32g",
               "--security-opt", "seccomp=unconfined", "--cap-add", "SYS_PTRACE", *env, *mounts,
               "--entrypoint", "bash", a.image, "-c", server_script(a, leg_dir, LEGS[leg], log_name), capture=True)
        crashed = False
        try:
            t_start = time.time()
            wait_healthy(base_url, server, a.health_timeout)
            ready = round(time.time() - t_start, 1)
            record.setdefault("server_ready_s", ready)
            log(f"leg {leg}: healthy after {ready} s")
            if start == 0:
                info = b"{}"
                for path in ("/get_server_info", "/server_info"):
                    try:
                        info = http(base_url + path)
                        break
                    except HTTP_ERRORS:
                        continue
                (leg_dir / "server_info.json").write_bytes(info)
            now = time.time()
            if t0 is None:
                t0 = now
                record["replay_start"] = utc(t0)
                segment_start = t0 + a.warmup
            else:
                segment_start = max(t0 + a.warmup, now + a.restart_warmup)
            remaining = t0 + a.duration - now
            replay = ["python3", "-m", "src.tools.agentx_workloads", "replay", "--traces", str(traces),
                      "--base-url", base_url, "--lanes", str(a.lanes), "--duration", str(int(remaining)),
                      "--max-ctx", str(a.context_length), "--max-out", str(REPLAY_MAX_OUT),
                      "--vocab-hi", str(prompt_vocab_hi(a.model_path)),
                      "--think-cap", str(a.think_cap),
                      "--seed", str(a.seed + start), "--log", str(leg_dir / "logs" / "replay_requests.jsonl")]
            docker("rm", "-f", client, check=False, capture=True)
            docker("run", "-d", "--rm", "--name", client, "--user", f"{os.getuid()}:{os.getgid()}",
                   "--network", "host", "-e", f"HOME={leg_dir / 'home'}", "-e", "PYTHONDONTWRITEBYTECODE=1",
                   "-v", f"{REPO_ROOT}:{REPO_ROOT}:ro", "-v", f"{corpus_dir}:{corpus_dir}:ro",
                   "-v", f"{out_root}:{out_root}", "-w", str(REPO_ROOT), "--entrypoint", "bash", a.image,
                   "-c", " ".join(shlex.quote(x) for x in replay)
                   + f" >> {shlex.quote(str(leg_dir / 'logs' / 'replay.log'))} 2>&1", capture=True)
            log(f"leg {leg}: replay {'started' if start == 0 else 'resumed'} ({a.lanes} lanes, {remaining:.0f} s)")
            for i, at in plan_windows(window_at, t0, segment_start, done):
                while time.time() < at and running(server):
                    time.sleep(2)
                if not running(server):
                    break
                w = profile_window(base_url, leg_dir / "traces" / f"w{i}", a.tp, a.window_steps, server,
                                   a.finalize_timeout)
                w["offset_s"] = window_at[i - 1]
                w["server_start"] = start
                record["windows"].append(w)
                done.add(i)
                log(f"leg {leg}: window w{i} at +{window_at[i - 1]:.0f} s: {w['status']}")
                if w["status"] == "server unreachable":  # hung: stop it so the crash path restarts it
                    log(f"leg {leg}: {w.get('error')}; stopping {server}")
                    docker("stop", "-t", "60", server, check=False, capture=True)
                if not running(server):
                    break
            # Requests still in flight at the deadline may decode for minutes; bound the drain.
            while running(client) and running(server) and time.time() < t0 + a.duration + a.drain_timeout:
                time.sleep(5)
            crashed = not running(server)
            end = min(time.time(), t0 + a.duration)
            if end > segment_start:
                record["segments"].append({"start": utc(segment_start), "end": utc(end), "server_log": log_name,
                                           "server_start": start})
        except RuntimeError as exc:  # the server died before it became healthy
            crashed = True
            log(f"leg {leg}: {exc}")
        finally:
            docker("stop", "-t", "60", client, check=False, capture=True)
            if not crashed:
                docker("stop", "-t", "120", server, check=False, capture=True)
            deadline = time.time() + 180
            while (running(server) or running(client)) and time.time() < deadline:
                time.sleep(3)
        if not crashed:
            break
        server_log = leg_dir / log_name
        text = server_log.read_text(errors="replace") if server_log.exists() else ""
        error, before = crash_excerpt(text)
        record["crashes"].append({"time": utc(time.time()), "server_start": start, "server_log": log_name,
                                  "error": error, "last_batches": before})
        log(f"leg {leg}: server exited: {error}")
        if t0 is None or record["restarts"] >= a.max_restarts or t0 + a.duration - time.time() < a.min_resume:
            log(f"leg {leg}: not restarting ({record['restarts']} restarts so far)")
            break
        record["restarts"] += 1
        start += 1
    for i, off in enumerate(window_at, 1):
        if i not in done:
            record["windows"].append({"dir": str(leg_dir / "traces" / f"w{i}"), "offset_s": off,
                                      "status": "skipped: server restart or end of replay"})
    record["server_crashed"] = bool(record["crashes"])
    record["replay_end"] = utc(time.time())
    if record["segments"]:  # the span, for readers of older captures
        record["steady_state"] = {"start": record["segments"][0]["start"], "end": record["segments"][-1]["end"]}
    first_log = leg_dir / "logs" / "server.log"
    text = first_log.read_text(errors="replace") if first_log.exists() else ""
    m = GRAPH_BS_RE.search(text)
    record["cuda_graph_bs"] = [int(x) for x in m.group(1).split(",")] if m else []
    (leg_dir / "leg.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"leg {leg}: done ({len(record['crashes'])} crash(es), {record['restarts']} restart(s))")
    return record


def git_revision() -> str:
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return out.stdout.strip() or "unknown"


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-path", required=True, help="model directory (an HF snapshot path works)")
    ap.add_argument("--served-model-name", default="")
    ap.add_argument("--model-name", required=True, help="model name recorded in the capture, e.g. Qwen/Qwen3-32B")
    ap.add_argument("--corpus", required=True,
                    help="AgentX traces.jsonl, its snapshot directory, or the HF dataset cache directory")
    ap.add_argument("--image", required=True, help="SGLang serving image")
    ap.add_argument("--out", required=True, help="capture root; this capture goes to <out>/<capture-id>/")
    ap.add_argument("--capture-id", required=True, help="name of this capture, e.g. qwen3-32b-tp8-mi355x")
    ap.add_argument("--legs", default="eager,graph", help="comma-separated legs: eager, graph")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--port", type=int, default=30591)
    ap.add_argument("--context-length", type=int, default=131072)
    ap.add_argument("--model-override-args", default="", help="JSON for --json-model-override-args (e.g. YaRN)")
    ap.add_argument("--mem-fraction", type=float, default=0.80, help="SGLang --mem-fraction-static")
    ap.add_argument("--chunked-prefill-size", type=int, default=16384,
                    help="SGLang --chunked-prefill-size; use the model's production (InferenceX agentic) recipe")
    ap.add_argument("--page-size", type=int, default=None, help="SGLang --page-size (default: SGLang's)")
    ap.add_argument("--kv-cache-dtype", default=None, help="SGLang --kv-cache-dtype, e.g. fp8_e4m3 (default: SGLang's)")
    ap.add_argument("--server-args", default="", help="extra SGLang server arguments (one string)")
    ap.add_argument("--mount", action="append", default=[], help="extra docker -v mount for the server")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment variable of the server container (repeatable), e.g. from the model's recipe")
    ap.add_argument("--lanes", type=int, default=32)
    ap.add_argument("--duration", type=float, default=1080, help="replay seconds per leg")
    ap.add_argument("--warmup", type=float, default=300, help="seconds before the steady state starts")
    ap.add_argument("--window-at", type=float, nargs="+", default=[360, 540, 720, 900],
                    help="profiler window offsets after the replay start (seconds)")
    ap.add_argument("--window-steps", type=int, default=100, help="forward steps per profiler window")
    ap.add_argument("--profile-legs", default="auto",
                    help="legs that take profiler windows: auto (eager when present, else graph), none, or a list")
    ap.add_argument("--think-cap", type=float, default=300.0, help="cap on the trace's think time (seconds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drain-timeout", type=float, default=180,
                    help="seconds to let in-flight requests finish after --duration before stopping the client")
    ap.add_argument("--max-restarts", type=int, default=3, help="server restarts per leg after a crash")
    ap.add_argument("--restart-warmup", type=float, default=120,
                    help="seconds after a restart before the steady state resumes (radix cache refill)")
    ap.add_argument("--min-resume", type=float, default=180,
                    help="restart only when at least this much replay time remains")
    ap.add_argument("--shape-hooks", default="default",
                    help="Python wrappers whose kernels get a recorded op with input shapes in the traces: "
                         "comma-separated module:function[=label] entries and presets "
                         f"({', '.join(sorted(shape_hooks.PRESETS))}); 'none' disables them")
    ap.add_argument("--health-timeout", type=float, default=1800)
    ap.add_argument("--finalize-timeout", type=float, default=900)


def main(a: argparse.Namespace) -> int:
    legs = [x.strip() for x in a.legs.split(",") if x.strip()]
    if not legs or any(x not in LEGS for x in legs):
        raise SystemExit(f"capture: --legs must name {sorted(LEGS)}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.capture_id):
        raise SystemExit("capture: --capture-id may use letters, digits, '_', '.', '-'")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", item) for item in a.env):
        raise SystemExit("capture: --env takes KEY=VALUE")
    if any(w < a.warmup for w in a.window_at) or any(w >= a.duration for w in a.window_at):
        raise SystemExit("capture: --window-at offsets must lie in [--warmup, --duration)")
    a.profile_set = profiled_legs(a.profile_legs, legs)
    try:
        a.shape_hooks = shape_hooks.canonical(a.shape_hooks)
    except ValueError as exc:
        raise SystemExit(f"capture: --shape-hooks: {exc}")
    traces, corpus_name, revision = corpus_traces(a.corpus)
    cap_dir = Path(a.out) / a.capture_id
    cap_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cap_dir / "capture.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "capture_id": a.capture_id, "model": a.model_name, "model_path": a.model_path, "image": a.image,
        "corpus": {"id": corpus_name, "revision": revision, "traces": str(traces)},
        "tp": a.tp, "context_length": a.context_length, "chunked_prefill_size": a.chunked_prefill_size,
        "page_size": a.page_size, "kv_cache_dtype": a.kv_cache_dtype, "server_args": a.server_args,
        "server_env": list(a.env),
        "shape_hooks": [t.spec() for t in shape_hooks.parse(a.shape_hooks)],
        "mem_fraction_static": a.mem_fraction, "model_override_args": a.model_override_args,
        "replay": {"lanes": a.lanes, "duration_s": a.duration, "warmup_s": a.warmup, "seed": a.seed,
                   "vocab_hi": prompt_vocab_hi(a.model_path),
                   "think_cap_s": a.think_cap, "window_at_s": a.window_at, "window_steps": a.window_steps,
                   "profiled_legs": sorted(a.profile_set)},
        "tool_revision": git_revision(),
    })
    from .replay import corpus_coverage
    manifest["corpus_coverage"] = corpus_coverage(str(traces), a.context_length, REPLAY_MAX_OUT)
    log(f"corpus coverage at {a.context_length} tokens: {manifest['corpus_coverage']}")
    manifest.setdefault("legs", {})
    for leg in legs:
        manifest["legs"][leg] = run_leg(a, leg, traces)
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"capture written to {cap_dir}; next: python3 -m src.tools.agentx_workloads census {cap_dir}")
    return 0
