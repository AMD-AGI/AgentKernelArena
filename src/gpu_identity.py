# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Board identity checks for GPU models that qualify specific products.

A gfx token names an instruction set, not a board. Consumer and workstation
gfx11 boards share ISA tokens, so a cheatsheet model that lists
``marketing_names`` qualifies only those boards, and a gfx11 device under any
other model is refused. The PCI device ID is recorded for provenance; it is
never used as identity.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

CHEATSHEET_PATH = (
    Path(__file__).resolve().parent / "prompts" / "cheatsheet" / "default_cheatsheet.yaml"
)
PRODUCT_GATED_ARCH_PREFIXES = ("gfx11",)
ROCMINFO_TIMEOUT_S = 10
_AGENT_HEADER = re.compile(r"^\s*Agent\s+\d+\s*$")
_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z ]*?):\s*(.*?)\s*$")
_CHIP_ID = re.compile(r"^\s*(\d+)\s*(?:\(\s*0x([0-9a-fA-F]+)\s*\))?\s*$")


@dataclass(frozen=True)
class GpuAgent:
    gfx_arch: str
    marketing_name: str
    pci_device_id: str | None = None
    node: int | None = None
    uuid: str | None = None

    def describe(self) -> str:
        name = self.marketing_name or "<empty>"
        node = "?" if self.node is None else self.node
        return (
            f"node {node} '{name}' {self.gfx_arch} "
            f"PCI {self.pci_device_id or 'unknown'}"
        )


@dataclass(frozen=True)
class ModelIdentity:
    model: str | None
    gfx_arch: str | None
    marketing_names: tuple[str, ...] = ()

    @property
    def product_gated(self) -> bool:
        return bool(self.marketing_names)

    def accepts(self, agent: GpuAgent) -> bool:
        accepted = {name.casefold() for name in self.marketing_names}
        return (
            agent.gfx_arch == self.gfx_arch
            and agent.marketing_name.strip().casefold() in accepted
        )


def _pci_device_id(chip_id: str) -> str | None:
    match = _CHIP_ID.match(chip_id)
    if not match:
        return None
    value = int(match.group(2), 16) if match.group(2) else int(match.group(1))
    return f"1002:{value:04x}" if value else None


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def parse_rocminfo(text: str) -> list[GpuAgent]:
    """Return the GPU agents of a rocminfo report, in report order."""

    blocks: list[list[str]] = []
    for line in text.splitlines():
        if _AGENT_HEADER.match(line):
            blocks.append([])
        elif blocks:
            blocks[-1].append(line)

    agents: list[GpuAgent] = []
    for block in blocks:
        fields: dict[str, str] = {}
        for line in block:
            # ISA sub-entries reuse "Name:" for amdgcn-amd-amdhsa--gfx targets.
            if line.strip() == "ISA Info:":
                break
            match = _FIELD.match(line)
            if match and match.group(1) not in fields:
                fields[match.group(1)] = match.group(2)
        gfx_arch = fields.get("Name", "").split(":", 1)[0]
        # The CPU agent also has a Marketing Name; only gfx agents are GPUs.
        if not re.fullmatch(r"gfx[0-9a-z]+", gfx_arch):
            continue
        agents.append(
            GpuAgent(
                gfx_arch=gfx_arch,
                marketing_name=fields.get("Marketing Name", "").strip(),
                pci_device_id=_pci_device_id(fields.get("Chip ID", "")),
                node=_int_or_none(fields.get("Node")),
                uuid=fields.get("Uuid") or None,
            )
        )
    return agents


def probe_gpu_agents() -> list[GpuAgent] | None:
    """Run rocminfo from PATH (honoring ROCR_VISIBLE_DEVICES); None if it cannot run."""

    binary = shutil.which("rocminfo")
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [binary], capture_output=True, text=True, timeout=ROCMINFO_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return parse_rocminfo(result.stdout)


def _load_architectures(cheatsheet_path: Path) -> Mapping[str, Any]:
    import yaml

    try:
        config = yaml.safe_load(Path(cheatsheet_path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    architectures = config.get("architecture") if isinstance(config, dict) else None
    return architectures if isinstance(architectures, dict) else {}


def load_model_identity(
    target_gpu_model: str | None, cheatsheet_path: Path = CHEATSHEET_PATH
) -> ModelIdentity:
    """Resolve a model the way preprocessing does (exact, upper, lower key)."""

    if target_gpu_model is None:
        return ModelIdentity(model=None, gfx_arch=None)
    model = str(target_gpu_model)
    architectures = _load_architectures(cheatsheet_path)
    entry = architectures.get(model) or architectures.get(model.upper()) or architectures.get(model.lower())
    if not isinstance(entry, dict):
        return ModelIdentity(model=model, gfx_arch=None)
    names = entry.get("marketing_names") or ()
    return ModelIdentity(
        model=model,
        gfx_arch=entry.get("gfx_arch"),
        marketing_names=tuple(str(name).strip() for name in names if str(name).strip()),
    )


def qualifying_models(gfx_arch: str, cheatsheet_path: Path = CHEATSHEET_PATH) -> list[str]:
    return [
        str(model)
        for model, entry in _load_architectures(cheatsheet_path).items()
        if isinstance(entry, dict)
        and entry.get("gfx_arch") == gfx_arch
        and entry.get("marketing_names")
    ]


def _is_product_gated_arch(gfx_arch: str | None) -> bool:
    return bool(gfx_arch) and gfx_arch.startswith(PRODUCT_GATED_ARCH_PREFIXES)


def identity_error(
    identity: ModelIdentity,
    agents: Sequence[GpuAgent] | None,
    eval_tools: Sequence[str] = (),
    cheatsheet_path: Path = CHEATSHEET_PATH,
) -> str | None:
    """Return the refusal reason for this model on these agents, or None."""

    observed = list(agents or ())
    if identity.product_gated:
        accepted = " or ".join(f"'{name}'" for name in identity.marketing_names)
        requirement = (
            f"target_gpu_model {identity.model} requires every visible GPU to report "
            f"{identity.gfx_arch} with marketing name {accepted} in rocminfo"
        )
        if agents is None:
            return f"{requirement}; rocminfo was not found on PATH or failed."
        if not observed:
            return f"{requirement}; rocminfo reported no GPU agents."
        refused = [agent for agent in observed if not identity.accepts(agent)]
        if refused:
            return f"{requirement}; refused: {', '.join(a.describe() for a in refused)}."
        if eval_tools:
            return (
                "Evaluation-tool sidecars are verified only for gfx950; "
                f"target_gpu_model {identity.model} ({identity.gfx_arch}) must run with "
                f"evaluation_tools disabled (enabled: {', '.join(eval_tools)})."
            )
        return None

    gated = [agent for agent in observed if _is_product_gated_arch(agent.gfx_arch)]
    if not gated:
        return None
    suggestions = sorted(
        {model for agent in gated for model in qualifying_models(agent.gfx_arch, cheatsheet_path)}
    )
    hint = (
        f" Use target_gpu_model {' or '.join(suggestions)} if the board is qualified."
        if suggestions
        else " No AKA target_gpu_model qualifies this board."
    )
    return (
        f"target_gpu_model {identity.model or '<unset>'} ({identity.gfx_arch or 'unknown arch'}) "
        f"does not identify a qualified gfx11 board; refused: "
        f"{', '.join(a.describe() for a in gated)}.{hint}"
    )


def identity_summary(identity: ModelIdentity, agents: Sequence[GpuAgent] | None) -> str:
    described = ", ".join(agent.describe() for agent in agents or ()) or "none"
    return f"GPU identity: target_gpu_model {identity.model} accepted {described}"


def write_identity_record(
    run_directory: Path,
    label: str,
    identity: ModelIdentity,
    agents: Sequence[GpuAgent] | None,
) -> Path:
    """Write gpu_identity/<label>.json; one file per role avoids worker races."""

    safe_label = re.sub(r"[^A-Za-z0-9_.-]", "_", label) or "serial"
    path = Path(run_directory) / "gpu_identity" / f"{safe_label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "target_gpu_model": identity.model,
        "gfx_arch": identity.gfx_arch,
        "accepted_marketing_names": list(identity.marketing_names),
        "role": label,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agents": [asdict(agent) for agent in agents or ()],
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def _config_target(config_path: str) -> tuple[str | None, tuple[str, ...]]:
    import yaml

    from src.eval_tools.config import EvalToolsConfig

    if not Path(config_path).is_file():
        raise SystemExit(f"config file not found: {config_path}")
    config = yaml.safe_load(Path(config_path).read_text()) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"config file must contain a mapping: {config_path}")
    try:
        eval_tools = EvalToolsConfig.from_mapping(config).enabled
    except ValueError:
        eval_tools = ()
    return config.get("target_gpu_model"), eval_tools


def smoke_identity_lines(selected_arch: str | None, config_path: str | None = None) -> list[str]:
    """Identity lines for the runtime smoke; raise SystemExit on refusal."""

    agents = probe_gpu_agents()
    if agents is None:
        lines = ["gpu_marketing_name=unavailable (rocminfo missing or failed)", "gpu_pci_device_id=unavailable"]
    else:
        lines = [
            "gpu_marketing_name=" + ("; ".join(a.marketing_name or "<empty>" for a in agents) or "none"),
            "gpu_pci_device_id=" + ("; ".join(a.pci_device_id or "unknown" for a in agents) or "none"),
        ]

    eval_tools: tuple[str, ...] = ()
    if config_path:
        model, eval_tools = _config_target(config_path)
        identity = load_model_identity(model)
    elif _is_product_gated_arch(selected_arch):
        models = qualifying_models(selected_arch)
        identity = load_model_identity(models[0]) if models else ModelIdentity(None, selected_arch)
    else:
        identity = ModelIdentity(model=None, gfx_arch=selected_arch)

    error = identity_error(identity, agents, eval_tools)
    if error:
        raise SystemExit(f"GPU identity refused: {error}")
    if identity.product_gated:
        lines.append(f"gpu_identity=PASS target_gpu_model={identity.model}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="target_gpu_model to check (default: report only)")
    parser.add_argument("--rocminfo-file", help="parse this rocminfo capture instead of probing")
    args = parser.parse_args(argv)

    if args.rocminfo_file:
        agents: list[GpuAgent] | None = parse_rocminfo(Path(args.rocminfo_file).read_text())
    else:
        agents = probe_gpu_agents()
    identity = load_model_identity(args.model)
    error = identity_error(identity, agents) if args.model else None
    print(json.dumps({
        "target_gpu_model": identity.model,
        "gfx_arch": identity.gfx_arch,
        "agents": None if agents is None else [asdict(agent) for agent in agents],
        "refusal": error,
    }, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
