import atexit
import json
import logging
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class HardwareControlError(RuntimeError):
    """Raised when hardware clock control cannot be applied safely."""


@dataclass
class AMDClockLockConfig:
    enabled: bool = False
    gpu_clock_level: Optional[int] = None
    mem_clock_level: Optional[int] = None
    device_id: int = 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_tail(text: str, limit: int = 1000) -> str:
    text = text or ""
    return text[-limit:]


def _parse_supported_clock_frequencies(output: str) -> Dict[str, List[Dict[str, Any]]]:
    supported: Dict[str, List[Dict[str, Any]]] = {}
    current_section: Optional[str] = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^GPU\[\d+\]\s*:\s*", "", line, flags=re.IGNORECASE)
        section_match = re.search(r"Supported\s+([a-z0-9]+)\s+frequencies", line, re.IGNORECASE)
        if section_match:
            current_section = section_match.group(1).lower()
            supported.setdefault(current_section, [])
            continue

        if current_section is None:
            continue

        freq_match = re.match(r"([A-Za-z0-9]+):\s*(\d+)Mhz(\s+\*)?$", line)
        if not freq_match:
            continue

        level_label = freq_match.group(1)
        mhz = int(freq_match.group(2))
        is_active = bool(freq_match.group(3))
        level_value = int(level_label) if level_label.isdigit() else level_label
        supported[current_section].append(
            {
                "level": level_value,
                "level_label": level_label,
                "mhz": mhz,
                "active": is_active,
            }
        )

    return supported


def _parse_current_clocks(output: str) -> Dict[str, Dict[str, Any]]:
    clocks: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(
        r"([a-z0-9]+)\s+clock\s+level:\s*([A-Za-z0-9]+):\s*\((\d+)Mhz\)",
        re.IGNORECASE,
    )

    for raw_line in output.splitlines():
        line = raw_line.strip()
        match = pattern.search(line)
        if not match:
            continue
        clock_name = match.group(1).lower()
        level_label = match.group(2)
        mhz = int(match.group(3))
        level_value = int(level_label) if level_label.isdigit() else level_label
        clocks[clock_name] = {
            "level": level_value,
            "level_label": level_label,
            "mhz": mhz,
        }

    return clocks


def _parse_perf_level(output: str) -> Optional[str]:
    match = re.search(r"Performance Level:\s*([A-Za-z0-9_]+)", output, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _parse_product_info(output: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    patterns = {
        "card_series": r"Card Series:\s*(.+)",
        "card_model": r"Card Model:\s*(.+)",
        "card_vendor": r"Card Vendor:\s*(.+)",
        "card_sku": r"Card SKU:\s*(.+)",
        "gfx_version": r"GFX Version:\s*(.+)",
        "device_name": r"Device Name:\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            info[key] = match.group(1).strip()
    return info


class AMDHardwareControlSession:
    """Run-level AMD clock control with persistent metadata and robust cleanup."""

    _SIGNALS = tuple(
        sig
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None))
        if sig is not None
    )

    def __init__(
        self,
        config: AMDClockLockConfig,
        raw_config: Dict[str, Any],
        config_error: Optional[str],
        run_directory: Path,
        target_gpu_model: str,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.raw_config = raw_config
        self.config_error = config_error
        self.run_directory = run_directory
        self.target_gpu_model = target_gpu_model
        self.logger = logger
        self.metadata_path = run_directory / "hardware_control.json"
        self._lock = Lock()
        self._atexit_registered = False
        self._signal_handlers_installed = False
        self._cleaned = False
        self._active = False
        self._applying = False
        self._previous_signal_handlers: Dict[int, Any] = {}

        self.metadata: Dict[str, Any] = {
            "status": "initialized",
            "target_gpu_model": target_gpu_model,
            "run_directory": str(run_directory),
            "config_key": "gpu_clock_frequence_lock",
            "config": {
                "enabled": config.enabled,
                "gpu_clock_level": config.gpu_clock_level,
                "mem_clock_level": config.mem_clock_level,
                "device_id": config.device_id,
            },
            "raw_config": raw_config,
            "config_error": config_error,
            "device": {"device_id": config.device_id},
            "supported_clocks": {},
            "snapshots": {
                "before_apply": None,
                "after_apply": None,
                "after_reset": None,
            },
            "apply": {
                "attempted": False,
                "success": False,
                "started_at": None,
                "ended_at": None,
                "commands": [],
                "error": None,
            },
            "reset": {
                "attempted": False,
                "success": False,
                "started_at": None,
                "ended_at": None,
                "reason": None,
                "commands": [],
                "error": None,
            },
            "events": [],
            "last_updated_at": _utc_now_iso(),
        }
        self._record_event("Hardware control session created")
        self._write_metadata()

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        run_directory: Path,
        target_gpu_model: str,
        logger: logging.Logger,
    ) -> "AMDHardwareControlSession":
        raw_section = config.get("gpu_clock_frequence_lock")
        if raw_section is None:
            raw_section = config.get("gpu_clock_frequency_lock", {})

        config_error: Optional[str] = None
        if raw_section is None:
            raw_section = {}
        if not isinstance(raw_section, dict):
            config_error = "gpu_clock_frequence_lock must be a mapping"
            raw_section = {}

        def _optional_int(name: str) -> Optional[int]:
            value = raw_section.get(name)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer or null")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            return value

        enabled = raw_section.get("enabled", False)
        if not isinstance(enabled, bool):
            config_error = "gpu_clock_frequence_lock.enabled must be true or false"
            enabled = False

        device_id = raw_section.get("device_id", 0)
        if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
            config_error = "gpu_clock_frequence_lock.device_id must be a non-negative integer"
            device_id = 0

        gpu_clock_level: Optional[int] = None
        mem_clock_level: Optional[int] = None
        if config_error is None:
            try:
                gpu_clock_level = _optional_int("gpu_clock_level")
                mem_clock_level = _optional_int("mem_clock_level")
            except ValueError as exc:
                config_error = str(exc)

        if config_error is None and enabled and gpu_clock_level is None and mem_clock_level is None:
            config_error = (
                "gpu_clock_frequence_lock.enabled is true but neither gpu_clock_level nor mem_clock_level is set"
            )

        parsed = AMDClockLockConfig(
            enabled=enabled,
            gpu_clock_level=gpu_clock_level,
            mem_clock_level=mem_clock_level,
            device_id=device_id,
        )
        return cls(parsed, raw_section, config_error, run_directory, target_gpu_model, logger)

    def install_exit_guards(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True
            self._record_event("Registered atexit cleanup handler")

        if self._signal_handlers_installed:
            self._write_metadata()
            return

        for sig in self._SIGNALS:
            self._previous_signal_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        self._signal_handlers_installed = True
        self._record_event("Installed signal handlers for hardware control cleanup")
        self._write_metadata()

    def apply(self) -> None:
        with self._lock:
            if self._cleaned:
                raise HardwareControlError("Hardware control session was already cleaned up")

        self.metadata["apply"]["attempted"] = True
        self.metadata["apply"]["started_at"] = _utc_now_iso()
        self.metadata["status"] = "preparing"
        self._record_event("Preparing hardware control")
        self._write_metadata()

        if self.config_error is not None:
            self.metadata["status"] = "config_error"
            self.metadata["apply"]["error"] = self.config_error
            self.metadata["apply"]["ended_at"] = _utc_now_iso()
            self._record_event(f"Configuration error: {self.config_error}")
            self._write_metadata()
            raise HardwareControlError(self.config_error)

        if not self.config.enabled:
            self.metadata["status"] = "disabled"
            self.metadata["apply"]["success"] = True
            self.metadata["apply"]["ended_at"] = _utc_now_iso()
            self._record_event("Hardware control disabled by config")
            self.logger.info("AMD hardware clock lock disabled by config; skipping apply")
            self._write_metadata()
            return

        if shutil.which("rocm-smi") is None:
            error = "rocm-smi not found in PATH; AMD clock lock cannot be applied"
            self.metadata["status"] = "apply_failed"
            self.metadata["apply"]["error"] = error
            self.metadata["apply"]["ended_at"] = _utc_now_iso()
            self._record_event(error)
            self._write_metadata()
            raise HardwareControlError(error)

        self._applying = True
        try:
            self.metadata["device"].update(self._query_device_info())
            supported = self._query_supported_clocks()
            self.metadata["supported_clocks"] = supported
            self.metadata["snapshots"]["before_apply"] = self._query_current_state()

            self._validate_requested_levels(supported)
            self._set_perf_level("manual", self.metadata["apply"]["commands"])
            self._active = True

            if self.config.gpu_clock_level is not None:
                self._run_rocm_smi(
                    ["--setsclk", str(self.config.gpu_clock_level)],
                    self.metadata["apply"]["commands"],
                    f"set sclk level to {self.config.gpu_clock_level}",
                    reject_stdout_substrings=["not supported on the given system"],
                )
                self._active = True

            if self.config.mem_clock_level is not None:
                self._run_rocm_smi(
                    ["--setmclk", str(self.config.mem_clock_level)],
                    self.metadata["apply"]["commands"],
                    f"set mclk level to {self.config.mem_clock_level}",
                    reject_stdout_substrings=["not supported on the given system"],
                )
                self._active = True

            self.metadata["snapshots"]["after_apply"] = self._query_current_state()
            self.metadata["status"] = "applied"
            self.metadata["apply"]["success"] = True
            self.metadata["apply"]["ended_at"] = _utc_now_iso()
            self._record_event("Hardware control applied successfully")
            self.logger.info(
                "Applied AMD clock lock on device %s (gpu_clock_level=%s, mem_clock_level=%s)",
                self.config.device_id,
                self.config.gpu_clock_level,
                self.config.mem_clock_level,
            )
            self._write_metadata()
        except Exception as exc:
            error = str(exc)
            self.metadata["status"] = "apply_failed"
            self.metadata["apply"]["success"] = False
            self.metadata["apply"]["error"] = error
            self.metadata["apply"]["ended_at"] = _utc_now_iso()
            self._record_event(f"Hardware control apply failed: {error}")
            self.logger.error("AMD hardware clock lock apply failed: %s", error)
            self._write_metadata()

            if self._active:
                try:
                    self._reset_impl(reason="apply_failure_cleanup")
                except Exception as reset_exc:  # pragma: no cover - best effort cleanup
                    reset_error = f"Automatic cleanup after apply failure also failed: {reset_exc}"
                    self.metadata["reset"]["error"] = reset_error
                    self._record_event(reset_error)
                    self._write_metadata()

            raise HardwareControlError(error) from exc
        finally:
            self._applying = False

    def cleanup(self, reason: str = "main_finally") -> None:
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True

        try:
            self._reset_impl(reason=reason)
        finally:
            self._restore_signal_handlers()
            self._write_metadata()

    def _reset_impl(self, reason: str) -> None:
        if self.metadata["reset"]["attempted"] and self.metadata["reset"]["success"] and not self._active:
            self._record_event(f"Cleanup already completed earlier; skipping duplicate reset ({reason})")
            return

        self.metadata["reset"]["attempted"] = True
        self.metadata["reset"]["started_at"] = _utc_now_iso()
        self.metadata["reset"]["reason"] = reason

        if not self.config.enabled:
            self.metadata["reset"]["success"] = True
            self.metadata["reset"]["ended_at"] = _utc_now_iso()
            self.metadata["status"] = "disabled"
            self._record_event(f"Cleanup skipped because hardware control is disabled ({reason})")
            return

        if not self._active:
            self.metadata["reset"]["success"] = True
            self.metadata["reset"]["ended_at"] = _utc_now_iso()
            if self.metadata["status"] == "apply_failed":
                self._record_event(f"No active clock lock to reset after apply failure ({reason})")
            else:
                self.metadata["status"] = "not_applied"
                self._record_event(f"No active clock lock to reset ({reason})")
            return

        try:
            self._run_rocm_smi(
                ["--resetclocks"],
                self.metadata["reset"]["commands"],
                "reset clocks to default",
                require_stdout_substrings=["successfully reset clocks"],
            )
            self._set_perf_level("auto", self.metadata["reset"]["commands"])
            self.metadata["snapshots"]["after_reset"] = self._query_current_state()
            self.metadata["reset"]["success"] = True
            self.metadata["reset"]["ended_at"] = _utc_now_iso()
            self.metadata["status"] = "reset"
            self._active = False
            self._record_event(f"Hardware control reset completed ({reason})")
            self.logger.info("Reset AMD clock lock on device %s (%s)", self.config.device_id, reason)
        except Exception as exc:
            self.metadata["reset"]["success"] = False
            self.metadata["reset"]["error"] = str(exc)
            self.metadata["reset"]["ended_at"] = _utc_now_iso()
            self.metadata["status"] = "reset_failed"
            self._record_event(f"Hardware control reset failed ({reason}): {exc}")
            self.logger.error("AMD hardware clock lock reset failed (%s): %s", reason, exc)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        self._record_event(f"Received signal {signal_name}, triggering hardware cleanup")
        self.logger.warning("Received signal %s; triggering AMD hardware clock cleanup", signal_name)
        self._write_metadata()
        self.cleanup(reason=f"signal:{signal_name}")
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    def _restore_signal_handlers(self) -> None:
        if not self._signal_handlers_installed:
            return
        for sig, previous in self._previous_signal_handlers.items():
            signal.signal(sig, previous)
        self._signal_handlers_installed = False
        self._record_event("Restored previous signal handlers")

    def _atexit_cleanup(self) -> None:
        self.cleanup(reason="atexit")

    def _query_device_info(self) -> Dict[str, Any]:
        product_result = self._run_rocm_smi(["--showproductname"], None, "query product info", check=False)
        device_info = _parse_product_info(product_result["stdout"])
        return device_info

    def _query_supported_clocks(self) -> Dict[str, List[Dict[str, Any]]]:
        result = self._run_rocm_smi(["--showclkfrq"], None, "query supported clock frequencies", check=False)
        supported = _parse_supported_clock_frequencies(result["stdout"])
        if not supported:
            raise HardwareControlError("Unable to parse supported clock frequencies from rocm-smi --showclkfrq")
        return supported

    def _query_current_state(self) -> Dict[str, Any]:
        clocks_result = self._run_rocm_smi(["--showclocks"], None, "query current clocks", check=False)
        perf_result = self._run_rocm_smi(["--showperflevel"], None, "query current perf level", check=False)
        return {
            "captured_at": _utc_now_iso(),
            "perf_level": _parse_perf_level(perf_result["stdout"]),
            "current_clocks": _parse_current_clocks(clocks_result["stdout"]),
        }

    def _validate_requested_levels(self, supported: Dict[str, List[Dict[str, Any]]]) -> None:
        def _available(clock_name: str) -> List[int]:
            return [
                int(entry["level"])
                for entry in supported.get(clock_name, [])
                if isinstance(entry.get("level"), int)
            ]

        if self.config.gpu_clock_level is not None:
            available_sclk = _available("sclk")
            if self.config.gpu_clock_level not in available_sclk:
                raise HardwareControlError(
                    f"Requested gpu_clock_level={self.config.gpu_clock_level} is not supported on device "
                    f"{self.config.device_id}. Available sclk levels: {available_sclk}"
                )

        if self.config.mem_clock_level is not None:
            available_mclk = _available("mclk")
            if self.config.mem_clock_level not in available_mclk:
                raise HardwareControlError(
                    f"Requested mem_clock_level={self.config.mem_clock_level} is not supported on device "
                    f"{self.config.device_id}. Available mclk levels: {available_mclk}"
                )

    def _set_perf_level(self, level: str, command_log: List[Dict[str, Any]]) -> None:
        self._run_rocm_smi(
            ["--setperflevel", level],
            command_log,
            f"set perf level to {level}",
        )

    def _run_rocm_smi(
        self,
        extra_args: List[str],
        command_log: Optional[List[Dict[str, Any]]],
        description: str,
        check: bool = True,
        require_stdout_substrings: Optional[List[str]] = None,
        reject_stdout_substrings: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cmd = ["rocm-smi", "-d", str(self.config.device_id)] + extra_args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        record = {
            "time": _utc_now_iso(),
            "description": description,
            "command": cmd,
            "returncode": result.returncode,
            "stdout_tail": _safe_tail(result.stdout),
            "stderr_tail": _safe_tail(result.stderr),
        }
        if command_log is not None:
            command_log.append(record)

        if check:
            stdout_lower = (result.stdout or "").lower()
            stderr_tail = _safe_tail(result.stderr, 300)
            stdout_tail = _safe_tail(result.stdout, 300)

            if result.returncode != 0:
                raise HardwareControlError(
                    f"rocm-smi command failed while trying to {description}: "
                    f"returncode={result.returncode}, stderr={stderr_tail}"
                )

            if reject_stdout_substrings:
                for needle in reject_stdout_substrings:
                    if needle.lower() in stdout_lower:
                        raise HardwareControlError(
                            f"rocm-smi command reported unsupported/failed state while trying to {description}: "
                            f"stdout={stdout_tail}"
                        )

            if require_stdout_substrings:
                missing = [needle for needle in require_stdout_substrings if needle.lower() not in stdout_lower]
                if missing:
                    raise HardwareControlError(
                        f"rocm-smi command did not confirm success while trying to {description}: "
                        f"missing_markers={missing}, stdout={stdout_tail}"
                    )

        return {
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _record_event(self, message: str) -> None:
        self.metadata["events"].append({"time": _utc_now_iso(), "message": message})
        self.metadata["last_updated_at"] = _utc_now_iso()

    def _write_metadata(self) -> None:
        self.run_directory.mkdir(parents=True, exist_ok=True)
        temp_path = self.metadata_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        temp_path.replace(self.metadata_path)


def create_hardware_control_session(
    config: Dict[str, Any],
    run_directory: Path,
    target_gpu_model: str,
    logger: logging.Logger,
) -> AMDHardwareControlSession:
    return AMDHardwareControlSession.from_config(
        config=config,
        run_directory=run_directory,
        target_gpu_model=target_gpu_model,
        logger=logger,
    )
