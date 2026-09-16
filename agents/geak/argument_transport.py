"""Versioned transport for the private GEAK dispatcher; no SDK/provider imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


MAX_SAFE_INTEGER = 2**53 - 1


# Injected only after verifying the pinned upstream dispatcher and marker.
# JSON.parse interprets data, never code. The scan detects duplicate keys before
# they can be lost by JSON.parse, including nested and escape-equivalent keys.
DECODER_JS = r'''// AKA GEAK argument transport v3
function arenaDecodeArgs(raw) {
  let value = raw;
  if (typeof raw === 'string') {
    value = JSON.parse(raw);
    let i = 0;
    const whitespace = () => { while (/[ \t\r\n]/.test(raw[i] || '\0')) i++; };
    function stringToken() {
      const start = i++;
      while (raw[i] !== '"') { i += raw[i] === '\\' ? 2 : 1; }
      i++;
      return JSON.parse(raw.slice(start, i));
    }
    function scan() {
      whitespace();
      if (raw[i] === '"') { stringToken(); return; }
      if (raw[i] === '{') {
        i++; whitespace();
        const keys = new Set();
        if (raw[i] === '}') { i++; return; }
        while (true) {
          whitespace();
          const key = stringToken();
          if (keys.has(key)) throw Error('GEAK args contain duplicate JSON keys');
          keys.add(key);
          whitespace(); i++; scan(); whitespace();
          if (raw[i++] === '}') return;
        }
      }
      if (raw[i] === '[') {
        i++; whitespace();
        if (raw[i] === ']') { i++; return; }
        while (true) { scan(); whitespace(); if (raw[i++] === ']') return; }
      }
      while (i < raw.length && !/[ \t\r\n,}\]]/.test(raw[i])) i++;
    }
    scan();
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw Error('GEAK args must be a JSON object');
  function check(item) {
    if (typeof item === 'number') {
      if (!Number.isFinite(item))
        throw Error('GEAK args must contain finite JSON numbers');
      if (Number.isInteger(item) && !Number.isSafeInteger(item))
        throw Error('GEAK args integer exceeds JavaScript safe integer range');
    }
    if (item !== null && typeof item === 'object') Object.values(item).forEach(check);
  }
  check(value);
  return value;
}
const arenaArgs = arenaDecodeArgs(args);
'''


FIXED_ARGS_GUARD_JS = """// AKA GEAK fixed arguments v4
if (args === null || typeof args !== 'object' || Array.isArray(args)
    || Object.keys(args).length !== 0)
  throw Error('GEAK dispatcher v4 requires empty object args');
"""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _canonical(value: Any, *, safe_integers: bool = False) -> tuple:
    """Use exact Python numeric equality, without its True == 1 aliasing."""
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        if (safe_integers and abs(value) > MAX_SAFE_INTEGER
                and (type(value) is int or value.is_integer())):
            raise ValueError("GEAK args integer exceeds JavaScript safe integer range")
        # Keep int precision: Python compares int/float exactly without rounding
        # both operands to float. V3 rejects unsafe integers before comparison.
        return ("number", value)
    if type(value) is str:
        return ("string", value)
    if type(value) is list:
        return ("array", tuple(_canonical(item, safe_integers=safe_integers) for item in value))
    if type(value) is dict and all(type(key) is str for key in value):
        return ("object", tuple(sorted(
            (key, _canonical(item, safe_integers=safe_integers)) for key, item in value.items())))
    raise ValueError("Not a JSON value")


def decode_workflow_args(value: Any) -> dict[str, Any]:
    """The private v3 dispatcher's data-only decoding, also used by collectors."""
    if type(value) is str:
        value = json.loads(value, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if type(value) is not dict:
        raise ValueError("GEAK args must be a JSON object")
    _canonical(value, safe_integers=True)
    return value


def validate_args_transport(expected: dict[str, Any] | None, transport: dict | None) -> None:
    """Fail before SDK launch unless opt-in names the actual adapted dispatcher."""
    if transport is None:
        return
    if (type(transport) is not dict
            or set(transport) != {"adapter_version", "adapted_workflow_sha256"}
            or type(transport["adapter_version"]) is not int or transport["adapter_version"] not in (3, 4)
            or type(expected) is not dict or set(expected) != {"scriptPath", "args"}
            or type(expected["scriptPath"]) is not str or type(expected["args"]) is not dict):
        raise ValueError("Invalid GEAK argument transport identity")
    if transport["adapter_version"] == 4 and expected["args"] != {}:
        raise ValueError("GEAK dispatcher v4 requires empty object args")
    decode_workflow_args(expected["args"])
    decoder = DECODER_JS if transport["adapter_version"] == 3 else FIXED_ARGS_GUARD_JS
    path = Path(expected["scriptPath"])
    if path.is_symlink():
        raise ValueError("GEAK dispatcher cannot be a symlink")
    source = path.read_bytes()
    if (hashlib.sha256(source).hexdigest() != transport["adapted_workflow_sha256"]
            or source.count(decoder.encode()) != 1):
        raise ValueError("GEAK dispatcher transport/hash mismatch")


def workflow_inputs_match(inputs: Any, expected: dict[str, Any], *,
                          args_transport: dict | None = None) -> bool:
    """Exact outer keys and complete JSON values; decoding is pinned v3-only.

    Does not mutate raw SDK input. External evidence collectors must pass the
    prepared engine's transport identity, never infer permission from a string.
    """
    validate_args_transport(expected, args_transport)
    try:
        if (type(inputs) is not dict or type(expected) is not dict
                or set(inputs) != {"scriptPath", "args"} or set(expected) != set(inputs)
                or type(inputs["scriptPath"]) is not str
                or inputs["scriptPath"] != expected["scriptPath"]):
            return False
        actual = (decode_workflow_args(inputs["args"])
                  if args_transport is not None and args_transport["adapter_version"] == 3
                  else inputs["args"])
        return _canonical(actual) == _canonical(expected["args"])
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False
