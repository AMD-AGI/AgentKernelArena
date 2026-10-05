"""Package the portable case contract in an explicitly opted-in task."""

import argparse
import os
import subprocess
from pathlib import Path

import yaml

if __package__:
    from .trusted_task_eval import PORTABLE_CONTRACT, relative_file
else:
    from trusted_task_eval import PORTABLE_CONTRACT, relative_file


def materialize(task, check=False):
    task = Path(task).resolve()
    config = yaml.safe_load((task / "config.yaml").read_text())
    descriptor = config.get("trusted_evaluation")
    if (not isinstance(descriptor, dict) or type(descriptor.get("schema_version")) is not int
            or descriptor["schema_version"] != 1):
        raise ValueError("task must explicitly enable trusted_evaluation schema_version 1")
    target = task / relative_file(descriptor.get("contract_file", "ut/evaluation_contract.py"))
    if not target.resolve().is_relative_to(task) or target.is_symlink():
        raise ValueError("contract destination must remain inside the task")
    if check:
        if target.read_bytes() != PORTABLE_CONTRACT.read_bytes():
            raise ValueError("packaged evaluation contract differs from canonical source")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["GOMAXPROCS"] = "1"
    subprocess.run(["rclone", "copyto", str(PORTABLE_CONTRACT), str(target), "--transfers", "64000",
                    "--progress", "--buffer-size", "0", "--config", os.devnull],
                   check=True, timeout=300, env=environment)
    materialize(task, check=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(materialize(args.task, args.check))


if __name__ == "__main__":
    main()
