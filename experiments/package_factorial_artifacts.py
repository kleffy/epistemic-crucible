"""Package lightweight, auditable v0.2 integrity and BC reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict

_HERE = pathlib.Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from crucible.agents.neural_ppo import NeuralPPOAgent  # noqa: E402
from crucible.factorial import generate_affordance_quartet, run_scripted_control  # noqa: E402
from experiments.train_factorial_bc import evaluate_agent_outcomes  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_HERE, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _gpu_metadata() -> list[dict[str, str]] | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    devices = []
    for line in result.stdout.splitlines():
        name, memory_mib, driver = (part.strip() for part in line.split(",", maxsplit=2))
        devices.append({"name": name, "memory_mib": memory_mib, "driver_version": driver})
    return devices


def package_artifacts(source: pathlib.Path, output: pathlib.Path) -> dict:
    integrity = source / "integrity_report.json"
    bc_dir = source / "bc"
    bc_summary = bc_dir / "factorial_bc_summary.json"
    invalid_summary = bc_dir / "factorial_bc_summary_in_sample_overlap.json"
    for required in (integrity, bc_summary, invalid_summary):
        if not required.exists():
            raise FileNotFoundError(required)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(integrity, output / "integrity_report.json")
    shutil.copy2(bc_summary, output / "factorial_bc_summary.json")

    checkpoints = sorted(bc_dir.glob("factorial_*_bc_seed*.pt"))
    checkpoint_hashes = {checkpoint.name: _sha256(checkpoint) for checkpoint in checkpoints}
    (output / "checkpoint_hashes.json").write_text(
        json.dumps(checkpoint_hashes, indent=2, sort_keys=True) + "\n"
    )

    invalid = {
        "artifact": invalid_summary.name,
        "sha256": _sha256(invalid_summary),
        "status": "invalid-do-not-report",
        "reason": "Evaluation seeds 0..63 overlapped BC demonstration-world seeds 0..99.",
        "replacement": "factorial_bc_summary.json evaluated on disjoint seeds 100..163",
    }
    (output / "invalid_evaluations.json").write_text(
        json.dumps(invalid, indent=2, sort_keys=True) + "\n"
    )

    controls = []
    for control in ("mechanism_oracle", "detector_policy", "cue_follower", "focal_uniform"):
        outcome = run_scripted_control(generate_affordance_quartet(100).cell(0, 1), control)
        controls.append({"control": control, **asdict(outcome)})
    (output / "sample_control_traces.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in controls)
    )

    bc_samples = []
    for regime in ("cue", "mechanism"):
        checkpoint = bc_dir / f"factorial_{regime}_bc_seed0.pt"
        agent = NeuralPPOAgent.load(checkpoint, device="cpu")
        for outcome in evaluate_agent_outcomes(agent, [100]):
            bc_samples.append({"regime": regime, "initialization_seed": 0, **asdict(outcome)})
    (output / "sample_bc_traces.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in bc_samples)
    )

    try:
        import torch

        torch_metadata = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available_at_packaging": torch.cuda.is_available(),
        }
    except ImportError:
        torch_metadata = None
    source_files = (
        pathlib.Path("pyproject.toml"),
        pathlib.Path("configs/factorial_bc_v02.yaml"),
        pathlib.Path("experiments/run_integrity_gate.py"),
        pathlib.Path("experiments/train_factorial_bc.py"),
        pathlib.Path("tests/test_quartet_invariance.py"),
        pathlib.Path("tests/test_attribution_metrics.py"),
    )
    manifest = {
        "schema_version": "0.2",
        "protocol_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_metadata,
        "gpu": _gpu_metadata(),
        "dependency_versions": {
            package: importlib.metadata.version(package) for package in ("numpy", "pyyaml", "torch")
        },
        "commands": {
            "integrity": "python experiments/run_integrity_gate.py",
            "training": (
                "python experiments/train_factorial_bc.py "
                "--config configs/factorial_bc_v02.yaml "
                "--output-dir results/factorial_v02/bc"
            ),
            "held_out_evaluation": (
                "python experiments/train_factorial_bc.py "
                "--config configs/factorial_bc_v02.yaml "
                "--output-dir results/factorial_v02/bc --evaluate-existing"
            ),
            "acceptance": (
                "pytest -q tests/test_quartet_invariance.py "
                "tests/test_attribution_metrics.py  # 1033 tests"
            ),
        },
        "source_files": {str(path): _sha256(path) for path in source_files},
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.name != "artifact_manifest.json":
            manifest["files"][path.name] = _sha256(path)
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default="results/factorial_v02")
    parser.add_argument("--output", type=pathlib.Path, default="results/reference/v02")
    args = parser.parse_args(argv)
    package_artifacts(args.source, args.output)


if __name__ == "__main__":
    main()
