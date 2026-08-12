"""Build and audit the deterministic anonymous code-and-results supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import zipfile
from typing import Any

_ROOT = pathlib.Path(__file__).parent.parent
_FIXED_ZIP_TIME = (2026, 8, 12, 0, 0, 0)
_TEXT_SUFFIXES = {
    ".cfg",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_DENIED_PARTS = {
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "CITATION.cff",
}
_PUBLIC_IDENTIFIERS = (
    "v0.2-protocol-rc2",
    "v0.2-confirmatory-freeze-20260811",
    "github.com/",
    "/home/",
    "/mnt/data/",
    "kleffy",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: pathlib.Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _neutralize(value: Any, *, key: str | None = None) -> Any:
    """Remove public ancestry and trace locations while retaining science settings."""
    if key in {"protocol_commit", "protocol_tag"}:
        return "protocol artifact P"
    if key in {"repository_commit", "freeze_commit", "freeze_tag"}:
        return "confirmatory freeze artifact F"
    if key in {"public_tag", "commit"}:
        return None
    if key == "trace":
        return "sealed trace omitted; digest retained"
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            neutral = _neutralize(child_value, key=child_key)
            if neutral is not None:
                result[child_key] = neutral
        return result
    if isinstance(value, list):
        return [_neutralize(item) for item in value]
    if isinstance(value, str) and (value.startswith("/home/") or value.startswith("/mnt/data/")):
        return "external sealed artifact; path omitted"
    return value


def _source_entries(root: pathlib.Path) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for path in sorted((root / "crucible").rglob("*.py")):
        entries[path.relative_to(root).as_posix()] = path.read_bytes()
    for relative in (
        pathlib.Path("experiments/build_submission_assets.py"),
        pathlib.Path("tests/test_attribution_metrics.py"),
        pathlib.Path("tests/test_quartet_invariance.py"),
        pathlib.Path("tests/test_viz.py"),
        pathlib.Path("configs/factorial_bc_v02.yaml"),
        pathlib.Path("LICENSE"),
    ):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[relative.as_posix()] = path.read_bytes()
    return entries


def _artifact_entries(root: pathlib.Path) -> dict[str, bytes]:
    base = root / "results/reference/v02"
    confirmatory = base / "confirmatory"
    for required in (
        base / "integrity_report.json",
        base / "factorial_bc_summary.json",
        confirmatory / "confirmatory_results.json",
        confirmatory / "manuscript_inputs.json",
        confirmatory / "structural_verification.json",
        confirmatory / "seal_hashes.json",
        confirmatory / "artifact_manifest.json",
        confirmatory / "manuscript/asset_manifest.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    entries: dict[str, bytes] = {
        "results/reference/v02/integrity_report.json": (
            base / "integrity_report.json"
        ).read_bytes(),
        "results/reference/v02/factorial_bc_summary.json": (
            base / "factorial_bc_summary.json"
        ).read_bytes(),
        "results/reference/v02/confirmatory/confirmatory_results.json": _json_bytes(
            _neutralize(json.loads((confirmatory / "confirmatory_results.json").read_text()))
        ),
        "results/reference/v02/confirmatory/manuscript_inputs.json": (
            confirmatory / "manuscript_inputs.json"
        ).read_bytes(),
        "results/reference/v02/confirmatory/structural_verification.json": _json_bytes(
            _neutralize(json.loads((confirmatory / "structural_verification.json").read_text()))
        ),
        "results/reference/v02/confirmatory/seal_hashes.json": (
            confirmatory / "seal_hashes.json"
        ).read_bytes(),
        "results/reference/v02/confirmatory/artifact_manifest.json": (
            confirmatory / "artifact_manifest.json"
        ).read_bytes(),
        "results/reference/v02/confirmatory/README.md": (confirmatory / "README.md").read_bytes(),
    }
    manifest_path = root / "configs/confirmatory_manifest.json"
    entries["configs/confirmatory_manifest.json"] = _json_bytes(
        _neutralize(json.loads(manifest_path.read_text()))
    )
    for path in sorted((confirmatory / "manuscript").iterdir()):
        if path.is_file():
            entries[path.relative_to(root).as_posix()] = path.read_bytes()
    return entries


def _metadata_entries() -> dict[str, bytes]:
    readme = r"""# Anonymous Epistemic Crucible supplement

This archive contains the frozen factorial protocol, crossed 2x2 and 3x2
generators, attribution metrics, integrity tests, compact frozen results, and the
deterministic generator for Tables 1--4 and Figures 1--4. Raw traces, caches,
weights, Git history, and the large immutable seal are intentionally omitted.
The seal digest and structural-verification record are included.

## Reproduce the included checks and assets

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q \
  tests/test_attribution_metrics.py tests/test_quartet_invariance.py tests/test_viz.py
MPLCONFIGDIR=.mpl-cache .venv/bin/python \
  experiments/build_submission_assets.py --output regenerated-assets
(cd results/reference/v02/confirmatory/manuscript && sha256sum -c SHA256SUMS)
```

The prebuilt assets were generated from the committed compact inputs. The
anonymous manifest names the frozen scientific states only as protocol artifact
P and confirmatory freeze artifact F.
"""
    requirements = """numpy>=1.24
pyyaml>=6.0
matplotlib>=3.5
pytest>=7.4
"""
    pyproject = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "anonymous-factorial-assay"
version = "0.2.0"
description = "Anonymous artifact for crossed behavioral policy attribution"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = ["numpy>=1.24", "pyyaml>=6.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["crucible"]
"""
    return {
        "README.md": readme.encode(),
        "requirements.txt": requirements.encode(),
        "pyproject.toml": pyproject.encode(),
    }


def _audit_entries(entries: dict[str, bytes]) -> None:
    email = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for name, content in entries.items():
        parts = pathlib.PurePosixPath(name).parts
        if any(part in _DENIED_PARTS for part in parts):
            raise ValueError(f"denied package entry: {name}")
        if pathlib.PurePosixPath(name).suffix.lower() not in _TEXT_SUFFIXES:
            continue
        lower = content.lower()
        for forbidden in _PUBLIC_IDENTIFIERS:
            if forbidden.lower().encode() in lower:
                raise ValueError(f"identity/public-repository string {forbidden!r} in {name}")
        if email.search(content):
            raise ValueError(f"email address in {name}")


def build_anonymous_supplement(*, root: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    """Build an audited deterministic ZIP and return its local hash record."""
    root = root.resolve()
    entries = {**_source_entries(root), **_artifact_entries(root), **_metadata_entries()}
    manifest_lines = [f"{_sha256_bytes(value)}  {name}" for name, value in sorted(entries.items())]
    entries["MANIFEST.sha256"] = ("\n".join(manifest_lines) + "\n").encode()
    _audit_entries(entries)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("ZIP member order/uniqueness audit failed")
        extracted = {name: archive.read(name) for name in names}
    _audit_entries(extracted)
    return {
        "schema_version": "0.2-anonymous-supplement-1",
        "path": str(output),
        "sha256": sha256(output),
        "size_bytes": output.stat().st_size,
        "members": len(entries),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=_ROOT)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    result = build_anonymous_supplement(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
