from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


def sha256_file_head(path: str, nbytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


@dataclass
class ArtifactPaths:
    run_dir: Path
    manifest: Path
    profile: Path
    domain_inference: Path
    candidate_checks: Path
    evidence_results: Path
    final_report: Path


def init_run(out_dir: str) -> ArtifactPaths:
    run_id = uuid4().hex
    run_dir = Path(out_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        run_dir=run_dir,
        manifest=run_dir / "run_manifest.json",
        profile=run_dir / "profile_summary.json",
        domain_inference=run_dir / "domain_inference.json",
        candidate_checks=run_dir / "candidate_checks.json",
        evidence_results=run_dir / "evidence_results.json",
        final_report=run_dir / "final_report.json",
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(dataset_ref: str, model: str, prompt_versions: Dict[str, str], sample_n: int) -> Dict[str, Any]:
    p = Path(dataset_ref)
    fingerprint = {
        "dataset_ref": dataset_ref,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
        "sha256_head": sha256_file_head(dataset_ref) if p.exists() and p.is_file() else None,
    }
    return {
        "dataset_fingerprint": fingerprint,
        "llm": {"model": model},
        "prompt_versions": prompt_versions,
        "sample_n": sample_n,
    }
