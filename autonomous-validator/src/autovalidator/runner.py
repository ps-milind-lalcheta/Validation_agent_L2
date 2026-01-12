from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import pandas as pd

from autovalidator.core.sampling import load_and_sample
from autovalidator.core.redaction import redact_dataframe
from autovalidator.core.profiling import profile_dataframe
from autovalidator.llm.client import LLMClient
from autovalidator.llm.schemas import ProfileSummary
from autovalidator.llm.orchestrator import ValidationOrchestrator
from autovalidator.checks.executor import execute_checks
from autovalidator.policy.decision import DecisionPolicyConfig, apply_policy
from autovalidator.storage.artifacts import init_run, write_json, build_manifest


@dataclass
class RunResult:
    run_id: str
    decision: str
    run_dir: str


def run_validation(dataset_ref: str, out_dir: str, sample_n: int = 500, seed: int = 42, policy_path: Optional[str] = None) -> RunResult:
    load_dotenv()

    paths = init_run(out_dir)
    run_id = paths.run_dir.name

    # Load + sample
    df_full, df_sample = load_and_sample(dataset_ref, n=sample_n, seed=seed)

    # Redact sample for LLM
    red = redact_dataframe(df_sample)
    df_masked = red.redacted_df

    # Profile masked sample
    profile_dict = profile_dataframe(df_masked, dataset_name=os.path.basename(dataset_ref), row_count_estimate=len(df_full))
    profile = ProfileSummary.model_validate(profile_dict)

    # LLM: infer + checks
    llm = LLMClient()
    orchestrator = ValidationOrchestrator(llm)

    domain = orchestrator.infer_domain(profile)
    candidate_checks = orchestrator.generate_checks(profile, domain)

    # Evidence: execute checks on masked sample (MVP). In production you might run on full dataset / partitions.
    evidence = execute_checks(df_masked, candidate_checks.checks)

    # Skeptic produces final report (includes decision recommendation)
    final_report = orchestrator.skeptic(profile, domain, candidate_checks, evidence)

    # Deterministic policy gate
    cfg = DecisionPolicyConfig.from_yaml(policy_path)
    decision = apply_policy(final_report, red.sensitive_score, cfg)

    # Store artifacts
    prompt_versions = {
        "domain_inferencer": "v1",
        "check_generator": "v1",
        "skeptic": "v1",
        "report_writer": "v1",
    }
    manifest = build_manifest(dataset_ref, llm.config.model, prompt_versions, sample_n)

    write_json(paths.manifest, manifest)
    write_json(paths.profile, profile.model_dump())
    write_json(paths.domain_inference, domain.model_dump())
    write_json(paths.candidate_checks, candidate_checks.model_dump())
    write_json(paths.evidence_results, [e.model_dump() for e in evidence])
    # override with deterministic decision
    fr = final_report.model_dump()
    fr["decision"] = decision
    write_json(paths.final_report, fr)

    return RunResult(run_id=run_id, decision=decision, run_dir=str(paths.run_dir))
