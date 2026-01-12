from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yaml

from autovalidator.llm.schemas import FinalReport


DEFAULT_POLICY_YAML = """quarantine:
  sensitive_score_threshold: 0.6
fail:
  high_confidence_threshold: 0.8
  overall_confidence_threshold: 0.75
warn:
  overall_confidence_threshold: 0.4
"""


@dataclass
class DecisionPolicyConfig:
    sensitive_score_threshold: float = 0.6
    high_confidence_threshold: float = 0.8
    overall_confidence_fail: float = 0.75
    overall_confidence_warn: float = 0.4

    @staticmethod
    def from_yaml(path: Optional[str] = None) -> "DecisionPolicyConfig":
        data = yaml.safe_load(DEFAULT_POLICY_YAML) if path is None else yaml.safe_load(open(path, "r", encoding="utf-8"))
        return DecisionPolicyConfig(
            sensitive_score_threshold=float(data["quarantine"]["sensitive_score_threshold"]),
            high_confidence_threshold=float(data["fail"]["high_confidence_threshold"]),
            overall_confidence_fail=float(data["fail"]["overall_confidence_threshold"]),
            overall_confidence_warn=float(data["warn"]["overall_confidence_threshold"]),
        )


def apply_policy(llm_report: FinalReport, sensitive_score: float, cfg: DecisionPolicyConfig) -> str:
    """Deterministic final gate. LLM recommends; policy enforces."""
    if sensitive_score >= cfg.sensitive_score_threshold:
        return "QUARANTINE"
    # accept LLM decision if consistent with confidence bands
    if llm_report.decision == "FAIL" and llm_report.overall_confidence >= cfg.overall_confidence_warn:
        return "FAIL"
    if llm_report.decision == "QUARANTINE":
        return "QUARANTINE"
    if llm_report.decision == "PASS_WITH_WARNINGS":
        return "PASS_WITH_WARNINGS"
    if llm_report.decision == "PASS":
        return "PASS"
    # fallback based on confidence
    if llm_report.overall_confidence >= cfg.overall_confidence_fail:
        return "FAIL"
    if llm_report.overall_confidence >= cfg.overall_confidence_warn:
        return "PASS_WITH_WARNINGS"
    return "PASS_WITH_WARNINGS"
