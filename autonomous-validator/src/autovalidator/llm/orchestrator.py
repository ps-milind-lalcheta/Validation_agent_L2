from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

from pydantic import BaseModel

from .client import LLMClient
from .schemas import (
    ProfileSummary,
    DomainInference,
    CandidateChecks,
    EvidenceResult,
    FinalReport,
)
from . import prompts


@dataclass
class OrchestratorResult:
    domain_inference: DomainInference
    candidate_checks: CandidateChecks
    evidence_results: list[EvidenceResult]
    final_report: FinalReport


class ValidationOrchestrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    @staticmethod
    def _schema_json(model: type[BaseModel]) -> str:
        return json.dumps(model.model_json_schema(), ensure_ascii=False)

    def infer_domain(self, profile: ProfileSummary) -> DomainInference:
        user = prompts.domain_inferencer_user(
            profile_summary_json=profile.model_dump_json(indent=2),
            schema_json=self._schema_json(DomainInference),
        )
        return self.llm.chat_json(prompts.DOMAIN_INFERENCER_SYSTEM, user, DomainInference)

    def generate_checks(self, profile: ProfileSummary, domain: DomainInference) -> CandidateChecks:
        user = prompts.check_generator_user(
            profile_summary_json=profile.model_dump_json(indent=2),
            domain_inference_json=domain.model_dump_json(indent=2),
            schema_json=self._schema_json(CandidateChecks),
        )
        return self.llm.chat_json(prompts.CHECK_GENERATOR_SYSTEM, user, CandidateChecks)

    def skeptic(self, profile: ProfileSummary, domain: DomainInference, checks: CandidateChecks, evidence: Sequence[EvidenceResult]) -> FinalReport:
        user = prompts.skeptic_user(
            profile_summary_json=profile.model_dump_json(indent=2),
            domain_inference_json=domain.model_dump_json(indent=2),
            candidate_checks_json=checks.model_dump_json(indent=2),
            evidence_results_json=json.dumps([e.model_dump() for e in evidence], indent=2, ensure_ascii=False),
            schema_json=self._schema_json(FinalReport),
        )
        return self.llm.chat_json(prompts.SKEPTIC_SYSTEM, user, FinalReport)

    def run(self, profile: ProfileSummary, evidence_results: Sequence[EvidenceResult]) -> OrchestratorResult:
        domain = self.infer_domain(profile)
        checks = self.generate_checks(profile, domain)
        final_report = self.skeptic(profile, domain, checks, evidence_results)
        return OrchestratorResult(
            domain_inference=domain,
            candidate_checks=checks,
            evidence_results=list(evidence_results),
            final_report=final_report,
        )
