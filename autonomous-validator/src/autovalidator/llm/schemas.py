from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class ColumnSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    inferred_type: Literal["string", "integer", "number", "boolean", "date", "datetime", "unknown"]
    example_values_masked: list[str] = Field(default_factory=list)
    null_rate: float = Field(ge=0.0, le=1.0)
    unique_rate: float = Field(ge=0.0, le=1.0)
    top_values_masked: list[str] = Field(default_factory=list)
    length_min: Optional[int] = Field(default=None, ge=0)
    length_max: Optional[int] = Field(default=None, ge=0)

    # Optional profiling enrichments (generic)
    pattern_samples: list[str] = Field(default_factory=list)
    numeric_stats: Optional[dict[str, float]] = None
    date_stats: Optional[dict[str, str]] = None


class ProfileSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "unknown_dataset"
    row_count_estimate: int = Field(ge=0)
    columns: list[ColumnSummary]
    notes: list[str] = Field(default_factory=list)


class DomainHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ColumnRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    role: Literal[
        "id",
        "name",
        "amount",
        "currency",
        "country",
        "date",
        "timestamp",
        "status",
        "category",
        "free_text",
        "tax_id",
        "email",
        "phone",
        "address",
        "medical_measurement",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class DomainInference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: list[DomainHypothesis]
    column_roles: list[ColumnRole]
    assumptions: list[str] = Field(default_factory=list)
    questions_to_reduce_uncertainty: list[str] = Field(default_factory=list)


CheckType = Literal[
    "not_null",
    "unique",
    "value_range",
    "length_range",
    "regex_match",
    "outlier_iqr",
    "date_not_future",
    "category_newness",
    "cross_field_dependency",
]


class CandidateCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    check_type: CheckType
    title: str
    rationale: str
    assumption: str
    severity_suggestion: Literal["info", "warn", "fail", "quarantine"]
    confidence: float = Field(ge=0.0, le=1.0)
    params: dict[str, Any]


class CandidateChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: list[CandidateCheck]


class EvidenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    check_type: CheckType
    total_evaluated: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    violation_rate: float = Field(ge=0.0, le=1.0)
    examples_masked: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    outcome: Literal["pass", "warn", "fail", "quarantine"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence_used: str
    alternative_explanations: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["PASS", "PASS_WITH_WARNINGS", "FAIL", "QUARANTINE"]
    overall_confidence: float = Field(ge=0.0, le=1.0)

    inferred_domains: list[DomainHypothesis]
    key_assumptions: list[str]
    top_issues: list[str]

    findings: list[Finding]

    human_summary: str
    audit_notes: list[str] = Field(default_factory=list)
