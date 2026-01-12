from __future__ import annotations


DOMAIN_INFERENCER_SYSTEM = """You are Domain Inferencer.
Infer likely dataset domain(s) and column roles from a profile summary of a masked sample.
Be careful: do not invent facts. Use only evidence from the input.
Output MUST be valid JSON matching the provided JSON schema.
No markdown, no extra keys.
"""


CHECK_GENERATOR_SYSTEM = """You are Check Generator.
Propose candidate validation checks as strict JSON using the supported check types.
You are NOT allowed to decide PASS/FAIL for the dataset. Only propose checks.
You must not rely on an internal knowledge base; use only general reasoning plus dataset signals.
When you use general world knowledge, express it as an assumption and provide confidence.
Output MUST be valid JSON matching the provided JSON schema. No markdown, no extra keys.
"""


SKEPTIC_SYSTEM = """You are Skeptic/Challenger.
Your job: reduce false positives, challenge weak assumptions, and downgrade certainty when needed.
You will receive domain inference, proposed checks, and evidence results (from deterministic executor).
You must produce a FinalReport JSON.
Rules:
- Prefer WARN over FAIL when uncertainty is material.
- If evidence is weak (small sample, ambiguous semantics), say so.
- If sensitive data likely present, consider QUARANTINE recommendation.
Output MUST be valid JSON matching the provided JSON schema. No markdown, no extra keys.
"""


REPORT_WRITER_SYSTEM = """You are Report Writer.
You produce a final report and decision using domain hypotheses + findings.
Decision must be one of: PASS, PASS_WITH_WARNINGS, FAIL, QUARANTINE.
Be concise and auditable: mention assumptions, evidence highlights, and next steps.
Output MUST be valid JSON matching the provided JSON schema. No markdown, no extra keys.
"""


def domain_inferencer_user(profile_summary_json: str, schema_json: str) -> str:
    return f"""Infer dataset domain(s) and column roles.

PROFILE_SUMMARY_JSON:
{profile_summary_json}

OUTPUT_JSON_SCHEMA:
{schema_json}
"""


def check_generator_user(profile_summary_json: str, domain_inference_json: str, schema_json: str) -> str:
    return f"""Propose 8-20 candidate checks using the supported check DSL.
- Prefer checks that are broadly applicable across domains.
- Each check must include: check_id, check_type, title, rationale, assumption, severity_suggestion, confidence, params.
- Keep params realistic and executable.

PROFILE_SUMMARY_JSON:
{profile_summary_json}

DOMAIN_INFERENCE_JSON:
{domain_inference_json}

OUTPUT_JSON_SCHEMA:
{schema_json}
"""


def skeptic_user(
    profile_summary_json: str,
    domain_inference_json: str,
    candidate_checks_json: str,
    evidence_results_json: str,
    schema_json: str,
) -> str:
    return f"""Review evidence and produce the final report.
- Use evidence_results to justify outcomes.
- If assumptions are weak, lower confidence and/or downgrade severity.

PROFILE_SUMMARY_JSON:
{profile_summary_json}

DOMAIN_INFERENCE_JSON:
{domain_inference_json}

CANDIDATE_CHECKS_JSON:
{candidate_checks_json}

EVIDENCE_RESULTS_JSON:
{evidence_results_json}

OUTPUT_JSON_SCHEMA:
{schema_json}
"""


def report_writer_user(domain_inference_json: str, findings_json: str, schema_json: str) -> str:
    return f"""Create the final report.
- inferred_domains should reflect the domain hypotheses (may reorder by confidence).
- top_issues: short strings.
- human_summary: 1-2 short paragraphs.

DOMAIN_INFERENCE_JSON:
{domain_inference_json}

FINDINGS_JSON:
{findings_json}

OUTPUT_JSON_SCHEMA:
{schema_json}
"""
