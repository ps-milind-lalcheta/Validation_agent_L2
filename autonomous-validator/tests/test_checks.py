import pandas as pd
from autovalidator.llm.schemas import CandidateCheck
from autovalidator.checks.executor import execute_check


def test_not_null_check():
    df = pd.DataFrame({"a": [1, None, 3]})
    chk = CandidateCheck(
        check_id="c1",
        check_type="not_null",
        title="a not null",
        rationale="test",
        assumption="none",
        severity_suggestion="warn",
        confidence=0.5,
        params={"column": "a", "max_null_rate": 0.0},
    )
    ev = execute_check(df, chk)
    assert ev.violation_count == 1
    assert ev.total_evaluated == 3


def test_regex_match():
    df = pd.DataFrame({"gst": ["ABCDE1234F", "bad", None]})
    chk = CandidateCheck(
        check_id="c2",
        check_type="regex_match",
        title="pattern",
        rationale="test",
        assumption="none",
        severity_suggestion="warn",
        confidence=0.5,
        params={"column": "gst", "pattern": "^[A-Z]{5}\\d{4}[A-Z]$","min_match_rate":0.8},
    )
    ev = execute_check(df, chk)
    assert ev.total_evaluated == 2
    assert ev.violation_count == 1
