import pandas as pd
from autovalidator.core.redaction import redact_dataframe


def test_redaction_masks_email_and_phone():
    df = pd.DataFrame({
        "email": ["test.user@example.com", None],
        "phone": ["+1 (555) 123-4567", "9999999999"],
        "note": ["hello", "world"],
    })
    res = redact_dataframe(df)
    red = res.redacted_df
    assert red.loc[0, "email"] != "test.user@example.com"
    assert "@" in red.loc[0, "email"]
    assert red.loc[0, "phone"] != "+1 (555) 123-4567"
    assert red.loc[0, "note"] == "hello"
