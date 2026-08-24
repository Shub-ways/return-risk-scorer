from scoring.score import score_record

LOW_RISK_RECORD = dict(
    customer_account_age_days=800,
    customer_past_orders_count=20,
    customer_past_returns_count=1,
    customer_past_return_rate=0.05,
    order_value=1200,
    item_category="Books",
    is_high_value_electronics=False,
    days_since_delivery=10,
    return_reason_code="damaged",
    shipping_billing_mismatch_flag=False,
    device_fingerprint_reuse_count=0,
    refund_to_different_method_flag=False,
)

HIGH_RISK_RECORD = dict(
    customer_account_age_days=15,
    customer_past_orders_count=2,
    customer_past_returns_count=1,
    customer_past_return_rate=0.5,
    order_value=25000,
    item_category="Electronics",
    is_high_value_electronics=True,
    days_since_delivery=2,
    return_reason_code="changed_mind",
    shipping_billing_mismatch_flag=True,
    device_fingerprint_reuse_count=6,
    refund_to_different_method_flag=True,
)


def test_low_risk_record_is_auto_approved():
    result = score_record(LOW_RISK_RECORD)
    assert result["decision"] == "auto_approve"
    assert result["risk_score"] < 0.5


def test_high_risk_record_is_flagged():
    result = score_record(HIGH_RISK_RECORD)
    assert result["decision"] in ("hold_for_review", "auto_deny")
    assert result["risk_score"] > LOW_RISK_RECORD.get("risk_score", 0)


def test_top_features_present_for_logistic_model():
    result = score_record(HIGH_RISK_RECORD)
    if result["model_name"] == "logistic_regression":
        assert 1 <= len(result["top_features"]) <= 3
        assert all({"feature", "value", "contribution", "direction"} <= f.keys() for f in result["top_features"])
