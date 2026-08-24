"""Shared feature engineering for training and scoring.

Both train.py and scoring/score.py must turn a raw return record into the
exact same feature vector, so this module is the single source of truth.
"""
import pandas as pd

NUMERIC_FEATURES = [
    "customer_account_age_days",
    "customer_past_orders_count",
    "customer_past_returns_count",
    "customer_past_return_rate",
    "order_value",
    "days_since_delivery",
    "device_fingerprint_reuse_count",
]

BOOLEAN_FEATURES = [
    "is_high_value_electronics",
    "shipping_billing_mismatch_flag",
    "refund_to_different_method_flag",
]

CATEGORICAL_FEATURES = ["item_category", "return_reason_code"]

RAW_FIELDS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None):
    """Turn raw records into a numeric feature matrix.

    If feature_columns is given (the column list saved at train time), the
    result is reindexed to exactly that column set so inference-time records
    line up with what the model was trained on, even if a category value
    wasn't present in this batch.
    """
    work = df.copy()
    for col in BOOLEAN_FEATURES:
        work[col] = work[col].astype(int)

    encoded = pd.get_dummies(work[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    X = pd.concat([work[NUMERIC_FEATURES + BOOLEAN_FEATURES], encoded], axis=1)
    X = X.astype(float)

    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)

    return X
