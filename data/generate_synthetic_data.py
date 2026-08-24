"""Synthetic labeled returns dataset for the Return-Risk Scorer.

Generates a mix of legitimate returns and four injected abuse patterns
(serial returner, wardrobing, device-reuse ring, refund-redirect). A
fraction of fraud rows are deliberately given "quiet" (legit-looking)
features and a fraction of legit rows are given "loud" (fraud-looking)
features, so the classes overlap the way real fraud data does instead of
being trivially/perfectly separable.

Output: data/returns_dataset.csv
"""
import numpy as np
import pandas as pd

SEED = 42
N_RECORDS = 4000
FRAUD_RATE_TARGET = 0.10
QUIET_FRAUD_FRACTION = 0.12    # sophisticated fraud that looks mostly normal
LOUD_LEGIT_FRACTION = 0.04     # legit customers who happen to look risky

CATEGORIES = ["Electronics", "Apparel", "Home", "Beauty", "Books", "Sports"]
REASONS = ["wrong_item", "damaged", "not_as_described", "no_longer_needed", "changed_mind", "defective"]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def _legit_features(rng):
    account_age = int(_clip(rng.normal(400, 250), 5, 2000))
    past_orders = int(_clip(rng.poisson(8), 0, 100))
    base_return_p = rng.choice([0.06, 0.22], p=[0.88, 0.12])
    past_returns = int(_clip(rng.binomial(past_orders, base_return_p) if past_orders > 0 else 0, 0, past_orders))
    past_return_rate = round(past_returns / past_orders, 3) if past_orders > 0 else 0.0

    category = rng.choice(CATEGORIES)
    order_value = round(float(_clip(rng.lognormal(mean=6.5, sigma=0.7), 100, 60000)), 2)
    days_since_delivery = int(_clip(rng.exponential(6) + 1, 1, 30))
    reason = rng.choice(REASONS, p=[0.15, 0.25, 0.15, 0.15, 0.15, 0.15])
    shipping_billing_mismatch = rng.random() < 0.06
    device_reuse_count = int(rng.choice([0, 1, 2], p=[0.90, 0.08, 0.02]))
    refund_to_different_method = rng.random() < 0.05

    return dict(
        customer_account_age_days=account_age,
        customer_past_orders_count=past_orders,
        customer_past_returns_count=past_returns,
        customer_past_return_rate=past_return_rate,
        order_value=order_value,
        item_category=category,
        days_since_delivery=days_since_delivery,
        return_reason_code=reason,
        shipping_billing_mismatch_flag=bool(shipping_billing_mismatch),
        device_fingerprint_reuse_count=device_reuse_count,
        refund_to_different_method_flag=bool(refund_to_different_method),
    )


def _fraud_features(rng, pattern):
    if pattern == "serial_returner":
        account_age = int(_clip(rng.normal(300, 200), 10, 1800))
        past_orders = int(_clip(rng.poisson(14), 3, 120))
        past_returns = int(_clip(rng.binomial(past_orders, rng.uniform(0.35, 0.7)), 1, past_orders))
        past_return_rate = round(past_returns / past_orders, 3)
        category = rng.choice(CATEGORIES)
        order_value = round(float(_clip(rng.lognormal(mean=6.6, sigma=0.7), 100, 60000)), 2)
        days_since_delivery = int(_clip(rng.exponential(4) + 1, 1, 20))
        reason = rng.choice(["changed_mind", "no_longer_needed"], p=[0.6, 0.4])
        shipping_billing_mismatch = rng.random() < 0.08
        device_reuse_count = int(rng.choice([0, 1], p=[0.85, 0.15]))
        refund_to_different_method = rng.random() < 0.12

    elif pattern == "wardrobing":
        account_age = int(_clip(rng.normal(500, 300), 20, 2000))
        past_orders = int(_clip(rng.poisson(6), 0, 60))
        past_returns = int(_clip(rng.binomial(past_orders, 0.2) if past_orders > 0 else 0, 0, past_orders))
        past_return_rate = round(past_returns / past_orders, 3) if past_orders > 0 else 0.0
        category = rng.choice(["Apparel", "Electronics"], p=[0.7, 0.3])
        order_value = round(float(_clip(rng.lognormal(mean=7.0, sigma=0.6), 1500, 60000)), 2)
        days_since_delivery = int(_clip(rng.exponential(3) + 1, 1, 14))
        reason = rng.choice(["no_longer_needed", "changed_mind"], p=[0.5, 0.5])
        shipping_billing_mismatch = rng.random() < 0.06
        device_reuse_count = int(rng.choice([0, 1], p=[0.88, 0.12]))
        refund_to_different_method = rng.random() < 0.15

    elif pattern == "device_ring":
        account_age = int(_clip(rng.normal(30, 20), 1, 120))
        past_orders = int(_clip(rng.poisson(1.5), 0, 10))
        past_returns = int(_clip(rng.binomial(past_orders, 0.3) if past_orders > 0 else 0, 0, past_orders))
        past_return_rate = round(past_returns / past_orders, 3) if past_orders > 0 else 0.0
        category = rng.choice(CATEGORIES)
        order_value = round(float(_clip(rng.lognormal(mean=6.8, sigma=0.7), 100, 60000)), 2)
        days_since_delivery = int(_clip(rng.exponential(5) + 1, 1, 25))
        reason = rng.choice(REASONS)
        shipping_billing_mismatch = rng.random() < 0.2
        device_reuse_count = int(_clip(rng.poisson(4), 1, 15))
        refund_to_different_method = rng.random() < 0.3

    else:  # refund_redirect
        account_age = int(_clip(rng.normal(70, 45), 1, 400))
        past_orders = int(_clip(rng.poisson(3), 0, 30))
        past_returns = int(_clip(rng.binomial(past_orders, 0.15) if past_orders > 0 else 0, 0, past_orders))
        past_return_rate = round(past_returns / past_orders, 3) if past_orders > 0 else 0.0
        category = rng.choice(CATEGORIES)
        order_value = round(float(_clip(rng.lognormal(mean=6.9, sigma=0.7), 100, 60000)), 2)
        days_since_delivery = int(_clip(rng.exponential(6) + 1, 1, 25))
        reason = rng.choice(REASONS)
        shipping_billing_mismatch = rng.random() < 0.4
        device_reuse_count = int(rng.choice([0, 0, 1, 2, 3], p=[0.35, 0.25, 0.15, 0.15, 0.10]))
        refund_to_different_method = rng.random() < 0.85

    return dict(
        customer_account_age_days=account_age,
        customer_past_orders_count=past_orders,
        customer_past_returns_count=past_returns,
        customer_past_return_rate=past_return_rate,
        order_value=order_value,
        item_category=category,
        days_since_delivery=days_since_delivery,
        return_reason_code=reason,
        shipping_billing_mismatch_flag=bool(shipping_billing_mismatch),
        device_fingerprint_reuse_count=device_reuse_count,
        refund_to_different_method_flag=bool(refund_to_different_method),
    )


def generate(n_records=N_RECORDS, fraud_rate=FRAUD_RATE_TARGET, seed=SEED):
    rng = np.random.default_rng(seed)

    n_fraud = int(n_records * fraud_rate)
    n_legit = n_records - n_fraud
    fraud_patterns = rng.choice(
        ["serial_returner", "wardrobing", "device_ring", "refund_redirect"],
        size=n_fraud,
        p=[0.3, 0.3, 0.2, 0.2],
    )

    rows = []

    for i in range(n_legit):
        # A minority of legit customers happen to trip fraud-shaped features
        # (shared devices, quick returns) — real false-positive bait.
        if rng.random() < LOUD_LEGIT_FRACTION:
            features = _fraud_features(rng, rng.choice(["serial_returner", "device_ring"]))
        else:
            features = _legit_features(rng)
        customer_id = f"cust_{rng.integers(1, 20000)}"
        is_high_value_electronics = features["item_category"] == "Electronics" and features["order_value"] > 15000
        rows.append(dict(
            return_id=f"ret_{i}",
            customer_id=customer_id,
            is_high_value_electronics=is_high_value_electronics,
            label_is_fraudulent_return=0,
            **features,
        ))

    for j, pattern in enumerate(fraud_patterns):
        i = n_legit + j
        # A chunk of fraud is "quiet" — sophisticated actors who don't trip
        # the obvious signals, capping how separable the classes really are.
        if rng.random() < QUIET_FRAUD_FRACTION:
            features = _legit_features(rng)
        else:
            features = _fraud_features(rng, pattern)
        customer_id = f"cust_{rng.integers(20000, 40000)}"
        is_high_value_electronics = features["item_category"] == "Electronics" and features["order_value"] > 15000
        rows.append(dict(
            return_id=f"ret_{i}",
            customer_id=customer_id,
            is_high_value_electronics=is_high_value_electronics,
            label_is_fraudulent_return=1,
            **features,
        ))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/returns_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} records to {out_path}")
    print(f"Fraud rate: {df['label_is_fraudulent_return'].mean():.3%}")
    print(df.groupby("label_is_fraudulent_return")[
        ["order_value", "device_fingerprint_reuse_count", "customer_past_return_rate"]
    ].mean())
