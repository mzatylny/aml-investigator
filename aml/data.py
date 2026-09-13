"""An original, deliberately simplified synthetic transaction generator."""

import numpy as np
import pandas as pd


def generate_transactions(seed=42, days=90, accounts=500, daily_transactions=250):
    if days < 10 or accounts < 20 or daily_transactions < 20:
        raise ValueError("Use at least 10 days, 20 accounts and 20 daily transactions.")
    rng = np.random.default_rng(seed)
    names = np.array([f"AC{n:05d}" for n in rng.permutation(accounts)])
    activity = rng.lognormal(0, 0.7, accounts)
    activity /= activity.sum()
    preferred = rng.integers(0, accounts, (accounts, 8))
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []

    def add(day, minute, source, destination, amount, label=0, pattern="background", scenario=""):
        if source == destination:
            destination = (destination + 1) % accounts
        rows.append({
            "timestamp": start + pd.Timedelta(days=day, minutes=float(minute)),
            "sender": names[source], "receiver": names[destination],
            "amount": round(max(1, float(amount)), 2), "currency": "EUR",
            "is_laundering": label, "pattern": pattern, "scenario_id": scenario,
        })

    for day in range(days):
        for _ in range(int(rng.poisson(daily_transactions))):
            source = int(rng.choice(accounts, p=activity))
            destination = int(rng.choice(preferred[source]) if rng.random() < 0.7 else rng.integers(accounts))
            minute = float(rng.uniform(0, 1440))
            add(day, minute, source, destination, rng.lognormal(5.8, 1.35))

        # Each day includes both illicit motifs and legitimate lookalikes.
        # Same account pool, amount distribution, currency and time range for both.
        for label in (0, 1):
            for motif in ("fan_in", "fan_out", "cycle", "chain"):
                ids = rng.choice(accounts, 7, replace=False)
                base = float(rng.uniform(60, 1150))
                amount = float(rng.lognormal(7.2, 0.75))
                step = float(rng.uniform(1, 16) if label else rng.uniform(3, 35))
                scenario = f"D{day:03d}-{label}-{motif}"
                pattern = motif if label else f"legitimate_{motif}"
                if motif == "fan_in":
                    total = 0.0
                    for j in range(1, 6):
                        part = amount * rng.uniform(0.65, 1.35)
                        total += part
                        add(day, base + j * step, ids[j], ids[0], part, label, pattern, scenario)
                    add(day, base + 6 * step, ids[0], ids[6], total * rng.uniform(0.8, 1), label, pattern, scenario)
                elif motif == "fan_out":
                    add(day, base, ids[6], ids[0], 5 * amount, label, pattern, scenario)
                    for j in range(1, 6):
                        add(day, base + j * step, ids[0], ids[j], amount * rng.uniform(0.7, 1.2), label, pattern, scenario)
                else:
                    length = 4 if motif == "cycle" else 5
                    for j in range(length):
                        destination = ids[(j + 1) % length] if motif == "cycle" else ids[j + 1]
                        add(day, base + j * step, ids[j], destination, amount * rng.uniform(0.85, 1.05), label, pattern, scenario)

    frame = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame.insert(0, "transaction_id", [f"TX{n:08d}" for n in range(len(frame))])
    return frame


def validate_transactions(frame, require_labels=False):
    required = {"transaction_id", "timestamp", "sender", "receiver", "amount", "currency"}
    if require_labels:
        required.add("is_laundering")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("The transaction file is empty.")
    result = frame.copy()
    for col in ("transaction_id", "sender", "receiver", "currency"):
        if result[col].isna().any() or result[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} must contain non-empty values.")
        result[col] = result[col].astype(str).str.strip()
    if result.transaction_id.duplicated().any():
        raise ValueError("transaction_id must be unique.")
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True, format="ISO8601", errors="coerce")
    if result.timestamp.isna().any():
        raise ValueError("Invalid timestamp; use ISO 8601, preferably with a UTC offset.")
    result["amount"] = pd.to_numeric(result.amount, errors="coerce")
    if not np.isfinite(result.amount).all() or result.amount.le(0).any():
        raise ValueError("amount must contain finite, positive numbers.")
    if not result.currency.eq("EUR").all():
        raise ValueError("This demo accepts EUR only. Convert amounts consistently before import.")
    if result.sender.eq(result.receiver).any():
        raise ValueError("Sender and receiver must be different accounts.")
    if "is_laundering" in result:
        result["is_laundering"] = pd.to_numeric(result.is_laundering, errors="coerce")
        if not result.is_laundering.isin([0, 1]).all():
            raise ValueError("is_laundering must contain only 0 or 1.")
        result["is_laundering"] = result.is_laundering.astype(int)
    return result.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
