"""Causal features: read past events, then update state after each timestamp batch."""

from collections import defaultdict, deque
import numpy as np
import pandas as pd
from .data import validate_transactions

FEATURES = [
    "log_amount", "hour_sin", "hour_cos", "sender_out_count_24h", "sender_in_count_24h",
    "receiver_in_count_24h", "sender_unique_receivers_24h", "receiver_unique_senders_24h",
    "amount_to_sender_mean", "amount_to_inflow_24h", "minutes_since_inflow",
    "seen_pair_24h", "closes_cycle_24h",
]


def build_features(transactions):
    tx = validate_transactions(transactions)
    outgoing, incoming = defaultdict(deque), defaultdict(deque)
    values = []

    def history(table, account, time):
        events = table[account]
        while events and events[0][0] < time - pd.Timedelta(hours=24):
            events.popleft()
        return events

    def closes_cycle(source, destination, time):
        # Existing destination -> ... -> source path, at most three hops.
        frontier, seen = {destination}, {destination}
        for _ in range(3):
            next_frontier = set()
            for account in frontier:
                neighbours = {e[1] for e in history(outgoing, account, time)}
                if source in neighbours:
                    return 1
                next_frontier.update(neighbours - seen)
            seen.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return 0

    for time, batch in tx.groupby("timestamp", sort=False):
        for row in batch.itertuples(index=False):
            sent = history(outgoing, row.sender, time)
            received = history(incoming, row.sender, time)
            receiver_in = history(incoming, row.receiver, time)
            total_in = sum(e[2] for e in received)
            mean_out = sum(e[2] for e in sent) / len(sent) if sent else row.amount
            hour = time.hour + time.minute / 60
            values.append([
                np.log1p(row.amount), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                len(sent), len(received), len(receiver_in), len({e[1] for e in sent}),
                len({e[1] for e in receiver_in}), min(row.amount / max(mean_out, 1), 100),
                min(row.amount / max(total_in, 1), 100),
                (time - received[-1][0]).total_seconds() / 60 if received else 1440,
                int(any(e[1] == row.receiver for e in sent)), closes_cycle(row.sender, row.receiver, time),
            ])
        # Simultaneous events cannot see each other, regardless of ID/input order.
        for row in batch.itertuples(index=False):
            outgoing[row.sender].append((time, row.receiver, row.amount))
            incoming[row.receiver].append((time, row.sender, row.amount))
    return tx, pd.DataFrame(values, columns=FEATURES)


def rule_signals(features):
    return pd.DataFrame({
        "rapid_forwarding": (features.minutes_since_inflow < 60) & features.amount_to_inflow_24h.between(0.5, 1.2),
        "fan_in": features.receiver_unique_senders_24h.ge(4),
        "fan_out": features.sender_unique_receivers_24h.ge(4),
        "cycle": features.closes_cycle_24h.eq(1),
    })


def rule_scores(features):
    return rule_signals(features).mul([0.4, 0.2, 0.2, 0.4]).sum(axis=1).clip(upper=1).to_numpy()


def evidence(features):
    labels = {
        "rapid_forwarding": "Recent incoming funds followed by a similar outgoing amount",
        "fan_in": "Receiver already received from 4+ accounts in the past 24h",
        "fan_out": "Sender already paid 4+ accounts in the past 24h",
        "cycle": "Transfer closes a directed cycle within the past 24h",
    }
    return rule_signals(features).apply(
        lambda row: "; ".join(labels[key] for key, value in row.items() if value)
        or "No predefined rule fired; review the model score and account history", axis=1,
    )
