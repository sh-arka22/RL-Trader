"""Append-only trial log. Every evaluation ever run, hash-chained.

PLAN.md S3 requires an immutable trial log, for one reason: the Deflated Sharpe Ratio and
PBO both need the TRUE number of trials. A researcher who runs 200 configurations and
reports the best of them, declaring n_trials=1, produces a number that is not merely
optimistic but undefined. The count cannot be self-reported honestly after the fact, so it
is recorded as it happens and chained so that a later edit is detectable.

Each record links to the previous by hash. `verify()` walks the chain; deleting or editing
any past line breaks it.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import pathlib


GENESIS = "0" * 64


def _hash(prev: str, payload: str) -> str:
    return hashlib.sha256((prev + payload).encode()).hexdigest()


class TrialLog:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = None
        with open(self.path) as fh:
            for line in fh:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else GENESIS

    def append(self, kind: str, params: dict, metrics: dict, note: str = "") -> dict:
        prev = self._tail_hash()
        body = {"ts": dt.datetime.now(dt.UTC).isoformat(), "kind": kind,
                "params": params, "metrics": metrics, "note": note, "prev": prev}
        payload = json.dumps(body, sort_keys=True, default=str)
        rec = {**body, "hash": _hash(prev, payload)}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        return rec

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, rec in enumerate(self.records()):
            body = {k: rec[k] for k in ("ts", "kind", "params", "metrics", "note", "prev")}
            expect = _hash(prev, json.dumps(body, sort_keys=True, default=str))
            if rec["prev"] != prev:
                return False, f"record {i}: prev mismatch"
            if rec["hash"] != expect:
                return False, f"record {i}: hash mismatch — the record was edited"
            prev = rec["hash"]
        return True, f"{len(self.records())} records, chain intact"

    def n_trials(self, kind: str | None = None) -> int:
        """The honest denominator for a Deflated Sharpe Ratio."""
        return sum(1 for r in self.records() if kind is None or r["kind"] == kind)
