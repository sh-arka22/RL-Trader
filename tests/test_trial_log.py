"""An editable log is not evidence."""
import json
from rltrader.eval.trial_log import TrialLog


def test_chain_verifies(tmp_path):
    log = TrialLog(tmp_path / "t.jsonl")
    for i in range(5):
        log.append("evaluation", {"i": i}, {"sharpe": i / 10})
    ok, msg = log.verify()
    assert ok, msg
    assert log.n_trials() == 5


def test_editing_a_past_record_is_detected(tmp_path):
    p = tmp_path / "t.jsonl"
    log = TrialLog(p)
    for i in range(4):
        log.append("evaluation", {"i": i}, {"sharpe": 0.1})
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    recs[1]["metrics"]["sharpe"] = 9.99          # flatter an old trial
    p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n")
    ok, msg = log.verify()
    assert not ok and "record 1" in msg


def test_deleting_a_record_is_detected(tmp_path):
    """The motive for deletion is to shrink the DSR denominator."""
    p = tmp_path / "t.jsonl"
    log = TrialLog(p)
    for i in range(4):
        log.append("evaluation", {"i": i}, {"sharpe": 0.1})
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert not log.verify()[0]


def test_counts_are_per_kind(tmp_path):
    log = TrialLog(tmp_path / "t.jsonl")
    log.append("evaluation", {}, {})
    log.append("hpo", {}, {})
    log.append("evaluation", {}, {})
    assert log.n_trials("evaluation") == 2 and log.n_trials() == 3
