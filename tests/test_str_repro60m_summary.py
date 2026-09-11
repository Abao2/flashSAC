from scripts.summarize_str_repro60m import checkpoint, read_evaluation, render, score


def test_missing_results_stay_pending_and_partial_is_explicit(tmp_path):
    assert checkpoint(tmp_path / "missing", 60002304)["status"] == "pending"
    pending = read_evaluation(tmp_path / "summary.json", tmp_path / "checkpoint")
    assert pending["status"] == "pending" and score(pending) == "PENDING"
    assert score({"counts": {"success": 2}, "episodes": 4, "complete_budget": False}) == "PARTIAL 2/4"
    item = {"transitions": 60002304, "online_brackets": [],
            "evaluations": {"deterministic": pending, "stochastic": pending}}
    report = {"created_utc": "test", "queue_status": "running", "protocol": "native", "limitations": [],
              "seeds": {"0": {"training_status": "running", "checkpoints": {"58596": item}}}}
    text = render(report)
    assert "|0|60,002,304|running|PENDING (checkpoint not saved)|PENDING|PENDING|" in text
    assert "0/128" not in text and "0/64" not in text
