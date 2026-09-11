import importlib.util
from pathlib import Path
import tempfile

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


spec = importlib.util.spec_from_file_location(
    "memory_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_str_memory_comparison.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write_events(path, samples, stamp):
    writer = EventFileWriter(str(path), filename_suffix=f".{stamp}")
    for step, value in samples:
        writer.add_event(Event(wall_time=stamp, step=step, summary=Summary(value=[
            Summary.Value(tag=tag, simple_value=value) for tag in summary.TAGS.values()])))
    writer.close()


def test_pending_multiple_files_and_matched_progress():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "new"
        baseline = Path(directory) / "baseline"
        a = baseline / "runs/str_full_nodr/full_distribution_seed0/run1"
        b = root / "runs/str_memory_comparison/state162_ff_seed0/run1"
        write_events(a, [(1_000_000, 300), (8_000_000, 600)], 1)
        write_events(a, [(8_000_000, 900), (100_000_000, 1200)], 2)
        write_events(b, [(1_000_000, 150), (9_000_000, 300)], 3)
        report = summary.build_report(root, baseline)
        aruns = report["groups"]["A"]["0"]["runs"]
        assert len(aruns) == 1  # Two files in one directory, not two runs.
        assert report["groups"]["C"]["0"]["status"] == "PENDING"
        assert report["groups"]["B"]["1"]["runs"] == []
        bsummary = report["groups"]["B"]["0"]["runs"][0]
        assert bsummary["milestones"]["20000000"]["status"] == "PENDING"
        match = report["matched"][0]["runs"][0]
        assert match["end_transition"] == 9_000_000
        metrics = match["A"]["metrics"]
        assert metrics["return"]["windows"] == 2  # Duplicate8M step counted once.
        assert metrics["return"]["mean"] == 600  # Excludes100M and old8M value.
        assert metrics["return"]["last_step"] == 8_000_000  # Never looks ahead.
        assert metrics["lift_event_fraction"]["mean"] == 2
        assert "PENDING" in summary.markdown(report)


def test_nonfinite_is_not_silently_averaged():
    stats = summary.window_stats({"return": [(1, float("nan"))]}, 1)
    assert stats["metrics"]["return"]["status"] == "NONFINITE"
    assert stats["metrics"]["return"]["mean"] is None
    assert stats["metrics"]["entropy"] is None


if __name__ == "__main__":
    test_pending_multiple_files_and_matched_progress()
    test_nonfinite_is_not_silently_averaged()
    print("2 summary checks passed")
