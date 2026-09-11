"""Summarize a DexToolBench batch-evaluation directory.

``run_all_evals_isaacgym.py`` writes one result per benchmark combination as::

    <run_dir>/<category>/<object>/<task>/<policy>/eval.json

This module validates those files against the 24 DexToolBench combinations and
prints per-combination results, per-category means, per-policy overall means,
and separate lists of missing and invalid result files.  It intentionally has
no Isaac Gym or GPU dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CATEGORY_TO_OBJECT_NAMES: Mapping[str, Tuple[str, ...]] = {
    "hammer": ("claw_hammer", "mallet_hammer"),
    "spatula": ("flat_spatula", "spoon_spatula"),
    "eraser": ("flat_eraser", "handle_eraser"),
    "screwdriver": ("long_screwdriver", "short_screwdriver"),
    "marker": ("sharpie_marker", "staples_marker"),
    "brush": ("blue_brush", "red_brush"),
}

CATEGORY_TO_TASK_NAMES: Mapping[str, Tuple[str, ...]] = {
    "hammer": ("swing_down", "swing_side"),
    "spatula": ("serve_plate", "flip_over"),
    "eraser": ("wipe_smile", "wipe_c"),
    "screwdriver": ("spin_vertical", "spin_horizontal"),
    "marker": ("draw_smile", "write_c"),
    "brush": ("sweep_forward", "sweep_right"),
}

DEFAULT_POLICY = "pretrained_policy"


@dataclass(frozen=True)
class Combination:
    category: str
    object_name: str
    task_name: str
    policy_name: str

    @property
    def label(self) -> str:
        return "/".join(
            (self.category, self.object_name, self.task_name, self.policy_name)
        )


@dataclass(frozen=True)
class EvalResult:
    combination: Combination
    path: Path
    status: str
    avg_goal_pct: Optional[float] = None
    avg_time_sec: Optional[float] = None
    num_episodes: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class MeanResult:
    completed: int
    expected: int
    avg_goal_pct: Optional[float]
    avg_time_sec: Optional[float]


@dataclass(frozen=True)
class EvalSummary:
    run_dir: Path
    policies: Tuple[str, ...]
    results: Tuple[EvalResult, ...]

    @property
    def successful(self) -> Tuple[EvalResult, ...]:
        return tuple(result for result in self.results if result.status == "ok")

    @property
    def missing(self) -> Tuple[EvalResult, ...]:
        return tuple(result for result in self.results if result.status == "missing")

    @property
    def failed(self) -> Tuple[EvalResult, ...]:
        return tuple(result for result in self.results if result.status == "failed")

    @property
    def is_complete(self) -> bool:
        return not self.missing and not self.failed

    def category_means(self) -> Dict[Tuple[str, str], MeanResult]:
        means: Dict[Tuple[str, str], MeanResult] = {}
        for category in CATEGORY_TO_OBJECT_NAMES:
            for policy in self.policies:
                selected = [
                    result
                    for result in self.results
                    if result.combination.category == category
                    and result.combination.policy_name == policy
                ]
                means[(category, policy)] = _mean_result(selected)
        return means

    def overall_means(self) -> Dict[str, MeanResult]:
        return {
            policy: _mean_result(
                [
                    result
                    for result in self.results
                    if result.combination.policy_name == policy
                ]
            )
            for policy in self.policies
        }


class EvalValidationError(ValueError):
    """Raised when an ``eval.json`` file does not match the expected schema."""


def expected_combinations(policies: Iterable[str]) -> Tuple[Combination, ...]:
    """Return all benchmark combinations in the batch runner's stable order."""

    policies = tuple(policies)
    combinations: List[Combination] = []
    for category, object_names in CATEGORY_TO_OBJECT_NAMES.items():
        for object_name in object_names:
            for task_name in CATEGORY_TO_TASK_NAMES[category]:
                for policy_name in policies:
                    combinations.append(
                        Combination(category, object_name, task_name, policy_name)
                    )
    return tuple(combinations)


def discover_policies(run_dir: Path) -> Tuple[str, ...]:
    """Discover policy directory names under valid benchmark task paths."""

    policies = set()
    for combination in expected_combinations(("",)):
        task_dir = (
            run_dir
            / combination.category
            / combination.object_name
            / combination.task_name
        )
        if not task_dir.is_dir():
            continue
        try:
            policies.update(
                child.name for child in task_dir.iterdir() if child.is_dir()
            )
        except OSError:
            # The corresponding combinations will be reported as missing below.
            continue
    return tuple(sorted(policies))


def _normalise_policies(
    policies: Optional[Sequence[str]], run_dir: Path
) -> Tuple[str, ...]:
    selected = tuple(dict.fromkeys(policies or discover_policies(run_dir)))
    if not selected:
        selected = (DEFAULT_POLICY,)
    for policy in selected:
        if not policy or policy in (".", "..") or Path(policy).name != policy:
            raise ValueError(f"invalid policy directory name: {policy!r}")
    return selected


def _number(
    data: Mapping[str, object],
    key: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalValidationError(f"{key} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise EvalValidationError(f"{key} must be finite")
    if minimum is not None and value < minimum:
        raise EvalValidationError(f"{key} must be >= {minimum:g}")
    if maximum is not None and value > maximum:
        raise EvalValidationError(f"{key} must be <= {maximum:g}")
    return value


def _number_list(
    data: Mapping[str, object],
    key: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> List[float]:
    values = data.get(key)
    if not isinstance(values, list) or not values:
        raise EvalValidationError(f"{key} must be a non-empty list")
    parsed = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvalValidationError(f"{key}[{index}] must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise EvalValidationError(f"{key}[{index}] must be finite")
        if minimum is not None and value < minimum:
            raise EvalValidationError(f"{key}[{index}] must be >= {minimum:g}")
        if maximum is not None and value > maximum:
            raise EvalValidationError(f"{key}[{index}] must be <= {maximum:g}")
        parsed.append(value)
    return parsed


def load_eval(path: Path) -> Tuple[float, float, int]:
    """Load and validate one batch-runner ``eval.json`` file."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise EvalValidationError("top-level JSON value must be an object")

    avg_goal_pct = _number(data, "avg_goal_pct", minimum=0.0, maximum=100.0)
    avg_time_sec = _number(data, "avg_time_sec", minimum=0.0)
    episode_goal_pcts = _number_list(
        data, "episode_goal_pcts", minimum=0.0, maximum=100.0
    )
    episode_lengths = _number_list(data, "episode_lengths", minimum=0.0)

    if len(episode_goal_pcts) != len(episode_lengths):
        raise EvalValidationError(
            "episode_goal_pcts and episode_lengths must have equal lengths"
        )
    calculated_goal_pct = sum(episode_goal_pcts) / len(episode_goal_pcts)
    if not math.isclose(avg_goal_pct, calculated_goal_pct, abs_tol=1e-6):
        raise EvalValidationError(
            "avg_goal_pct does not match the mean of episode_goal_pcts"
        )
    return avg_goal_pct, avg_time_sec, len(episode_goal_pcts)


def summarize_run(
    run_dir: Path, policies: Optional[Sequence[str]] = None
) -> EvalSummary:
    """Collect and validate results from one batch-evaluation run directory."""

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise ValueError(f"run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise ValueError(f"run path is not a directory: {run_dir}")

    selected_policies = _normalise_policies(policies, run_dir)
    results = []
    for combination in expected_combinations(selected_policies):
        path = (
            run_dir
            / combination.category
            / combination.object_name
            / combination.task_name
            / combination.policy_name
            / "eval.json"
        )
        if not path.exists():
            results.append(
                EvalResult(
                    combination,
                    path,
                    "missing",
                    error="eval.json not found",
                )
            )
            continue
        if not path.is_file():
            results.append(
                EvalResult(
                    combination,
                    path,
                    "failed",
                    error="eval.json is not a regular file",
                )
            )
            continue
        try:
            avg_goal_pct, avg_time_sec, num_episodes = load_eval(path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            EvalValidationError,
        ) as error:
            results.append(
                EvalResult(combination, path, "failed", error=str(error))
            )
        else:
            results.append(
                EvalResult(
                    combination,
                    path,
                    "ok",
                    avg_goal_pct=avg_goal_pct,
                    avg_time_sec=avg_time_sec,
                    num_episodes=num_episodes,
                )
            )
    return EvalSummary(run_dir, selected_policies, tuple(results))


def _mean_result(results: Sequence[EvalResult]) -> MeanResult:
    successful = [result for result in results if result.status == "ok"]
    if not successful:
        return MeanResult(0, len(results), None, None)
    goal_values = [
        result.avg_goal_pct
        for result in successful
        if result.avg_goal_pct is not None
    ]
    time_values = [
        result.avg_time_sec
        for result in successful
        if result.avg_time_sec is not None
    ]
    return MeanResult(
        len(successful),
        len(results),
        sum(goal_values) / len(goal_values),
        sum(time_values) / len(time_values),
    )


def _format_number(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        (format_row(headers), separator, *(format_row(row) for row in text_rows))
    )


def render_summary(summary: EvalSummary) -> str:
    """Render a deterministic, human-readable summary."""

    lines = [
        "DexToolBench evaluation summary",
        f"Run directory: {summary.run_dir}",
        (
            f"Expected: {len(summary.results)} | OK: {len(summary.successful)} | "
            f"Missing: {len(summary.missing)} | Failed: {len(summary.failed)}"
        ),
        "",
        "Combination details",
    ]
    detail_rows = []
    for result in summary.results:
        detail_rows.append(
            (
                result.combination.category,
                result.combination.object_name,
                result.combination.task_name,
                result.combination.policy_name,
                _format_number(result.avg_goal_pct),
                _format_number(result.avg_time_sec),
                result.num_episodes if result.num_episodes is not None else "-",
                result.status.upper(),
            )
        )
    lines.append(
        _format_table(
            (
                "category",
                "object",
                "task",
                "policy",
                "avg_goal_pct",
                "avg_time_sec",
                "episodes",
                "status",
            ),
            detail_rows,
        )
    )

    lines.extend(("", "Category means (successful combinations only)"))
    category_rows = []
    for (category, policy), mean in summary.category_means().items():
        category_rows.append(
            (
                category,
                policy,
                f"{mean.completed}/{mean.expected}",
                _format_number(mean.avg_goal_pct),
                _format_number(mean.avg_time_sec),
            )
        )
    lines.append(
        _format_table(
            ("category", "policy", "completed", "avg_goal_pct", "avg_time_sec"),
            category_rows,
        )
    )

    lines.extend(("", "Overall means (successful combinations only)"))
    overall_rows = []
    for policy, mean in summary.overall_means().items():
        overall_rows.append(
            (
                policy,
                f"{mean.completed}/{mean.expected}",
                _format_number(mean.avg_goal_pct),
                _format_number(mean.avg_time_sec),
            )
        )
    lines.append(
        _format_table(
            ("policy", "completed", "avg_goal_pct", "avg_time_sec"),
            overall_rows,
        )
    )

    lines.extend(("", f"Missing items ({len(summary.missing)})"))
    if summary.missing:
        lines.extend(
            f"- {result.combination.label}: {result.path.relative_to(summary.run_dir)}"
            for result in summary.missing
        )
    else:
        lines.append("- none")

    lines.extend(("", f"Failed items ({len(summary.failed)})"))
    if summary.failed:
        lines.extend(
            (
                f"- {result.combination.label}: {result.error} "
                f"({result.path.relative_to(summary.run_dir)})"
            )
            for result in summary.failed
        )
    else:
        lines.append("- none")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a DexToolBench batch-evaluation run."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="timestamped output directory produced by run_all_evals_isaacgym.py",
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        metavar="NAME",
        help=(
            "expected policy directory (repeatable); by default policy directories "
            f"are discovered, falling back to {DEFAULT_POLICY!r}"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="return exit code 0 even when results are missing or invalid",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_run(args.run_dir, args.policies)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(render_summary(summary))
    if not summary.is_complete and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
