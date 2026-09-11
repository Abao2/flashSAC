"""Reproduce CPU-only reward figures from real TensorBoard events; no training imports."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parent
DIAG = ROOT.parent
METRICS = {
    "return": ("episode/return", "总回报", "分 / episode"),
    "fingertip": ("episode/cumulative/fingertip_delta_rew", "指尖靠近奖励", "分 / episode"),
    "lifting": ("episode/cumulative/lifting_rew", "连续抬升奖励", "分 / episode"),
    "lift_bonus": ("episode/cumulative/lift_bonus_rew", "一次性抬升奖励", "分 / episode"),
    "keypoint": ("episode/cumulative/keypoint_rew", "朝目标前进奖励", "分 / episode"),
    "goals": ("episode/final/successes", "完成目标数", "goals / episode"),
}


def load_events(directory, tags):
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = accumulator.Tags()["scalars"]
    curves = {}
    for key, tag in tags.items():
        events = accumulator.Scalars(tag) if tag in available else []
        unique = {event.step: float(event.value) for event in sorted(events, key=lambda event: event.wall_time)}
        points = sorted(unique.items())
        if any(not np.isfinite(value) for _, value in points):
            raise ValueError(f"Nonfinite scalar: {directory} / {tag}")
        curves[key] = points
    return curves


def one_run(parent):
    directories = sorted({p.parent for p in parent.rglob("events.out.tfevents*")})
    if len(directories) != 1:
        raise ValueError(f"Expected exactly one run, not silently merged restarts: {parent}: {directories}")
    return directories[0]


def bin_curve(points, width, limit=None):
    """Equal-weight TB-window means in fixed transition bins; no extrapolation."""
    groups = {}
    for step, value in points:
        if step < 0 or (limit is not None and step > limit):
            continue
        groups.setdefault(max(0, int((step - 1) // width)), []).append((step, value))
    return {index: (float(np.mean([p[0] for p in rows])), float(np.mean([p[1] for p in rows])), len(rows))
            for index, rows in sorted(groups.items())}


def local_data():
    tags = {key: fields[0] for key, fields in METRICS.items()}
    specs = [("full_seed0", "完整STR：A140 FF seed0",
              DIAG / "full_nodr_20260909_1004/runs/str_full_nodr/full_distribution_seed0")]
    specs += [(f"simple_seed{seed}", f"固定eraser：seed{seed}",
               DIAG / f"overnight_20260908/repro60m/runs/repro60m/arm1_hand01_seed{seed}") for seed in range(3)]
    live_parent = DIAG / "control_ab_20260909_arm1/runs/str_control_ab/full_arm1_seed0"
    if list(live_parent.rglob("events.out.tfevents*")):
        specs.append(("full_arm1_live", "完整STR：arm=1，对照训练", live_parent))
    result = {}
    for key, label, parent in specs:
        directory = one_run(parent)
        result[key] = {"label": label, "directory": str(directory), "tags": tags,
                       "event_files": [str(p) for p in sorted(directory.glob("events.out.tfevents*"))],
                       "curves": load_events(directory, tags)}
    return result


def style():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "axes.axisbelow": True})


def draw_local(data, sources):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7))
    red, green = "#C44E52", "#167D8D"
    for ax, (metric, (_, title, unit)) in zip(axes.flat, METRICS.items()):
        failed = bin_curve(data["full_seed0"]["curves"][metric], 1_000_000, 100_000_000)
        ax.plot([row[0] / 1e6 for row in failed.values()], [row[1] for row in failed.values()],
                color=red, lw=2.1, label="完整STR，A140 FF seed0，arm=.1")
        seeds = [bin_curve(data[f"simple_seed{s}"]["curves"][metric], 1_000_000, 60_000_000) for s in range(3)]
        # A band only where all three training seeds actually have observations.
        common = sorted(set.intersection(*(set(seed) for seed in seeds)))
        x = np.array([np.mean([seed[i][0] for seed in seeds]) for i in common]) / 1e6
        y = np.array([[seed[i][1] for i in common] for seed in seeds])
        for values in y:
            ax.plot(x, values, color=green, lw=.7, alpha=.20)
        ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=green, alpha=.16)
        ax.plot(x, np.median(y, axis=0), color=green, lw=2.1,
                label="固定eraser，state162 FF，arm=1：三seed中位数 / 全范围")
        for source in sources:
            if not source.get("include_early"):
                continue
            curve = bin_curve(source.get("early_curves", {}).get(metric, []), 1_000_000, 100_000_000)
            if curve:
                ax.plot([v[0] / 1e6 for v in curve.values()], [v[1] for v in curve.values()],
                        color="#7753A6", lw=1.7, ls="-.", label="原STR / SAPG：KUKA+Sharpa，全DR，早期100M")
        if "full_arm1_live" in data:
            live = bin_curve(data["full_arm1_live"]["curves"][metric], 1_000_000, 100_000_000)
            if live:
                end = max(p[0] for p in data["full_arm1_live"]["curves"][metric]) / 1e6
                ax.plot([v[0] / 1e6 for v in live.values()], [v[1] for v in live.values()],
                        color="#3366BB", lw=1.7, ls="--", marker="." if len(live) < 3 else None,
                        label=f"完整STR，仅arm=1：已记录{end:.1f}M")
        ax.set(title=title, xlabel="环境 transitions（百万，M）", ylabel=unit, xlim=(0, 100))
        ax.grid(alpha=.19)
        if metric == "lift_bonus":
            ax.axhline(300, color=".55", lw=.8, ls=":")
            ax.text(.98, .68, "300分 = 曾越过抬升阈值\n不等于稳定抓取", transform=ax.transAxes, va="top", ha="right", fontsize=9)
        if metric == "lifting":
            ax.text(.98, .95, "达到抬升阈值后此项关闭\n下降不一定代表退步", transform=ax.transAxes, va="top", ha="right", fontsize=9)
        if metric == "goals":
            ax.text(.98, .45, "固定任务最多1个\n完整任务最多50个\n不是同任务成功率", transform=ax.transAxes, va="top", ha="right", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=1, frameon=False, fontsize=10)
    fig.suptitle("同一早期预算：FlashSAC完整任务 / 固定任务 / 原STR", y=.99, fontsize=17)
    fig.text(.5, .045, "每1M transitions内等权平均TB日志窗口；不是按全体episode重新加权。淡线与阴影展示全部3个简单任务seed，不是置信区间。",
             ha="center", fontsize=9)
    fig.text(.5, .02, "原SAPG统计block5，FlashSAC统计全部env；DR/任务/控制/窗口不同，不能公平排名。简单版seed1最终确定性评测仅1/64。",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.13, hspace=.39, wspace=.29)
    fig.savefig(ROOT / "local_reward_components.png", dpi=180)
    plt.close(fig)


def remote_data():
    path = ROOT / "remote_sources/plot_sources.json"
    if not path.exists():
        return []
    result = []
    for source in json.loads(path.read_text()):
        item = dict(source)
        if "curves_file" in source or "curves" in source:
            payload = json.loads(Path(source["curves_file"]).read_text()) if "curves_file" in source else source
            item["curves"] = payload["curves"]
            item["source_metadata"] = payload.get("metadata", {})
            if "early_curves_file" in source:
                item["early_curves"] = json.loads(Path(source["early_curves_file"]).read_text())["curves"]
            result.append(item)
            continue
        item["curves"] = {}
        cached = {directory: load_events(Path(directory), {key: mapping["tag"] for key, mapping in source["metrics"].items()})
                  for directory in source["event_dirs"]}
        for key, mapping in source["metrics"].items():
            # Each metric explicitly declares its x-axis conversion and units.
            combined = {}
            for directory in source["event_dirs"]:
                curve = cached[directory][key]
                for step, value in curve:
                    combined[step * mapping["step_multiplier"]] = value
            item["curves"][key] = sorted(combined.items())
        result.append(item)
    return result


def draw_remote(sources):
    if not sources:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7))
    colors = ["#6652A0", "#D98C2F", "#398B61"]
    for ax, (metric, (_, title, _)) in zip(axes.flat, METRICS.items()):
        units = {source["metrics"][metric]["unit"] for source in sources if source["curves"].get(metric)}
        if len(units) > 1:
            raise ValueError(f"Do not put incompatible remote units on one axis: {metric}: {units}")
        plotted = False
        for source, color in zip(sources, colors):
            points = source["curves"].get(metric, [])
            if not points:
                continue
            width = source.get("bin_transitions", 100_000_000)
            curve = ({i: (p[0], p[1], 1) for i, p in enumerate(points)} if source.get("already_binned")
                     else bin_curve(points, width))
            ax.plot([v[0] / 1e9 for v in curve.values()], [v[1] for v in curve.values()],
                    color=color, lw=1.8, label=source["label"])
            plotted = True
        ax.set(title=title, xlabel="原记录累计环境 transitions（十亿，B）", ylabel=next(iter(units), "未记录"))
        ax.grid(alpha=.19)
        if not plotted:
            ax.text(.5, .5, "源TB未记录可核对的该项\n不填0、不推算", transform=ax.transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=1, frameon=False, fontsize=10)
    fig.suptitle("原STR / SAPG 历史训练参考：单独尺度、单独统计口径", y=.99, fontsize=17)
    fig.text(.5, .035, "每100M transitions内均值；横轴为全部env累计样本，纵轴为SAPG block5的episode统计。续训段未左移；机器人与DR不同。",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.80, bottom=.13, hspace=.39, wspace=.29)
    fig.savefig(ROOT / "original_str_reference.png", dpi=180)
    plt.close(fig)


def self_test():
    curve = bin_curve([(1, 1), (2, 3), (12, 9), (21, 100)], 10, 20)
    assert curve == {0: (1.5, 2.0, 2), 1: (12.0, 9.0, 1)}
    assert bin_curve([], 10) == {}
    print("Binning checks passed")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    style()
    local = local_data()
    remote = remote_data()
    (ROOT / "curves.json").write_text(json.dumps({"local": local, "remote": remote}, ensure_ascii=False, allow_nan=False) + "\n")
    draw_local(local, remote)
    draw_remote(remote)
    print(f"Saved local_reward_components.png; remote references: {len(remote)}; raw verified scalars: curves.json")


if __name__ == "__main__":
    main()
