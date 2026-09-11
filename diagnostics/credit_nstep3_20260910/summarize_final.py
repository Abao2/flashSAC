"""CPU-only final comparison; reuse existing event reader/binning, never train."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'reward_comparison_20260909'))
from plot_reward_comparison import load_events, one_run, bin_curve, style, plt

TAGS = dict(return_='episode/return', goals='episode/final/successes',
            tolerance='task/current_success_tolerance',
            keypoint='episode/cumulative/keypoint_rew',
            lift_bonus='episode/cumulative/lift_bonus_rew',
            fall='episode/final/done_fall', hand_far='episode/final/done_hand_far',
            height_hold='episode/final/height_proxy_above_10cm_1s')
SPECS = [
    ('n1', '基准：hand=.1，n=1', ROOT.parent / 'control_ab_20260909_arm1/runs',
     ROOT.parent / 'control_ab_20260909_arm1/play_20260909_v4', '#64748b', '--'),
    ('hand1', '手平滑对照：hand=1，n=1', ROOT.parent / 'hand_response_20260910/candidate/runs',
     ROOT.parent / 'hand_response_20260910/play', '#bd6548', ':'),
    ('n3', '三步回报：hand=.1，n=3', ROOT / 'candidate/runs', ROOT / 'play', '#167d8d', '-'),
]


def main():
    report = {'budget': 100003840, 'window': [90003840, 100003840], 'groups': {},
              'notes': ['Window means are unweighted TB logged-window means, not episode-weighted rates.',
                        'Play counts describe up to20s per first episode; censored episodes stay in initial64 denominator.',
                        'Goal-positive is not50-goal success; height proxy is not contact/grasp proof.',
                        'Same initial task/curriculum, but n3 advances curriculum; final play freezes tolerance=.075.']}
    for key, label, parent, play, color, ls in SPECS:
        directory = one_run(parent)
        curves = load_events(directory, TAGS)
        assert curves['return_'][-1][0] == report['budget']
        row = report['groups'][key] = dict(label=label, event_directory=str(directory),
                                         curves=curves, last_10m={}, play={})
        for metric, points in curves.items():
            values = [v for s, v in points if report['window'][0] < s <= report['window'][1]]
            row['last_10m'][metric] = dict(mean=sum(values)/len(values), windows=len(values),
                                         last=points[-1][1]) if values else None
        for mode in ('deterministic', 'stochastic'):
            episodes = json.loads((play / f'{mode}_episodes.json').read_text())
            assert len(episodes) == 64 and len({e['env_id'] for e in episodes}) == 64
            row['play'][mode] = dict(
                initial_episodes=len(episodes), completed=sum(e['complete'] for e in episodes),
                goal_positive=sum(e['goals'] >= 1 for e in episodes), total_goals=sum(e['goals'] for e in episodes),
                full50=sum(e['complete'] and e['goals'] >= 50 for e in episodes),
                height_hold_1s=sum(e['longest_above_10cm_s'] >= 1 for e in episodes),
                failure_union=sum(e['termination_reasons']['done_fall'] or e['termination_reasons']['done_hand_far'] for e in episodes),
                timeouts=sum(e['termination_reasons']['done_timeout'] for e in episodes),
                censored=[{'env_id':e['env_id'], 'type':e['asset_type'], 'goals':e['goals'], 'steps':e['steps']}
                          for e in episodes if not e['complete']])
    (ROOT / 'final_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    for ax, metric, title, unit in zip(axes, ['return_', 'goals', 'tolerance'],
                                     ['环境回报', '真正到达的目标数', '课程容差（越低越严格）'],
                                     ['分 / episode', 'goals / episode', 'tolerance (m)']):
        for key, label, _, _, color, ls in SPECS:
            bins = bin_curve(report['groups'][key]['curves'][metric], 1000000)
            ax.plot([v[0]/1e6 for v in bins.values()], [v[1] for v in bins.values()],
                    color=color, ls=ls, lw=2, label=label)
        ax.set(title=title, xlabel='环境 transitions（百万）', ylabel=unit, xlim=(0, 100.01))
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
    fig.suptitle('完整 STR + FlashSAC：同 seed、同 100M 预算', y=.99, fontsize=16)
    fig.text(.5, .035, '每1M内等权平均TB日志窗口；单seed，无置信区间。arm=1、DR关闭；物体/目标/网络/reward不改。', ha='center', fontsize=10)
    fig.subplots_adjust(left=.07, right=.985, top=.74, bottom=.19, wspace=.32)
    fig.savefig(ROOT / 'training_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: {'last_10m': v['last_10m'], 'play': v['play']} for k,v in report['groups'].items()}, indent=2))


if __name__ == '__main__':
    main()
