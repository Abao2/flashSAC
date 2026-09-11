"""Static CPU scientific figure from completed frozen checkpoint evaluations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
report = json.loads((root / 'summary.json').read_text())
assert report['queue_status'] == 'completed'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 13, 'axes.labelsize': 11})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.3), sharey=True)
colors = ['#2373A8', '#D48724', '#39876D']
for axis, sampling, title in zip(axes, ['deterministic', 'stochastic'],
                                ['Deterministic policy · 64 episodes', 'Native stochastic policy · 128 episodes']):
    for seed in range(3):
        counts, percentages = [], []
        for step in ['48830', '58596']:
            item = report['seeds'][str(seed)]['checkpoints'][step]['evaluations'][sampling]
            assert item['status'] == 'complete' and item['complete_budget'] and item['model_unchanged']
            n, total = item['counts']['success'], item['episodes']
            counts.append(f'{n}/{total}')
            percentages.append(100 * n / total)
        positions = np.arange(2) + (seed - 1) * .25
        axis.bar(positions, percentages, width=.22, color=colors[seed], label=f'Training seed {seed}', zorder=3)
        for x, y, count in zip(positions, percentages, counts):
            axis.text(x, y + 2, count, ha='center', va='bottom', fontsize=10)
    axis.set_title(title, pad=13)
    axis.set_xticks([0, 1], ['50M', '60M'])
    axis.set_xlabel('Training budget (transitions)')
    axis.set_ylim(0, 112)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(axis='y', color='#D8DDE3', linewidth=.7, zorder=0)
axes[0].set_ylabel('Goal success (%)')
fig.suptitle('Native FlashSAC · fixed SimToolReal task', fontsize=17, x=.075, ha='left', y=.97)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .905), ncol=3, frameon=False)
fig.text(.075, .045, 'Fresh continuous training; no BC or checkpoint/replay reload. Frozen evaluations use common rollout seed 0.', fontsize=10)
fig.text(.075, .014, 'Fixed eraser / start / goal, DR off. Bars are completed-episode proportions; no independent-IID confidence intervals.', fontsize=10)
fig.subplots_adjust(left=.075, right=.985, top=.77, bottom=.18, wspace=.18)
fig.savefig(root / 'reproducibility.png', dpi=170, facecolor='white')
plt.close(fig)
print(root / 'reproducibility.png')
