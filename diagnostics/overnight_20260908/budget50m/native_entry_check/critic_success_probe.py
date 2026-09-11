"""CPU-only paired checkpoint queries on two fixed512-transition rollout batches."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path('/home/abao/flashsac-robotics')
ROOT = REPO / 'diagnostics/overnight_20260908/budget50m/native_entry_check'
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from diagnose_str_critic import checkpoint_models, reward_denominator, diagnose, actor_gradient_metrics, summary
from diagnose_str_rollouts import model_digest
from summarize_str_control_ab import sha256


def main():
    output = ROOT / 'critic_success_comparison.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    checkpoints = {'old_failed50m': ROOT.parent / 'models/arm1_hand01/seed0/step48830',
                   'new_successful50m_plus5m': ROOT / 'training/normal/models/final/step4883'}
    datasets = {'failed_rollout512': ROOT.parent / 'evaluations/arm1_hand01_seed0/step48830/stochastic/transitions.npz',
                'successful_episode_rollout512': ROOT / 'evaluations/normal/relativefinal/stochastic/transitions.npz'}
    report = {'device': 'cpu', 'selection_seed': 0, 'random_actions_per_state': 32,
              'target_batch_size': 128, 'global_gradient_samples': 512, 'gradient_repeats': 4,
              'entropy_mc_repeats': 8, 'gamma': .99, 'checkpoint_hashes': {}, 'batches': {},
              'limits': ['Both checkpoints query identical recorded rows/actions within each batch; batches remain separate.',
                        'Successful batch is conditioned on successful episodes from the new native stochastic rollout; not all128 episodes or a replay-frequency estimate.',
                        'Uniform512 transition sampling does not preserve every rare bonus/terminal event; selected counts and per-phase summaries are explicit.',
                        'Neither dataset is historical training replay. Different checkpoint next-policies produce different diagnostic TD targets.',
                        'TD targets use pre-reset final_obs, current frozen alpha/normalizer, current next-policy, and target training cross-BN on fixed mixed current/next128-row batches.',
                        'Actor/Q sensitivity uses inference BN; raw actor parameter gradients use training actor cross-BN on all512 current+next pairs, Critic inference BN.',
                        'Gradients exclude Adam preconditioning, AMP, unit projection and actual updates. They do not establish true Q accuracy or causal mechanism.',
                        'Terminal failures/successes do not bootstrap; timeouts bootstrap final_obs. Q is soft normalized value, not raw episodic reward.',
                        'CPU FP32 uncompiled queries only. All checkpoint tensors and files remain unchanged.']}
    for label, path in checkpoints.items():
        report['checkpoint_hashes'][label] = {f.name: sha256(f) for f in sorted(path.glob('*.pt'))}
    for name, path in datasets.items():
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        assert metadata['next_obs_is_pre_reset'] is True
        with np.load(path, allow_pickle=False) as saved:
            eligible = np.flatnonzero(saved['success']) if name.startswith('successful') else np.arange(len(saved['obs']))
            ids = np.random.default_rng(0).choice(eligible, 512, replace=False)
            keys = ['obs', 'next_obs', 'actions', 'rewards', 'terminated', 'truncated', 'episode_id', 'step_in_episode', 'phase']
            data = {k: saved[k][ids].copy() for k in keys}
        batch = {'source': str(path), 'sha256': sha256(path), 'metadata_sha256': sha256(path.parent / 'metadata.json'),
                 'eligible_rows': len(eligible), 'selected_ids': ids.tolist(), 'phase_counts': dict(zip(*[x.tolist() for x in np.unique(data['phase'], return_counts=True)])),
                 'terminated': int(data['terminated'].sum()), 'truncated': int(data['truncated'].sum()),
                 'raw_reward_gt200': int((data['rewards'] > 200).sum()), 'models': {}}
        actions = {}
        for label, checkpoint in checkpoints.items():
            actor, critic, target, alpha, norm = checkpoint_models(checkpoint, 'cpu')
            before = [model_digest(m) for m in (actor, critic, target)]
            denominator = reward_denominator(norm, 5.)
            obs, nxt = [torch.as_tensor(data[k], dtype=torch.float32) for k in ('obs', 'next_obs')]
            torch.manual_seed(42)
            phase_report = diagnose(actor, critic, target, alpha, denominator, data,
                                    random_actions=32, batch_size=128, gradient_repeats=4)
            torch.manual_seed(142)
            global_gradient = actor_gradient_metrics(actor, critic, obs, nxt, np.full(512, 'all'), alpha, 4)['all']
            with torch.no_grad():
                mu, std = actor.get_mean_and_std(obs, training=False)
                entropy = []
                for repeat in range(8):
                    torch.manual_seed(1000 + repeat)
                    _, info = actor(obs, training=False)
                    entropy.append(-info['log_prob'].numpy())
            weighted = {k: sum(v['sampled_transitions'] * v['metrics'][k]['mean'] for v in phase_report.values()) / 512
                        for k in next(iter(phase_report.values()))['metrics']}
            actions[label] = {'mu': mu.numpy(), 'std': std.numpy(), 'action': mu.tanh().numpy()}
            assert [model_digest(m) for m in (actor, critic, target)] == before
            batch['models'][label] = {'checkpoint': str(checkpoint), 'alpha': alpha, 'reward_denominator': denominator,
                'global_raw_actor_gradient': global_gradient, 'global_metrics_weighted_by_selected_rows': weighted,
                'tanh_entropy_nats_8mc': summary(np.stack(entropy)),
                'pretanh_mu': summary(mu.numpy()), 'pretanh_std': summary(std.numpy()),
                'deterministic_action': summary(mu.tanh().numpy()), 'phases': phase_report, 'models_unchanged': True}
        old, new = actions.values()
        batch['new_minus_old_policy'] = {field: {'rmse': float(np.sqrt(np.mean((new[field] - old[field]) ** 2))),
            'mean_abs_delta': float(np.mean(np.abs(new[field] - old[field]))),
            'arm_rmse': float(np.sqrt(np.mean((new[field][:, :7] - old[field][:, :7]) ** 2))),
            'hand_rmse': float(np.sqrt(np.mean((new[field][:, 7:] - old[field][:, 7:]) ** 2)))} for field in old}
        report['batches'][name] = batch
    for label, path in checkpoints.items():
        assert report['checkpoint_hashes'][label] == {f.name: sha256(f) for f in sorted(path.glob('*.pt'))}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved unchanged-checkpoint CPU comparison: {output}')


if __name__ == '__main__':
    main()
