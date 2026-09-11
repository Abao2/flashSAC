"""Synthetic GPU learner smoke; not simulation, evaluation, or learning evidence."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from flash_rl.agents import create_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    rng = np.random.default_rng(20260909)
    OmegaConf.register_new_resolver('eval', lambda expression: eval(expression), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(REPO / 'configs')):
        cfg = hydra.compose(config_name='simtoolreal_full_lstm32')
    cfg.agent.buffer_max_length = 100000  # Exercise wrap without allocating the10M trial buffer.
    n, actor_dim, critic_dim, action_dim = 2048, 140, 162, 29
    obs_space = gym.spaces.Box(-10., 10., (n, actor_dim + critic_dim), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (n, action_dim), dtype=np.float32)
    info = dict(actor_observation_size=(actor_dim,), critic_observation_size=(critic_dim,),
                critic_observation_offset=actor_dim)
    print('Creating full-size history actor/critic on GPU', flush=True)
    agent = create_agent(obs_space, action_space, info, cfg.agent)
    actor = agent._actor.network
    before = actor.lstm.weight_ih_l0.detach().clone()
    observation = rng.normal(0, .3, obs_space.shape).astype(np.float32)
    for step in range(60):
        # Native startup skips sample_actions while replay is below minimum.
        if agent.can_start_training():
            actions = agent.sample_actions(step, {'next_observation': observation}, training=True)
            assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
        else:
            actions = rng.uniform(-1, 1, action_space.shape).astype(np.float32)
        final = rng.normal(0, .3, obs_space.shape).astype(np.float32)
        term, trunc = np.zeros(n, bool), np.zeros(n, bool)
        if step in (20, 45):
            term[:12], trunc[12:24] = True, True
        transition = dict(observation=observation, action=actions,
                          reward=rng.normal(0, .5, n).astype(np.float32),
                          terminated=term, truncated=trunc, next_observation=final)
        agent.process_transition(transition)
        observation = final.copy()
        observation[term | trunc] = 0.123  # Distinct auto-reset state, not final state.
    assert len(agent._replay_buffer) == 100000 and agent._replay_buffer._total_written == 60 * n
    assert agent._critic_network_observation_dim == 4674
    expected_count = 60 * n
    assert float(agent.reward_normalizer.G_rms.count) == expected_count
    print('Replay wrap, random warmup, online history populated; compiling critic updates', flush=True)
    torch.cuda.reset_peak_memory_stats()
    measured_start = None
    losses = []
    for update in range(args.updates):
        if update == 10:
            torch.cuda.synchronize()
            measured_start = time.monotonic()
        metrics = agent.update()
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        losses.append(metrics)
        if update in (0, 9, args.updates - 1):
            print(f'Finite native update {update + 1}/{args.updates}: {metrics}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - measured_start if measured_start is not None else None
    assert not torch.equal(before, actor.lstm.weight_ih_l0)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    assert float(agent.reward_normalizer.G_rms.count) == expected_count  # Not multiplied by H or optimizer calls.
    report = dict(status='PASS', kind='synthetic_learner_smoke_not_eval',
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  num_envs=n, history_length=32, actor_observation_dim=actor_dim,
                  critic_current_observation_dim=critic_dim, critic_network_observation_dim=4674,
                  actor_parameter_count=sum(p.numel() for p in actor.parameters()),
                  critic_parameter_count=sum(p.numel() for p in agent._critic.network.parameters()),
                  amp=agent._cfg.use_amp, critic_compile=agent._cfg.use_compile,
                  actor_compile=False, native_updates=args.updates,
                  lstm_parameters_changed=True, losses_finite=True,
                  normalizer_real_transition_count=expected_count,
                  replay_wrap_tested=True, timeout_finalobs_cpu_regression_required=True,
                  measured_updates=max(args.updates - 10, 0), measured_seconds=elapsed,
                  updates_per_second=(args.updates - 10) / elapsed if elapsed else None,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  final_metrics=losses[-1], config=asdict(agent._cfg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != 'config'}), flush=True)


if __name__ == '__main__':
    main()
