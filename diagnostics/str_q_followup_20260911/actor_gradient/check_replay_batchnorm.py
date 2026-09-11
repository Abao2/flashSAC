"""CPU-only actual-replay actor batch-BN versus execution running-BN comparison."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import resource

import torch

from check_actor_gradient import CHECKPOINT, SEED, checkpoint_models, model_digest, summary, tensor_digest


def file_metadata(path):
    value = path.stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino)


def compare(mean_train, std_train, mean_eval, std_eval, critic_obs, critic):
    batch_action, running_action = mean_train.tanh(), mean_eval.tanh()
    q_batch, _ = critic(critic_obs, batch_action, training=False)
    q_running, _ = critic(critic_obs, running_action, training=False)
    q_batch = torch.minimum(q_batch[0], q_batch[1])
    q_running = torch.minimum(q_running[0], q_running[1])
    diff = batch_action - running_action
    return dict(
        action_abs_difference=summary(diff.abs().numpy()),
        per_state_action_difference_l2=summary(diff.norm(dim=-1).numpy()),
        arm_action_abs_difference=summary(diff[:, :7].abs().numpy()),
        hand_action_abs_difference=summary(diff[:, 7:].abs().numpy()),
        action_components_difference_gt_01=float((diff.abs() > .1).float().mean()),
        action_components_difference_gt_05=float((diff.abs() > .5).float().mean()),
        batch_bn_fraction_abs_action_ge_098=float((batch_action.abs() >= .98).float().mean()),
        running_bn_fraction_abs_action_ge_098=float((running_action.abs() >= .98).float().mean()),
        batch_bn_action_abs=summary(batch_action.abs().numpy()),
        running_bn_action_abs=summary(running_action.abs().numpy()),
        batch_bn_pretanh_std=summary(std_train.numpy()),
        running_bn_pretanh_std=summary(std_eval.numpy()),
        frozen_q_batch_bn_mean_action=summary(q_batch.numpy()),
        frozen_q_running_bn_mean_action=summary(q_running.numpy()),
        frozen_q_difference_batch_minus_running=summary((q_batch - q_running).numpy()),
        fraction_batch_bn_mean_action_has_higher_q=float((q_batch > q_running).float().mean()),
    )


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    assert not torch.cuda.is_initialized()
    replay_path = CHECKPOINT / 'replay_buffer.pt'
    replay_before = file_metadata(replay_path)
    actor, critic, target, alpha, normalizer = checkpoint_models(CHECKPOINT, 'cpu')
    actor.requires_grad_(False)
    actor.eval(); critic.eval(); target.eval()
    original_before = dict(actor=model_digest(actor), critic=model_digest(critic),
                           target=model_digest(target), normalizer=tensor_digest(normalizer), alpha=alpha)
    checkpoint_files = [CHECKPOINT / f'{name}.pt' for name in
                        ('actor', 'critic', 'target_critic', 'temperature', 'reward_normalizer')]
    file_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files}
    # Same mmap API as scripts/check_str_replay_coverage.py. Access ONLY selected
    # row indices; never materialize, clone, scan, or hash the whole 24GB replay.
    replay = torch.load(replay_path, weights_only=True, mmap=True, map_location='cpu')
    n = int(replay['num_in_buffer'])
    ring_idx = int(replay['current_idx'])
    assert n == 10_000_000 and replay['observation'].shape[1] == 302
    rows = []
    with torch.no_grad():
        for sample_id in range(3):
            seed = SEED + sample_id
            generator = torch.Generator(device='cpu').manual_seed(seed)
            ids = torch.randint(n, (2048,), generator=generator)
            current = replay['observation'][ids].clone()
            next_obs = replay['next_observation'][ids].clone()
            assert current.shape == next_obs.shape == (2048, 302)
            combined = torch.cat([current, next_obs], dim=0)
            actor_input = combined[:, :140]
            critic_input = combined[:, 140:]
            probe = copy.deepcopy(actor)
            mean_batch, std_batch = probe.get_mean_and_std(actor_input, training=True)
            mean_running, std_running = actor.get_mean_and_std(actor_input, training=False)
            row = dict(seed=seed, replay_index_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                       selected_row_sha256=hashlib.sha256(combined.numpy().tobytes()).hexdigest(),
                       selected_unique_rows=int(ids.unique().numel()))
            for name, sl in [('current', slice(0, 2048)), ('next_auxiliary', slice(2048, 4096))]:
                row[name] = compare(mean_batch[sl], std_batch[sl], mean_running[sl], std_running[sl],
                                    critic_input[sl], critic)
            rows.append(row)
            del probe, current, next_obs, combined, actor_input, critic_input
    del replay
    original_after = dict(actor=model_digest(actor), critic=model_digest(critic),
                          target=model_digest(target), normalizer=tensor_digest(normalizer), alpha=alpha)
    assert original_before == original_after
    assert replay_before == file_metadata(replay_path)
    assert file_hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint_files}
    assert not torch.cuda.is_initialized()
    result = dict(status='complete', device='cpu', dtype='float32', cuda_initialized=False,
        checkpoint=str(CHECKPOINT), replay=str(replay_path), replay_rows=n, ring_current_idx=ring_idx,
        replay_load='torch.load(weights_only=True, mmap=True, map_location=cpu); indexed rows only',
        replay_metadata_before=replay_before, replay_metadata_after=file_metadata(replay_path),
        batches=3, current_rows_per_batch=2048, actual_actor_bn_rows_per_batch=4096,
        source_file_sha256=file_hashes, original_state_before=original_before, original_state_after=original_after,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        method='Same actual replay current+next observations, actor training=True in disposable clone versus original training=False; tanh(mean), no action sampling; same frozen critic training=False.',
        sampled_batches=rows,
        limitations=[
            'Three sampled batches from the final saved 10M-row replay, not all historical training batches or seeds.',
            'Next observations are real stored n-step next/final observations; no next state was fabricated.',
            'Q evaluates tanh(mean) actions, not the full expected stochastic SAC objective or actual physical return.',
            'No optimizer step, running-stat adaptation experiment, AMP or weight normalization step.',
            'A mode-dependent action difference is measurable here but does not by itself establish a training failure cause.'],
    )
    out = Path(__file__).with_name('replay_batchnorm_result.json')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    compact = [dict(seed=r['seed'], mean_action_abs_difference=r['current']['action_abs_difference']['mean'],
                    q_difference=r['current']['frozen_q_difference_batch_minus_running']['mean'],
                    batch_saturation=r['current']['batch_bn_fraction_abs_action_ge_098'],
                    running_saturation=r['current']['running_bn_fraction_abs_action_ge_098']) for r in rows]
    print(json.dumps(dict(output=str(out), peak_rss_mib=result['peak_rss_kib']/1024,
                          current_batches=compact, source_states_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
