"""CPU-only frozen STR actor-gradient probe; not an optimizer/training update."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path('/home/abao/flashsac-robotics')
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from diagnose_str_critic import checkpoint_models, summary
from diagnose_str_rollouts import model_digest

CHECKPOINT = REPO / 'diagnostics/credit_nstep3_continue_20260910/candidate/models/seed0/step195320'
TRACE = REPO / 'diagnostics/str_fourway_audit_20260911/q_hardreset_baseline/reference_trace.npz'
PHASES = (0, 60, 120)
SEED = 20260911
REPEATS = 8


def tensor_digest(value):
    digest = hashlib.sha256()
    for key, item in sorted(value.items()):
        digest.update(key.encode())
        if isinstance(item, torch.Tensor):
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(item).encode())
    return digest.hexdigest()


def flatten_grad(parts, params, select):
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for i, (p, g) in enumerate(zip(params, parts)) if select(i)])


def gradient_relationship(qg, hg):
    qn, hn, total = qg.norm(), hg.norm(), qg + hg
    tn = total.norm()
    return dict(q_l2=float(qn), entropy_l2=float(hn), total_l2=float(tn),
                entropy_over_q=float(hn / qn.clamp_min(1e-30)),
                q_entropy_cosine=float((qg @ hg) / (qn * hn).clamp_min(1e-30)),
                q_total_cosine=float((qg @ total) / (qn * tn).clamp_min(1e-30)),
                cancellation_ratio=float(tn / (qn + hn).clamp_min(1e-30)),
                local_q_descent_gain_fraction=float((qg @ total) / qn.square().clamp_min(1e-30)))


def action_stats(action, q_gradient):
    jacobian = 1 - action.detach().square()
    dq_dz = q_gradient * jacobian
    qn, zn = q_gradient.norm(dim=-1), dq_dz.norm(dim=-1)
    return dict(abs_action=summary(action.detach().abs().numpy()),
                fraction_abs_action_ge_095=float((action.detach().abs() >= .95).float().mean()),
                fraction_abs_action_ge_098=float((action.detach().abs() >= .98).float().mean()),
                fraction_tanh_jacobian_lt_001=float((jacobian < .01).float().mean()),
                tanh_jacobian=summary(jacobian.numpy()),
                dq_da_l2=summary(qn.numpy()), dq_dz_l2=summary(zn.numpy()),
                tanh_gradient_norm_retention=summary((zn / qn.clamp_min(1e-30)).numpy()),
                arm_dq_da_l2=summary(q_gradient[:, :7].norm(dim=-1).numpy()),
                hand_dq_da_l2=summary(q_gradient[:, 7:].norm(dim=-1).numpy()))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    actor, critic, target, alpha, normalizer = checkpoint_models(CHECKPOINT, 'cpu')
    actor.requires_grad_(False)
    actor.eval(); critic.eval(); target.eval()
    source_files = [CHECKPOINT / (n + '.pt') for n in
                    ('actor', 'critic', 'target_critic', 'temperature', 'reward_normalizer')]
    file_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    before = dict(actor=model_digest(actor), critic=model_digest(critic), target=model_digest(target),
                  reward_normalizer=tensor_digest(normalizer), alpha=alpha)
    with np.load(TRACE) as data:
        all_obs = np.array(data['obs'])
    assert all_obs.shape == (121, 32, 302)
    observations = torch.from_numpy(np.concatenate([all_obs[t] for t in PHASES]))
    ao, co = observations[:, :140], observations[:, 140:]
    assert actor.embedder.w.w.weight.shape[1] == 140
    assert critic.embedder.w.weight.shape[-1] == 162 + 29
    with torch.no_grad():
        mean, std = actor.get_mean_and_std(ao, training=False)
    action = mean.tanh().requires_grad_(True)
    q, _ = critic(co, action, training=False)
    qmin = q.min(0).values
    dq = torch.autograd.grad(qmin.sum(), action)[0].detach()
    inference = {}
    for index, t in enumerate(PHASES):
        sl = slice(index * 32, (index + 1) * 32)
        inference[str(t)] = dict(action_stats(action[sl], dq[sl]),
                                min_q=summary(qmin[sl].detach().numpy()),
                                pre_tanh_std=summary(std[sl].numpy()))

    # The checkpoint replay batch is unavailable here. Match the real 4096-row
    # actor BN size with a fixed *local trajectory* context; never invent s123.
    # Rows 0:96 are the three query phases; remaining rows only supply BN context.
    rng = np.random.default_rng(SEED)
    bank = all_obs.reshape(-1, 302)
    context_indices = rng.choice(len(bank), 4096 - len(ao), replace=True)
    actor_batch = torch.cat([ao, torch.from_numpy(bank[context_indices, :140])])
    phase_rows = {str(t): [] for t in PHASES}
    for repeat in range(REPEATS):
        torch.manual_seed(SEED + repeat)
        probe = copy.deepcopy(actor).requires_grad_(True)
        actions_all, info = probe(actor_batch, training=True)
        actions, logp = actions_all[:len(ao)], info['log_prob'][:len(ao)]
        qs, _ = critic(co, actions, training=False)
        qmin = qs.min(0).values
        da = torch.autograd.grad(qmin.sum(), actions, retain_graph=True)[0].detach()
        named = tuple(probe.named_parameters())
        params = tuple(p for _, p in named)
        selectors = {
            'all': lambda i: True,
            'mean_head': lambda i: named[i][0].startswith(('predictor.mean_w', 'predictor.mean_bias')),
            'std_head': lambda i: named[i][0].startswith(('predictor.std_w', 'predictor.std_bias')),
            'backbone': lambda i: not named[i][0].startswith('predictor.'),
        }
        for index, t in enumerate(PHASES):
            sl = slice(index * 32, (index + 1) * 32)
            q_loss, entropy_loss = -qmin[sl].mean(), alpha * logp[sl].mean()
            qparts = torch.autograd.grad(q_loss, params, retain_graph=True, allow_unused=True)
            hparts = torch.autograd.grad(entropy_loss, params, retain_graph=True, allow_unused=True)
            groups = {name: gradient_relationship(flatten_grad(qparts, params, select),
                                                 flatten_grad(hparts, params, select))
                      for name, select in selectors.items()}
            phase_rows[str(t)].append(dict(repeat=repeat, seed=SEED + repeat,
                q_loss=float(q_loss.detach()), entropy_loss=float(entropy_loss.detach()),
                sampled_entropy=float(-logp[sl].mean().detach()), groups=groups,
                sampled_action_stats=action_stats(actions[sl], da[sl])))
        del probe

    train_summary = {}
    for t, rows in phase_rows.items():
        train_summary[t] = {group: {key: summary([row['groups'][group][key] for row in rows])
                                   for key in rows[0]['groups'][group]}
                            for group in rows[0]['groups']}
    after = dict(actor=model_digest(actor), critic=model_digest(critic), target=model_digest(target),
                 reward_normalizer=tensor_digest(normalizer), alpha=alpha)
    assert before == after
    assert all(file_hashes[p.name] == hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files)
    assert not torch.cuda.is_initialized()
    result = dict(status='complete', checkpoint=str(CHECKPOINT), trace=str(TRACE),
        seed=SEED, repeats=REPEATS, device='cpu', dtype='float32', cuda_initialized=False,
        learning=False, optimizer_step=False, alpha=alpha, bc_alpha=0,
        actual_actor_loss='mean(alpha * log_pi - minimum(Q1,Q2)); no BC',
        actual_bn_modes='Actor training=True on concatenated observation context; critic training=False.',
        selected_steps=list(PHASES), selected_envs=list(range(32)),
        context=dict(rows=4096, queried_rows=96, remaining_rows=4000,
                     origin='Seeded resampling from all 121x32 reference states; local diagnostic context, NOT replay current/next pairs.',
                     reference_obs_sha256=hashlib.sha256(all_obs.tobytes()).hexdigest(),
                     sampled_context_index_sha256=hashlib.sha256(context_indices.tobytes()).hexdigest()),
        source_file_sha256=file_hashes, original_state_before=before, original_state_after=after,
        deterministic_inference=inference, actor_training_mode_summary=train_summary,
        actor_training_mode_repeats=phase_rows,
        limitations=[
            'The 32 environments and 0/60/120 probes are local reference trajectory states, not the true training replay distribution.',
            's120 has no recorded successor. No successor was fabricated; actor BN context is explicit and only approximate.',
            'Actor BN updates happen ONLY in disposable clones; source model, critic, target, alpha and reward-normalizer states remain unchanged.',
            'Parameter gradients are raw float32, before Adam preconditioning, AMP, learning rate and weight renormalization.',
            'Tanh attenuation describes a local chain derivative, not proof that learning is blocked or actions are physically unsafe.',
            'Direction refers to learned Q locally, not measured environment returns.'],
    )
    output = Path(__file__).with_name('result.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    compact = {t: {k: v['mean'] for k, v in values['all'].items()} for t, values in train_summary.items()}
    print(json.dumps(dict(output=str(output), alpha=alpha, actor_training_mode_all_params=compact,
                          source_states_unchanged=True, cuda_initialized=False), indent=2))


if __name__ == '__main__':
    main()
