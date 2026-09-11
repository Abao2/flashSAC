#!/usr/bin/env python3
"""Compare pinned upstream and adapted FlashSAC in isolated processes.

No simulator, rollout, production edits or checkpoint writes. Normal n=3 rows
must match upstream; shorter timeout targets have separate oracle tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path('/home/abao/Documents/Codex/flashsac-official-play/FlashSAC')
PIN = '87edc9061150ae9e962dd84e6544e27a1554b3ab'
RUN = ROOT / 'diagnostics/credit_nstep3_20260910'
CHECKPOINT = RUN / 'candidate/models/seed0/step48830'
NAMES = ('actor', 'critic', 'target_critic', 'temperature')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix='', result=None):
    import torch
    result = {} if result is None else result
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, prefix + '/' + str(key), result)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            flatten(item, prefix + '/' + str(i), result)
    elif isinstance(value, torch.Tensor):
        result[prefix] = value.detach().cpu().clone()
    else:
        result[prefix] = value
    return result


def prepare_fixture(path, batch_size):
    import torch
    import yaml
    cfg = yaml.safe_load((RUN / 'expected_resolved_config.yaml').read_text())['agent']
    data = torch.load(CHECKPOINT / 'replay_buffer.pt', weights_only=True,
                      map_location='cpu', mmap=True)
    assert cfg['n_step'] == 3 and cfg.get('actor_history_length', 1) == 1
    assert data['observation'].shape[1] == 302 and data['action'].shape[1] == 29
    generator = torch.Generator().manual_seed(271828)
    candidates = torch.randint(data['num_in_buffer'], (batch_size * 8,), generator=generator)
    full = (data['discount'][candidates] - cfg['gamma'] ** cfg['n_step']).abs() < 1e-7
    indices = candidates[full][:batch_size * 2]
    assert len(indices) == batch_size * 2
    batches = [{k: v[idx].clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
               for idx in indices.split(batch_size)]
    assert any((b['terminated'] == 0).any() for b in batches)
    assert not torch.equal(batches[0]['observation'][:, :140], batches[0]['observation'][:, 140:280])
    model = {name: torch.load(CHECKPOINT / f'{name}.pt', weights_only=True, map_location='cpu')
             for name in NAMES}
    normalizer = torch.load(CHECKPOINT / 'reward_normalizer.pt', weights_only=True, map_location='cpu')
    agent_state = torch.load(CHECKPOINT / 'agent_state.pt', weights_only=True, map_location='cpu')
    fixture = dict(config=cfg, batches=batches, indices=indices, model=model,
                   normalizer=normalizer, agent_state=agent_state)
    torch.save(fixture, path)
    return dict(batch_size=batch_size, source=str(CHECKPOINT),
                index_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                full_n3_rows=len(indices), terminated_rows=int(sum(b['terminated'].sum() for b in batches)),
                truncated_rows=int(sum(b['truncated'].sum() for b in batches)),
                actor_critic_prefix_equal=bool(torch.equal(batches[0]['observation'][:, :140],
                                                         batches[0]['observation'][:, 140:280])))


def worker(args):
    # Import only the selected repository. Each worker is a separate interpreter.
    sys.path.insert(0, str(args.repo))
    import contextlib
    from dataclasses import fields
    import gymnasium as gym
    import numpy as np
    import torch
    from flash_rl.agents.flashSAC import agent as am
    from flash_rl.agents.flashSAC import update as um
    from flash_rl.agents.flashSAC import network as architecture
    from flash_rl.agents.flashSAC import layer as layers
    from flash_rl.agents.utils import network as nm
    from flash_rl.agents.utils import reward_normalization as rm
    from flash_rl.agents.utils import scheduler as sm

    torch.set_num_threads(4)
    train_numeric = '_trainflags' in args.mode
    deterministic = not train_numeric or args.mode.endswith('_deterministic')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = train_numeric
    torch.backends.cudnn.allow_tf32 = train_numeric
    torch.backends.cudnn.benchmark = train_numeric
    torch.set_float32_matmul_precision('high' if train_numeric else 'highest')
    fixture = torch.load(args.fixture, weights_only=True, map_location='cpu')
    device = 'cuda' if 'cuda' in args.mode else 'cpu'
    compiled = 'compiled' in args.mode
    amp = 'amp' in args.mode
    stance = contextlib.nullcontext() if compiled else torch.compiler.set_stance('force_eager')
    args.output.mkdir(parents=True, exist_ok=True)
    with stance:
        values = {k: v for k, v in fixture['config'].items() if k in {f.name for f in fields(am.FlashSACConfig)}}
        values.update(device_type=device, buffer_device_type='cpu', buffer_max_length=8,
                      buffer_min_length=1, sample_batch_size=args.batch_size,
                      use_compile=compiled, use_amp=amp)
        torch.manual_seed(161803)
        cfg = am.FlashSACConfig(**values)
        native_layout = args.case == 'native_fresh'
        width = 162 if native_layout or args.side == 'official' else 302
        info = {'actor_observation_size': (140,)}
        if args.side == 'adapted':
            info.update(critic_observation_size=(162,), critic_observation_offset=0 if native_layout else 140)
        agent = am.FlashSACAgent(gym.spaces.Box(-np.inf, np.inf, shape=(1, width), dtype=np.float32),
                                 gym.spaces.Box(-1., 1., shape=(1, 29), dtype=np.float32), info, cfg)
        if args.case == 'str_trained':
            for name in NAMES:
                net = getattr(agent, '_' + name)
                saved = fixture['model'][name]
                module = getattr(net.network, '_orig_mod', net.network)
                module.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in saved['network_state_dict'].items()})
                if net.optimizer is not None:
                    net.optimizer.load_state_dict(saved['optimizer_state_dict'])
                    # CPU proof uses CPU Adam; production mode preserves fused CUDA Adam.
                    for group in net.optimizer.param_groups:
                        group['fused'] = device == 'cuda'
                    net.scheduler.load_state_dict(saved['scheduler_state_dict'])
                net.update_step = saved['update_step']
            agent._update_step = fixture['agent_state']['update_step']
            if amp:
                agent._grad_scaler.load_state_dict(fixture['agent_state']['grad_scaler_state_dict'])
        norm = agent.reward_normalizer
        for key in ('G_r', 'G_r_max'):
            setattr(norm, key, fixture['normalizer'][key].to(device).clone())
        for key in ('mean', 'var', 'count'):
            setattr(norm.G_rms, key, fixture['normalizer']['G_rms_' + key].to(device).clone())

        trace, counters = {}, {}

        def record(key, obj):
            flatten(obj, key, trace)

        def forward_hook(name):
            def hook(module, positional, keywords, output):
                i = counters.get(name, 0)
                counters[name] = i + 1
                record(f'forward/{name}/{i}/inputs', keywords)
                record(f'forward/{name}/{i}/outputs', output)
            return hook

        # Hooks observe existing calls only; never add a BN/RNG-changing forward.
        # Production compiled mode is intentionally unhooked to preserve its graphs.
        if not compiled:
            for name in NAMES:
                getattr(agent, '_' + name).network.register_forward_hook(forward_hook(name), with_kwargs=True)
        def optimizer_hook(name):
            def hook(optimizer, positional, keywords):
                module = getattr(getattr(agent, '_' + name).network, '_orig_mod', getattr(agent, '_' + name).network)
                record('step_gradients/' + name, {k: p.grad for k, p in module.named_parameters()})
            return hook

        for name in ('actor', 'critic', 'temperature'):
            getattr(agent, '_' + name).optimizer.register_step_post_hook(optimizer_hook(name))
        original_target = um._compute_categorical_td_target

        def target_spy(**kwargs):
            output = original_target(**kwargs)
            if not compiled:
                record('td_target/distribution', output)
                record('td_target/interior_mass_mean', output[:, 1:-1].sum(1).mean())
                record('td_target/discount', torch.as_tensor(kwargs['gamma'], device=device).expand(args.batch_size))
            return output

        if not compiled:
            um._compute_categorical_td_target = target_spy
        original_update = am._update_networks

        def update_spy(**kwargs):
            record('prepared_batch', {k: v for k, v in kwargs['batch'].items() if k != 'discount'})
            record('actor_update_due', kwargs['do_actor_update'])
            result = original_update(**kwargs)
            record('losses_and_metrics', result)
            return result

        am._update_networks = update_spy

        def snapshot():
            for name in NAMES:
                net = getattr(agent, '_' + name)
                module = getattr(net.network, '_orig_mod', net.network)
                record('model/' + name, module.state_dict())
                if not compiled:
                    record('gradients/' + name, {k: p.grad for k, p in module.named_parameters()})
                if net.optimizer is not None:
                    record('optimizer/' + name, net.optimizer.state_dict())
                    record('scheduler/' + name, net.scheduler.state_dict())
            record('normalizer', dict(G_r=norm.G_r, G_r_max=norm.G_r_max,
                                     mean=norm.G_rms.mean, var=norm.G_rms.var, count=norm.G_rms.count))
            record('agent_step', agent._update_step)
            record('grad_scaler', agent._grad_scaler.state_dict())
            record('rng/cpu', torch.get_rng_state())
            if device == 'cuda':
                record('rng/cuda', torch.cuda.get_rng_state())

        # Match RNG after construction, checkpoint loading and hook registration.
        torch.manual_seed(141421)
        snapshot()
        torch.save(trace, args.output / 'initial.pt')
        for step in range(args.steps):
            trace.clear()
            counters.clear()
            raw = {k: v.clone() for k, v in fixture['batches'][step % 2].items()}
            if native_layout:
                for key in ('observation', 'next_observation'):
                    # Same prefix convention supported by the unmodified native Agent.
                    raw[key] = torch.cat((raw[key][:, :140], raw[key][:, -22:]), dim=1)
            if args.side == 'official' and not native_layout:
                # Official core accepts independent actor/critic tensors. Its native
                # Agent does not support STR's non-prefix transport layout.
                prepared = {k: v.to(device) for k, v in raw.items() if k not in ('observation', 'next_observation', 'discount')}
                for key in ('observation', 'next_observation'):
                    transport = raw[key].to(device)
                    prepared['actor_' + key] = transport[:, :140]
                    prepared[key] = transport[:, 140:302]
                prepared['reward'] = norm.normalize_rewards(prepared['reward'])
                am._update_networks(batch=prepared, actor=agent._actor, critic=agent._critic,
                                    target_critic=agent._target_critic, temperature=agent._temperature,
                                    cfg=agent._cfg, do_actor_update=agent._update_step % agent._cfg.actor_update_period == 0,
                                    device=agent._device, grad_scaler=agent._grad_scaler)
                agent._update_step += 1
            else:
                # Fix sample indices/data only; execute the complete actual Agent.update.
                agent._replay_buffer.sample = lambda: raw
                agent.update()
            snapshot()
            torch.save(trace, args.output / f'step{step:02}.pt')
            print(f'{args.side} {args.case} {args.mode} update {step+1}/{args.steps}', flush=True)
        sources = {name: {'path': module.__file__, 'sha256': digest(module.__file__)}
                   for name, module in [('agent', am), ('update', um), ('network_wrapper', nm),
                                        ('reward_normalizer', rm), ('scheduler', sm), ('architecture', architecture), ('layers', layers)]}
        for entry in sources.values():
            assert Path(entry['path']).is_relative_to(args.repo), entry
        (args.output / 'provenance.json').write_text(json.dumps(dict(sources=sources, torch=torch.__version__,
            device=device, compiled=compiled, amp=amp, steps=args.steps, case=args.case,
            forward_hooks_enabled=not compiled, optimizer_gradient_hooks=True,
            deterministic_algorithms=deterministic, tf32=train_numeric, cudnn_benchmark=train_numeric), indent=2))


def compare(left, right):
    import torch
    assert left.keys() == right.keys(), (left.keys() - right.keys(), right.keys() - left.keys())
    failures, groups = [], {}
    for key, a in left.items():
        b = right[key]
        group = key.split('/')[0]
        stat = groups.setdefault(group, {'fields': 0, 'max_abs_diff': 0., 'unequal_fields': 0})
        stat['fields'] += 1
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                failures.append({'field': key, 'reason': 'shape/dtype mismatch'})
                continue
            finite = not (a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()))
            diff = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
            equal = torch.equal(a, b) and finite
            stat['max_abs_diff'] = max(stat['max_abs_diff'], diff)
        else:
            equal, diff = a == b, None
        if not equal:
            stat['unequal_fields'] += 1
            failures.append({'field': key, 'max_abs_diff': diff})
    return {'pass_exact': not failures, 'groups': groups, 'first_differences': failures[:20],
            'num_different_fields': len(failures)}


def main(args):
    import time
    import torch
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff', 'HEAD', '--', 'flash_rl', 'configs/agent/flashSAC.yaml'], text=True)
    before = {p.name: digest(p) for p in CHECKPOINT.glob('*.pt') if p.name != 'replay_buffer.pt'}
    report = dict(upstream_commit=PIN, upstream_root=str(UPSTREAM), adapted_root=str(ROOT),
                  checkpoint_before=before, cases=[], strict_bitwise=True, script_sha256=digest(__file__),
                  self_check=args.self_check,
                  scope='Full-n3 learner parity, NOT physics/environment parity or training convergence.')
    started = time.monotonic()
    # Temporary tensor traces are removed after comparison; JSON evidence persists.
    with tempfile.TemporaryDirectory(prefix='flashsac-update-parity-') as directory:
        temp = Path(directory)
        fixture = temp / 'fixture.pt'
        report['fixture'] = prepare_fixture(fixture, args.batch_size)
        for case in args.cases:
            for mode in args.modes:
                result = dict(case=case, mode=mode, snapshots=[])
                for side, repo in [('official', UPSTREAM), ('adapted', ROOT)]:
                    worker_side = args.self_check or side
                    if args.self_check:
                        repo = UPSTREAM if args.self_check == 'official' else ROOT
                    output = temp / case / mode / side
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--side', worker_side,
                               '--repo', str(repo), '--fixture', str(fixture), '--output', str(output),
                               '--case', case, '--mode', mode, '--steps', str(args.steps), '--batch-size', str(args.batch_size)]
                    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0')
                    log = args.output / f'{case}_{mode}_{side}.log'
                    with log.open('w') as stream:
                        proc = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    if proc.returncode:
                        raise RuntimeError(f'{side} {case} {mode} failed: {log}')
                    result[side] = json.loads((output / 'provenance.json').read_text())
                for filename in ['initial.pt'] + [f'step{i:02}.pt' for i in range(args.steps)]:
                    base = temp / case / mode
                    a = torch.load(base / 'official' / filename, weights_only=True, map_location='cpu')
                    b = torch.load(base / 'adapted' / filename, weights_only=True, map_location='cpu')
                    entry = compare(a, b)
                    entry['snapshot'] = filename
                    if 'prepared_batch/reward' in b:
                        entry['normalized_reward_range'] = [float(b['prepared_batch/reward'].min()), float(b['prepared_batch/reward'].max())]
                    if 'td_target/interior_mass_mean' in b:
                        entry['target_interior_mass_mean'] = float(b['td_target/interior_mass_mean'])
                    if filename == 'initial.pt':
                        initial = b
                    else:
                        entry['parameter_changes'] = {}
                        for name in NAMES:
                            prefix = 'model/' + name + '/'
                            keys = [k for k in b if k.startswith(prefix) and isinstance(b[k], torch.Tensor)
                                    and b[k].is_floating_point() and not any(x in k for x in ('running_mean', 'running_var'))]
                            entry['parameter_changes'][name] = max(float((b[k] - initial[k]).abs().max()) for k in keys)
                    result['snapshots'].append(entry)
                # Negative control: the comparator must reject a perturbed weight.
                key = next(k for k, v in b.items() if k.startswith('model/actor') and isinstance(v, torch.Tensor) and v.is_floating_point())
                b[key] = b[key].clone() + .01
                result['comparator_negative_control_detected'] = not compare(a, b)['pass_exact']
                result['nonzero_parameter_update'] = all(v > 0 for v in result['snapshots'][-1]['parameter_changes'].values())
                result['pass_exact'] = (all(s['pass_exact'] for s in result['snapshots'])
                                        and result['comparator_negative_control_detected'] and result['nonzero_parameter_update'])
                report['cases'].append(result)
                print(f"{case} {mode}: {'PASS exact' if result['pass_exact'] else 'DIFFERENCES'}", flush=True)
                (args.output / 'result.json').write_text(json.dumps(report, indent=2))
    report['checkpoint_after'] = {p.name: digest(p) for p in CHECKPOINT.glob('*.pt') if p.name != 'replay_buffer.pt'}
    report['checkpoint_unchanged'] = report['checkpoint_after'] == before
    report['elapsed_seconds'] = time.monotonic() - started
    report['pass_exact'] = all(c['pass_exact'] for c in report['cases']) and report['checkpoint_unchanged']
    (args.output / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('pass_exact', 'checkpoint_unchanged', 'elapsed_seconds')}, indent=2))
    if not report['pass_exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--cases', nargs='+', default=['native_fresh', 'str_fresh', 'str_trained'])
    parser.add_argument('--modes', nargs='+', default=['cpu_fp32'])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--self-check', choices=['official', 'adapted'])
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--side', choices=['official', 'adapted'])
    parser.add_argument('--case', choices=['native_fresh', 'str_fresh', 'str_trained'])
    parser.add_argument('--mode', choices=['cpu_fp32', 'cuda_fp32', 'cuda_amp', 'cuda_compiled_amp', 'cuda_compiled_amp_trainflags', 'cuda_compiled_amp_trainflags_deterministic'])
    main(parser.parse_args())
