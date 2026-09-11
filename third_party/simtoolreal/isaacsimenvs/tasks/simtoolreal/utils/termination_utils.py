"""Termination and success-tolerance curriculum helpers."""

from __future__ import annotations

import torch


def update_tolerance_curriculum(env) -> None:
    """Shrink success tolerance when completed episodes average enough goals."""
    env._frame_counter += 1
    term = env.cfg.termination
    if env._frame_counter - env._last_curriculum_update >= term.tolerance_curriculum_interval:
        successes = env._prev_episode_successes.float()
        eligible_mask = None
        if hasattr(env, "_curriculum_eligible_mask"):
            eligible_mask = env._curriculum_eligible_mask()
        if eligible_mask is not None:
            successes = successes[eligible_mask]

        threshold = term.tolerance_curriculum_success_threshold
        if hasattr(env, "_curriculum_success_threshold"):
            custom_threshold = env._curriculum_success_threshold()
            if custom_threshold is not None:
                threshold = float(custom_threshold)

        if successes.numel() > 0 and successes.mean().item() >= threshold:
            new_tol = env._current_success_tolerance * term.tolerance_curriculum_increment
            new_tol = max(min(new_tol, term.success_tolerance), term.target_success_tolerance)
            env._current_success_tolerance = new_tol
            env._last_curriculum_update = env._frame_counter

    # Eval pins the success criterion.
    if term.eval_success_tolerance is not None:
        env._current_success_tolerance = float(term.eval_success_tolerance)


def compute_terminations(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Update goal-hit state and return ``(terminated, truncated)``."""
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    is_success = env._is_success

    # Authoritative updates on goal-hit.
    env._successes = env._successes + is_success.long()
    env._pending_goal_reset.copy_(is_success)

    # Termination causes.
    object_z_local = env.object.data.root_pos_w[:, 2] - env_origins[:, 2]
    fall = object_z_local < 0.1

    if term_cfg.max_consecutive_successes > 0:
        max_successes_reached = env._successes >= term_cfg.max_consecutive_successes
    else:
        max_successes_reached = torch.zeros_like(fall)

    hand_far = env._curr_fingertip_distances.max(dim=-1).values > 1.5

    # Exact legacy resetWhenDropped rule (env.py:_compute_resets): after the
    # object has ever crossed the lift threshold, reset if it falls below the
    # initial object height for this episode.  _lifted_object is latched, so
    # its value from the preceding reward hook is authoritative here.
    if term_cfg.reset_when_dropped:
        dropped = (object_z_local < env._object_init_z) & env._lifted_object
    else:
        dropped = torch.zeros_like(fall)

    terminated = fall | max_successes_reached | hand_far | dropped
    # A goal hit starts a fresh per-goal horizon after reward calculation, so
    # it must not simultaneously become a time-limit truncation.
    truncated = (
        # Legacy Isaac Gym checks progress >= max_episode_length - 1.
        (env.episode_length_buf >= env.max_episode_length - 1)
        & ~is_success
        & ~terminated
    )
    env._termination_reasons = {
        "fall": fall,
        "max_successes": max_successes_reached,
        "hand_far": hand_far,
        "dropped": dropped,
        "timeout": truncated,
    }
    return terminated, truncated


__all__ = ["update_tolerance_curriculum", "compute_terminations"]
