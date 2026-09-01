"""Gym registration for the Dex4D Isaac Lab port."""

import gymnasium as gym


gym.register(
    id="FlashSAC-Dex4D-XArm6Leap-Direct-v0",
    entry_point="robotics_tasks.dex4d.dex4d_env:Dex4DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotics_tasks.dex4d.dex4d_env_cfg:Dex4DEnvCfg",
    },
)

gym.register(
    id="FlashSAC-Dex4D-M6Wuji-Direct-v0",
    entry_point="robotics_tasks.dex4d.dex4d_env:Dex4DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotics_tasks.dex4d.dex4d_env_cfg:Dex4DM6WujiEnvCfg",
    },
)

gym.register(
    id="FlashSAC-Dex4D-M6Wuji-Stage12-Direct-v0",
    entry_point="robotics_tasks.dex4d.dex4d_env:Dex4DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotics_tasks.dex4d.dex4d_env_cfg:Dex4DM6WujiStage12EnvCfg",
    },
)

gym.register(
    id="FlashSAC-Dex4D-XArm6Leap-Stage12-Direct-v0",
    entry_point="robotics_tasks.dex4d.dex4d_env:Dex4DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robotics_tasks.dex4d.dex4d_env_cfg:Dex4DStage12EnvCfg",
    },
)
