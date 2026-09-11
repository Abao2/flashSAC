# xug6 原 STR / SAPG 曲线来源

从本机 abao 直接只读 SSH xug6 提取；没有经 Mac，没有修改远程文件、启动训练或评测。未下载958MB事件文件，只流式读取标量。

## 来源与覆盖

- 入口：`/home/lixiyuan/simtoolreal/tb/s2r_seed0_full/merged`
- 解析目标：`builds/build-20260824-163012-1787560212769022941`
- 事件文件：`events.out.tfevents.1787560212.ubuntu-MS03-CE0-000.62630.0`，958,118,699 bytes。
- 每个提取tag均169,001个点，step0→66,453,504,000；逐tag与`verify_report.json`计数、首尾step一致，且严格递增。
- 历史包含xug6从scratch到12.897B、两段xug52续跑、最后xug6续跑。此次没有连接xug52；来源按原合并manifest追溯。

## 机器人和 DR 的直接证据

保存的原始配置：`/home/lixiyuan/simtoolreal/outputs/2026-08-15/23-25-21/.hydra/config.yaml`。

- line110：`assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`。
- line89：24,576环境；line256–257：arm/hand moving average都是0.1。
- line318/320/322：obs delay、action delay、object state delay/noise均true；xyz noise0.01m、rotation noise5°、joint velocity noise0.1。
- line392：`load_checkpoint: false`；line457–461：`mixed_expl_learn_param`、exploration scale0.005、block size4,096。

这确认曲线对应原 KUKA+Sharpa、带DR的原算法训练，不是Wuji，也不是此次FlashSAC DR-off实验。当前读取到的保存配置是初始run；后续合并段的每次源配置没有在本次逐段重审。

## 指标口径（必须保留在图中）

远端`rl_games/rl_games/common/a2c_common.py:355`起设置`ignore_env_boundary=max(...,num_actors−block_size)=20480`；line921–926只用后4096环境更新reward并传给observer。`isaacgymenvs/utils/rlgames_utils.py:157`起先裁剪infos，然后累计episode指标。

因此原SAPG return与episode分项是**block5子组**，横轴仍为全部24,576环境的global transitions，不能称全环境均值。`rewards/step`是用step做横轴的episode return，不是单步即时reward。`episode_final/successes`是每episode完成目标数，**不是成功率**；`successes`另保留为tracker，不混成同一指标。step0是首个训练日志点，不是独立零步评测。

## 本地数据格式

- `scalars_early_raw_100m.json`：每tag保留255个原始日志点，0→99,876,864；`points_kind=raw_logged_points_step_le100M`。
- `scalars_binned100m.json`：每tag665个固定100M区间的日志算术均值。横坐标为该bin内实际日志steps的均值；`bin_counts`保存各bin点数；`first_raw`/`last_raw`另保存真实首尾点。不是episode加权均值，也不冒充原始点。
- 两者`curves`都是canonical key→`[[step,value],...]`；key为return/fingertip/lifting/lift_bonus/keypoint/goals/successes_tracker。`metadata.source_tags`保留真实TB tag。

接近100M的最后原始值：return69.0935、fingertip49.9601、lifting8.58725、lift_bonus41.4、keypoint1.46570、goals0.00033333。最终66.45B原始值：return8152.2773、goals8.343；不是与FlashSAC100M等预算的对照。
