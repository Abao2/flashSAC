# STR 单目标仿真状态诊断（2026-09-08）

目的：验证 FlashSAC 能否从桌上抓起一个 STR eraser 并搬到固定空中 pose。
这是放宽观测、关闭随机化的学习诊断，不是论文复现、泛化测试或真机策略。

## 不变的部分

- 当前 STR 原机器人：KUKA IIWA + Sharpa，29 维原动作、原控制器计算。
- STR 原 reward 公式、原累计 10 步近目标成功计数、已修复的 timeout/final_obs 接口。
- FlashSAC actor 128 宽/2 残差块、双分布式 critic 256 宽/2 块/101档、BN/RMSNorm/权重约束。
- 1024 env、batch 2048、每1024条transition更新2次、n_step=1、GPU replay上限10M。

## 本轮明确改变

- Actor 从 140 维原观测改读已有 162 维仿真状态，Critic仍读162维。
  wrapper的324维只是两段输入拼接；Q网络实际输入162+29=191维。
- 固定一个 eraser、固定起点和单个 goal；成功一次即结束，不再追50个目标。
- 关闭观测/动作延迟、物体噪声、速度噪声、随机外力/外矩；无摩擦和尺寸随机化。
- arm/hand moving average=1.0（仿真快训设置，不是原真机滤波设置）。
- tolerance固定为.02；实际最大关键点距离阈值=.02×1.5=.03m。
- 上限10,000,384 transitions，每976次交互（999,424 transitions）保存checkpoint。
  10M是第一轮诊断上限，不是收敛保证；学习率日程随本轮预算缩短。

## 起点和目标

都是环境局部的物体root pose，四元数顺序wxyz。

- 固定资产采样seed42的eraser collision尺寸：
  0.14605714451279328 × 0.05659969709057025 × 0.049932924209851834 m。
- 桌面z=.38+.30/2=.53m。
- 物体初始z=.53+半高+.001=.555966462104926m，底部距桌面1mm。
- 起点：`[0, 0, .555966462104926, 1, 0, 0, 0]`。
- 目标：`[.05, 0, .705966462104926, 1, 0, 0, 0]`。
- 初始目标误差约15.81cm；目标要求横移5cm、上移15cm，保持朝向。
- 原lifted判据要求超过初始高度10cm；目标3cm容差要求至少约12cm抬升。
  因而不能只在桌上推动就达标。动态抓取可行性仍需训练/play验证。

## 文件和命令

完整配置：`configs/simtoolreal_state_teacher.yaml`。
启动入口（不要在已有训练运行时重复启动）：

```bash
cd /home/abao/flashsac-robotics
./scripts/run_str_state_teacher.sh
```

本机Python：`/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python`。
GPU预检脚本：`scripts/check_str_state_teacher.py`，应与训练错开运行。

本次后台服务：`flashsac-str-state-teacher-20260908.service`。

```bash
systemctl --user status flashsac-str-state-teacher-20260908.service
tail -n 20 /home/abao/flashsac-robotics/diagnostics/state_teacher_20260908/train.log
```

TensorBoard（绝对路径无需conda激活）：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/tensorboard --logdir=/home/abao/flashsac-robotics/runs/simtoolreal_diagnostics --host=127.0.0.1 --port=6010 --load_fast=false
```

主要看 `episode/final/all_goals_hit`（本轮单目标episode成功率）、
`episode/cumulative/lift_bonus_rew`、`episode/return`，以及各done原因。
训练时是带探索的成功率；独立确定性checkpoint评测另算，不能混为泛化成功率。
`all_goals_hit`在旧50目标任务与本轮单目标任务含义不同，不能直接对比。

TensorBoard目录：`runs/simtoolreal_diagnostics/state162_eraser_onegoal_seed0/`。
模型目录：`models/simtoolreal_diagnostics/state162_eraser_onegoal_seed0/`。
预检/训练日志：`diagnostics/state_teacher_20260908/`。

## 已验证

- adapter及n-step/bootstrap单元测试：15 passed。
- 16环境、700步真实仿真预检：162/162/29形状正确，干净Actor/Critic输入一致，
  目标固定，无NaN/Inf；16次自然timeout，final_obs确为reset前状态。
- 这只证明预检通过，学习效果以实际曲线和独立rollout为准。

独立评测（先等训练结束，避免同时占用GPU）：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python /home/abao/flashsac-robotics/scripts/eval_str_state_teacher.py --checkpoint /home/abao/flashsac-robotics/models/simtoolreal_diagnostics/state162_eraser_onegoal_seed0/Isaacsimenvs-SimToolReal-Direct-v0/seed0-0908-153538/step9760 --episodes 100 --num-envs 16
```

输出每回合 `EVAL_EPISODE` 和总计 `EVAL_RESULT`。最终误差取reset前状态，
不是下一回合起点；位置误差单位m、角度误差单位rad。这里是固定起点的重复评测，
不是新物体/新目标泛化率。`lifted`指曾抬高超过10cm；原success累计近目标10步，
不要求连续，不能单凭它宣称稳定持握。162维也仍不包含所有内部计数器，
例如累计近目标步数，因此不能宣称它是严格完整的Markov状态。

本轮训练循环完成10,000,384 transitions（9766次交互）。trainer按周期保存，
最后保存的是step9760，即9,994,240 transitions；最后6次交互没有补存。
上述评测使用这份实际存在的最后checkpoint，而不是虚构step9766。

## 本轮结果：尚未学会搬运

2026-09-08，本机1024环境完成10,000,384 transitions，训练循环约5分55秒
（不含约半分钟仿真启动及后续评测）；后台训练正常退出，没有自动追加预算。
Actor/online critic/target critic的中途checkpoint参数检查均有限值。

最后保存的9,994,240-transition checkpoint，固定起点确定性评测100回合：

| 指标 | 结果 |
| --- | --- |
| 单目标成功 | 0/100 |
| 曾抬高超过10cm | 0/100 |
| 最终平均位置误差 | 0.15901052m，即15.90cm |
| 最终平均角度误差 | 0.02377983rad，约1.36度 |
| 最终平均最大关键点误差 | 0.15918050m（成功阈值0.03m） |
| 原STR episode return均值 | 607.08465 |
| 回合长度 | 全部600步，以超时结束 |

目标误差没有明显改善。这里goal朝向本就与起点相同，所以低角度误差不是完成重定向；
drop指标虽然为0，但因为没有达到lift阈值，不能解释为抓握稳定。
原lifting_rew在物体未抬起时也有正的高度项，reward不为零不能说明抓取成功。

结论范围：这轮短预算、单seed、固定任务的仿真状态诊断没有训起来；
不能据此证明FlashSAC无法解决STR，也不能把失败直接归因于缺少LSTM。
原始逐回合数据与汇总见`diagnostics/state_teacher_20260908/eval_10m.log`。
