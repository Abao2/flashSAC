# FlashSAC × STR：恢复原训练任务分布，关闭 DR

状态：2026-09-09 10:25 本机 seed0 已完成场景/agent初始化并进入真实网络更新；seed1/2及独立评测串行排队。10:26读取TensorBoard已到2,560,000 transitions，25个scalar标签均有限，actor/critic/temperature loss已实际写出。此时完成目标均值仍为0、容差0.075：仅确认正常开训，不宣称已学会。用户授权“恢复，去试试”。

GPU预检通过：2048环境×700步、实际覆盖1200个资产及全部6类；actor140/critic162/拼接302/动作29；初始位置/姿态和目标位置/姿态均2048个不同值；1256次真终止、1711次时间截断，pre-reset final_obs检查通过。预检无网络更新，不是训练成功率。CPU回归共29项通过。

启动后进程PID2753454，实测GPU占用30,211MiB（本机是48GB版4090，不代表24GB卡可用同配置）；20个启动源码hash复核一致，原有代码和昨晚模型均未覆盖。

服务：`flashsac-str-full-nodr-20260909.service`；队列状态看本目录`queue_status.json`，实际命令看`manifest.json`；预检证据`preflight_gpu.json`，启动源码快照`source_hashes.json`。原先夜间自动续查仍暂停，本轮没有新建定时通知或操作远端。

## 这次恢复什么

- 原 KUKA+Sharpa，140 维 actor；162 维 critic 状态（adapter 拼接输入302维），29维动作。
- 原任务6类工具、12个尺寸子分布，每个子分布100个模型，共1200个（hammer200/screwdriver200/marker100/spatula200/eraser100/brush400）；随机物体/关节初态、随机首目标、后续随机位置和旋转增量目标；最多50个连续目标。
- 原 reward、手臂/手指平滑系数0.1/0.1、600步每目标超时，以及0.075→0.01容差课程。没有固定起点/终点/轨迹覆盖。
- 仅关闭当前 Lab 实现里的 DR、观测噪声、外力/扭矩和延迟；物体形状分布和初态随机性保留。
- 原生前馈 FlashSAC 从零训练；不加载昨晚162维 actor，不做BC，不改核心训练算法。这不是PPO/SAPG，也不是论文完整预算/24任务或真机复现。

## 首轮预算与判据

3个训练seed顺序跑，每个100,003,840 transitions（2048环境×48,830步）；总计300,011,520 transitions。每9,766步保存一次，约20/40/60/80/100M。2048环境覆盖全1200资产池，同时每批交互更新4次，保持与昨晚1024环境/2更新相同的每transition更新比例。其余沿用昨晚配方：replay10M、预填100k、batch2048、Actor每2次Critic更新一次、原网络/熵参数、固定LR衰减19,532更新。预算为第一轮尝试，不是收敛时间承诺。

配置：`/home/abao/flashsac-robotics/configs/simtoolreal_full_nodr.yaml`。原始任务：`/home/abao/simtoolreal/isaacsimenvs/cfg/task/SimToolReal.yaml`。除了DR，任务设置由注册入口继承，不复制一份容易漂移的奖励/重置配置。

所有实验在本机进行；GPU阶段串行，显存不足等待，不终止其他任务。独立输出，不覆盖历史结果，不做Git提交推送，不操作真机或远端训练。

## 查看曲线

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/full_nodr_20260909_1004/runs --host=127.0.0.1 --port=6008
```

浏览器：`http://127.0.0.1:6008`。若端口占用，换空闲端口。

每个独立进程都需重新构建完整USD场景；本次预检场景初始化约6分钟，不是学习停滞。训练日志在本目录`logs/seed0_train100m.log`；TensorBoard事件在场景/agent初始化后才出现。

优先看每回合完成目标数、抬升、return、当前容差；`all_goals_hit` 表示完成整条50目标链，不是单个goal命中率。独立确定性评测会固定容差分别报告初始0.075和目标0.01，不能混成同一成功率。物理关键点阈值还需乘reward.keypoint_scale=1.5。

资产路径会先shuffle再按索引分配；32个环境可能包含6类，但不能覆盖全1200模型池。正式评测使用1200环境、每个资产一个回合，并根据实际路径/索引记录分类型和全池覆盖。未完成/缺失评测不记作0。
