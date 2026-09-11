# FlashSAC完整STR：接着100M候选继续训练

实际恢复已核验（北京时间14:44）：TB累计101,027,840，actor/critic正在更新，tol=.032285、无非有限scalar。resume_loaded.json确认源update195120、10M replay和原RMS已加载。tb/nstep3_full已包含两份原始event，横轴204800→101027840连续保留；证据见RESUME_VERIFIED.json。这只是重启后的早期窗口，不据此判断收敛。

用户指出尚未收敛不应停训；本轮继续现有n_step=3候选，不重新初始化模型，不重跑已结束对照。本机执行，不改变远程任务、真机或其他hw进程。

## 当前阶段

从../credit_nstep3_20260910/candidate/models/seed0/step48830加载，追加400,015,360 transitions：累计100,003,840→500,019,200。预计约2.5–3小时，以真实吞吐为准。500M是下次检查点，不是预先宣布的收敛点；自动续查根据目标完成、失败、容差和数值趋势检查是否继续，不将预算结束当作收敛。

保持arm EMA=1、hand EMA=.1、n_step=3、gamma=.99、2048env、完整6类/1200资产随机任务、原reward/reset/成功标准/课程、FlashSAC网络和现有优化器调度。DR仍关闭；KUKA+Sharpa，不是Wuji。

## 恢复与边界

恢复全部模型、Adam、LR scheduler、GradScaler、reward RMS和10M replay；起始native update195120，学习率.00015。replay先CPU暂存再写入已有GPU空间，避免双份GPU replay；主机需约25.5GB额外可用内存。

运行时恢复_current_success_tolerance=.03228504075；配置上限仍.075、下限仍.01，原课程规则不变，不使用eval固定容差。旧课程计时/各环境历史successes未保存，本次重新等待3000个并行步后再检查收紧。

由于物理环境重新reset，只清reward normalizer中每个环境未完成episode的G_r，不清RMS/G_r_max；否则会把旧episode累积接到新episode上。环境/RNG、noise repeat状态和未完n-step窗口未恢复；原生首步会采一次随机动作。此为保留学习状态的续训，不是逐步精确重放。

TB日志统一加100003840步偏移，原始旧events保留；不手改reward值。续训原始文件在candidate/runs；合并单条曲线入口在tb/nstep3_full，链接旧新event文件。中间checkpoint相对续训阶段命名：step48830=累计200M、step97660=300M、step146490=400M、step195320=500M。每20M存模型，每100M存replay；所有路径独立，不覆盖100M源模型。

## 运行文件

- 配置：../../configs/simtoolreal_full_nstep3_continue.yaml。
- 完整命令/来源：manifest.json；队列状态：queue_status.json；日志：train.log。
- 入口仍为../../scripts/train_str_holding.py，只补恢复元数据、课程数值和TB偏移，不复制FlashSAC算法。
- 实际加载证据：candidate/resume_loaded.json；final checkpoint及replay验证后才写training_complete.json。
- 到阶段末再检查是否继续，不宣称达到.01便已收敛；可靠性需要同协议测试。20秒play中的45goal不是平均，也不是50-goal全链完成。

本机看完整历史：

```bash
/home/abao/play2perfect/.venv_isaacsim/bin/tensorboard --logdir=/home/abao/flashsac-robotics/diagnostics/credit_nstep3_continue_20260910/tb --host=127.0.0.1 --port=6008
```

没有启动新的TB服务器；若6008已占用，用现有服务器或另选端口。检查完整历史链接与event实际写入后再使用。
