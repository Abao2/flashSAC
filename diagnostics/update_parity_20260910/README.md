# FlashSAC 学习器迁移：逐次更新验证

日期：2026-09-10。

## 结论

**已验证：在受控的确定性计算设置下，适配版和固定版本的官方 FlashSAC，在本次真实 STR 数据上的学习更新逐元素一致。**

这包括 CPU FP32、GPU FP32、GPU AMP，以及编译 + AMP + TF32。每组连续执行 6 次更新，不在更新之间重新加载网络，覆盖 Actor 更新和不更新的两个分支。

**没有宣称：默认非确定性 GPU 设置能逐位一致、完整环境接口已全部证明、或长期训练必然收敛。** 默认训练数值开关下，严格逐位比较未通过；官方代码自身重复也出现了同量级差异。没有通过放宽阈值把它改成 PASS。

本次没有修改训练算法、超参数、正式 checkpoint 或 replay，没有启动仿真训练。只新增验证脚本、独立 oracle 测试及本目录证据。

## “相同的一次更新”比较什么

两边各自加载相同网络权重、Adam 状态、BatchNorm 状态、目标网络、温度、学习率调度器、奖励归一化状态和随机数状态；拿到同一批经验，实际执行各自的更新代码。

比较：

1. Actor/Critic 输入切片、归一化后的 reward、是否轮到 Actor 更新。
2. 现有前向调用的动作、log probability、Q 输出和 categorical TD target。
3. Actor/Critic/温度 loss、反向梯度。
4. 更新后的参数、BN buffers、Adam 动量、学习率、目标网络、GradScaler 和 RNG 状态。

顺序遵循官方实际代码：轮到 Actor 时，先 Actor，再温度，再使用更新后的 Actor/温度训练 Critic，最后目标网络 EMA；另一个分支只执行 Critic 和目标网络更新。

编译模式不加网络前向 hooks，以免改变被编译的图；该模式验证输入、loss、优化器更新时捕获的梯度及完整更新后状态，**不声称已逐个检查编译图内部的 TD target**。非编译模式额外记录这些中间量。没有为了记录指标额外运行一次网络前向。

## 数据和独立性

- 官方源码：`/home/abao/Documents/Codex/flashsac-official-play/FlashSAC`。
- 固定官方 commit：`87edc9061150ae9e962dd84e6544e27a1554b3ab`；检查相关 tracked 源码未被修改。
- 适配源码：`/home/abao/flashsac-robotics`；两版通过独立 Python 进程导入，各自记录实际源码路径及 SHA256，避免误把同一模块导入两次。
- 真实数据来自 `diagnostics/credit_nstep3_20260910/candidate/models/seed0/step48830`。
- 从保存的 10M replay 中固定抽取两批各 2048 条，共 4096 条有效长度为 3 的经验；包含 3 条终止样本、3 条时间截断样本。短于 3 步的边界另用独立 oracle 检查。
- Actor 输入 140 维、Critic 输入 162 维、动作 29 维。断言 Actor 内容并不等于 Critic 的前 140 维，不能靠重复输入误过测试。
- Replay 通过 mmap 读取并 clone 采样张量，没有整体搬到 GPU。保存的模型和优化器文件执行前后 SHA256 一致。没有对整个约 25 GB replay 文件做全文件哈希；脚本没有写回它。

## 三种测试情形

| 情形 | 官方侧 | 适配侧 |
|---|---|---|
| `native_fresh` | 原生支持的 Actor-prefix 布局，完整 `Agent.update()` | 同一布局，完整 `Agent.update()` |
| `str_fresh` | 新初始化网络；手工独立切分 STR 两份语义输入，调用未修改的官方 `_update_networks()` | 302 维 transport 经完整 `Agent.update()` 拆成 140/162 维 |
| `str_trained` | 恢复真实已训练权重、优化器等状态，再走官方核心更新 | 恢复相同状态，走完整适配版 `Agent.update()` |

`fresh` 指网络与 Adam 新初始化；reward normalizer 使用同一份真实训练统计，以免不合适的尺度使目标全部饱和。前两组首次 TD target 内部区间概率质量均约 0.88；训练态组约为 1.0，不是目标全被裁到边界后得到的无效一致。

**官方原生 Agent 不支持 STR 的独立双观测 transport。** 因此 STR 情形证明的是“适配版观测拆分 + 官方学习核心更新”的等价，不冒充官方原入口直接支持 STR。

## 实际结果

| 设置 | 情形 | 每组更新数 | 严格结果 |
|---|---|---:|---|
| CPU FP32、确定性 | 三种情形 | 6 | 全部逐元素一致，最大差值 0 |
| GPU FP32、确定性 | 真实训练态 | 6 | 全部逐元素一致，最大差值 0 |
| GPU AMP、确定性 | 真实训练态 | 6 | 全部逐元素一致，最大差值 0 |
| GPU compile + AMP、确定性 | 真实训练态 | 6 | 全部逐元素一致，最大差值 0 |
| 保留训练 TF32/benchmark，开启确定性 | 真实训练态 | 6 | 全部逐元素一致，最大差值 0 |
| 默认训练数值开关，非确定性 | 真实训练态 | 6 | 严格逐位比较未通过，详见下节 |

网络参数确实发生了非零变化。例如确定性编译组，六次更新后相对初始副本的最大参数变化：Actor 约 0.000700，Critic 约 0.001106，目标网络约 0.000341，温度参数约 0.000120。这不是双方都没更新造成的相同。

原始报告：

- [CPU 三组](cpu_final/result.json)
- [GPU FP32 / AMP](cuda/result.json)
- [确定性编译 AMP](compiled_v2/result.json)
- [保留 TF32 的确定性对照](tf32_deterministic/result.json)

## 默认 GPU 数值模式的差异：不隐瞒，不直接算迁移错误

保留训练时 `TF32=True`、cuDNN benchmark、compile、AMP，且不强制确定性算法时：

- 跨版本首次观察到的差异在 Critic 梯度，最大约 `9.54e-7`；首次更新的模型状态最大差约 `1.19e-7`。
- 六次更新后模型状态最大差约 `1.088e-4`。这里“模型状态”包括 BN buffers，不能把这个数全部称为可训练权重误差。
- **官方版本与自身重复**也产生相同量级的差异，本次记录的逐步模型状态最大差甚至与跨版本对照一致。
- **适配版本与自身重复**也并非所有字段逐位相同：出现约 `5.82e-11` 的梯度差和约 `7.28e-12` 的 Adam 动量差，但本次六步模型状态逐位相同。
- 保留 TF32 等开关，仅开启确定性算法后，跨版本重新完全一致。因此不能把 TF32 本身说成随机源。

这支持本次差异与非确定性数值执行有关，**未发现迁移特有的更新公式差异**；不证明数值差异对长期训练一定无害，也不修改严格测试阈值。

原始报告：[跨版本](trainflags/result.json)、[官方自身重复](official_self/result.json)、[适配版自身重复](adapted_self/result.json)。这些报告的 `pass_exact=false` 均保留。

编译取证的初次尝试在读取过期 CUDA graph 梯度缓存时失败。后改为优化器更新后立即复制梯度，只修正测试记录时机，未改算法。初次日志保留在 `compiled/`，有效结果在 `compiled_v2/`。

## 终止和时间截断：不以错误的上游行为为标准

新增 `tests/test_str_td_target_oracle.py` 使用独立标量 Bellman 累加和 bisect 插值投影，不复制 tensor scatter 实现：

- 真实 Torch replay 的 n=3、有效 k=1/2/3，覆盖终止、超时、两标志并存和正常继续。
- 真终止不 bootstrap；仅超时使用 `gamma**k`；两标志同时为真时终止优先。
- 检查概率质量、饱和、精确 bin、插值和熵项符号。
- **负对照**故意把短超时的 discount 改回固定 `gamma**3`，k=1/2 均能被检出；正确终止与完整 k=3 不误报。

此 oracle 的容差为 `atol=3e-6, rtol=0`，对照脚本的完整 n=3 比较仍要求逐元素精确相等。前者不是为让跨版本测试通过而放宽阈值。

回归实测：`28 passed, 10 subtests passed`。范围为 external adapter、n-step、n3 配置、holding 旁路统计、replay 恢复和本次 oracle。它们不等价于完整真实物理环境证明。

## 如何复现

在 `/home/abao/flashsac-robotics` 下，用已安装的 Python，无需启动 Isaac Sim：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python scripts/check_flashsac_update_parity.py --output diagnostics/update_parity_recheck/cpu --steps 6
```

保留训练数值开关、开启确定性运算的 GPU 对照：

```bash
/home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python scripts/check_flashsac_update_parity.py --output diagnostics/update_parity_recheck/gpu --cases str_trained --modes cuda_compiled_amp_trainflags_deterministic --steps 6
```

CPU 回归：

```bash
CUDA_VISIBLE_DEVICES='' TORCH_COMPILE_DISABLE=1 /home/abao/Documents/Codex/flashsac-official-play/.venv/bin/python -m pytest -q tests/test_external_isaaclab_adapter.py tests/test_n_step_discount.py tests/test_str_nstep3_config.py tests/test_str_holding_metrics.py tests/test_torch_buffer_resume.py tests/test_str_td_target_oracle.py
```

脚本依赖上面固定的官方 checkout、真实训练配置和 checkpoint/replay。每次使用新的 output 目录可保留历史报告。过程中只产生临时网络副本；临时 tensor traces 在比较后清理，JSON 和日志保留。

## 证据边界与下一步

这是两批数据、六次连续更新、指定模型状态和数值模式上的可复查证据，不是所有可能输入的形式化证明。它没有验证真实环境的观测构造、动作物理含义、终止分类、全局资产分布或长期收敛。

**下一步应补齐完整 STR 原入口与 wrapper 的同动作对照及官方策略成功正对照。之后再做学习率日程单变量实验。** 本次没有开始这两项工作，也没有改动现有训练。
