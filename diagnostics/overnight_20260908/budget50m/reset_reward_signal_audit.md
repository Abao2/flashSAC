# Reset reward字段：CPU配对敏感性

同一批32个真实fresh-reset观测，只把reward字段[161:162]从0替换为1.0014024353027344；该值来自128个成功BC终止回合的raw reward中位数100.14024353027344×.01。其余字段完全不变，两网络均未更新。

| 模型 | deterministic动作绝对变化均值/最大 | pre-tanh sigma绝对变化均值 | 条件tanh动作std：前→后 |
| --- | ---: | ---: | ---: |
| native hand.1 50M | .0195893 / .136699 | .0347952 | .198230→.195761 |
| selected BC control | .00268287 / .00774628 | .00208684 | .00198443→.00215321 |

不是rollout成功率，也不是已确认的失败原因。std是8个独立MC噪声/state、共同随机数的条件动作波动估计；不含原生重复噪声时间相关或controller滤波。BC原评测使用arm.1/hand.1；此处仅比较同一观测的网络响应，不做控制器比较。

代码链、实际轨迹和完整解释见[train/eval核对](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/train_eval_mismatch_audit.md)。完整数字、输入来源和SHA256在[JSON](/home/abao/flashsac-robotics/diagnostics/overnight_20260908/budget50m/reset_reward_signal_audit.json)。
