# 100M vs 500M：固定同难度检查策略是否退步

状态：已准备，未启动GPU评测。先等当前500M训练正常结束并核验完整保存；不要与正在运行的Isaac训练同时启动。

触发证据：容差自132259840固定为.029056536，400–420M、420–440M、440–460M的训练窗口平均目标数分别1.5171、1.4496、1.3478；reward分别1903.28、1835.92、1733.62。没有NaN/Inf，但没有满足继续加至1B的明确进步条件。

使用现有play_str_full_diagnostic.py，唯一代码改动是参数化原先硬编码的.075。新增--tolerance默认仍.075，所有冻结、元数据、每步断言统一用请求值；22项CPU测试通过。训练入口、reward、reset、课程、网络不改。

两个模型均使用simtoolreal_full_arm1_nstep3、seed0、64env、每模式1200步（20秒）、固定tol=.02905653603374958，顺序运行旧100M和新500M。各自包含deterministic和原生stochastic两种模式并录视频。每个env只计首episode；20秒仍未结束为censored，不是失败。goal-positive不是50-goal全链成功；观察到的总goals包含删失轨迹的已完成部分，不等于完整回合均值。

评测结束后必须CPU比较两次运行的deterministic_initial.npz、stochastic_initial.npz全部数组（finite、相同shape，max_abs<=1e-5），以及env_assets.json的env_id/asset_idx/asset_type/文件basename（只忽略临时目录）；不能只凭seed声称初态匹配。检查两份play_complete.json的status=complete、tol正确、model_before=model_after，再比较每模式的64轨迹goal-positive、total observed goals、completed/censored、fall/hand_far失败和全50-goal完成数。

这不是原论文24任务评测，不代表真实抓取成功或真机可部署；本轮只回答相同任务难度下的旧/新策略差异。不得把当前平台期说成收敛，也不自动追加1B训练。

启动前检查GPU无其他冲突。使用manifest.json与run_queue.py现有串行队列；队列启动门槛会要求父训练成功且500M的7个保存文件非空。当前日截止22:30（UTC14:30），不使用旧诊断过期的CLI截止时间。读取queue_status.json和old_100m/new_500m的play_complete.json判断完成，不只看进程是否存在。
