# hand EMA 0.1 → 1：100M结果（2026-09-10）

结论：在本次单seed、同100M预算下，hand=1明显退步，不能据此继续推荐完全取消手平滑。恢复hand=.1作为下一组基准。尚未训成稳定抓取搬运。

训练和play均完成，training_complete.json、play/play_complete.json为complete；最终step48830，replay包含10,000,000条、ring index3840、大小25,480,003,499字节。CPU mmap读取字段/形状成功，没有重新加载到GPU。训练2739秒，play395秒。

## 同预算训练窗口

90,003,840 < transitions <= 100,003,840，99个日志窗口等权均值；不是所有episode按数量加权的精确率。

| 指标 | arm1/hand.1基准 | arm1/hand1 |
|---|---:|---:|
| Return | 529.02 | 329.06 |
| Goals/episode | .15134 | .000320 |
| Lift bonus | 275.84 | 275.62 |
| Keypoint reward | 21.19 | 6.71 |
| Goal bonus | 182.28 | 2.87 |
| 手速度惩罚 | -7.39 | -15.59 |
| Episode步数 | 96.77 | 79.51 |

约92%曾越过抬升高度没有变，但goal和后续保持明显恶化。不能用lift bonus或更低的done_fall单独宣布变好。

## 同协议配对play

每格为基准 → hand1；确定性和随机各64个首episode，全部完成，tol=.075。

| 指标 | 确定性 | 原生随机 |
|---|---:|---:|
| 曾抬高超过10cm | 62→53 | 59→53 |
| 连续抬高至少1秒 | 8→1 | 7→0 |
| 至少到达1个goal | 11→0 | 10→0 |
| 总goal数 | 32→0 | 28→0 |
| fall终止 | 54→40 | 55→44 |
| hand_far终止 | 8→24 | 8→21 |
| timeout | 2→1 | 2→1 |

终止条件可重叠。新组63/64（两模式各自）发生fall或hand_far真终止；更低的fall计数来自失败类型转移，不是抓稳了。连续抬高是高度代理，没有ContactSensor，不是抓握成功率。

## 证据核验

基准play路径：../control_ab_20260909_arm1/play_20260909_v4。两组同模式的初始obs、joint_pos/vel、object、goal、prev/cur targets全部逐元素相同。资产文件名和env分配相同；临时目录不同属于正常生成路径变化。

新组play前后model digest均ee961afad4d1c80c854055bbdff9a760148d34858994e4764f7fb13d4b1f9a28，未训练或改模型。actor.pt SHA256为73887aa2fc044bb2bb7cf607c16d4b64f1c32397fb3e07faccedcef7635fa7f6。

视频play/paired_play.mp4经ffprobe验证h264、960×720、30fps、1200帧、40秒。已查看0.7秒四视角帧，所选首episode物体已明显离开桌面/手附近或正在落下；视频后续自动reset不纳入首episode统计。

下一组只测n_step 1→3，详见../credit_nstep3_20260910/README.md；完整STR任务、reward和网络不改，不将本次失败当成FlashSAC不能完成任务的证明。
