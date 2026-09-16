# E08 global completion bound v1 — evidence report

## 1. 当前进度

Audit、qualification、bound validation、graph build、S0/S1 compare 和 independent verification 均已执行。共 0 个场景取得 P 合同参考，构建 0 张冻结图；最终验证记录 0 条，其中 0 条在加密检查下通过。实验合同没有在结果后放松。

## 2. IK 诊断

实际 XML 审计了 21 个已解析文件。q6 扰动的最大位置变化为 1.11022e-16 m，最大轴变化为 6.664e-08 rad，碰撞状态变化 0 次；据任务、范围和已建模碰撞，纯 q6 消去判定为 `True`。六行与五行完整路径结果见 `ik_comparison.csv`；这是实现诊断，不计规划创新。轴反向的 cross residual 退化单独记录在 `preflight.json`。

## 3. 数学正确性

确定性随机小图 32 张，检查 3079 个前缀，其中 2912 个可完成。压缩状态与完整 visit-count oracle 不一致 0 次；不可采纳下界 0 次；误剪枝 0 次；S0/S1 与独立 oracle 搜索不一致 0 次。固定 membership 的分段组合、活动状态、共享瓶颈、面积分位和 OFF 重构案例由单元测试覆盖。

## 4. 完整曲面主结果

资格通过场景：无。主比较共 0 次；加密最终检查通过 0/0。S0/S1 完整结束的配对若有差异，运行器会直接报错而不生成本报告。完整逐场景结果在 `search_results.csv`、`anytime.csv` 和 `final_verification.csv`。

## 5. 机制解释

S1 自然产生 0 次 completion-bound 剪枝，下界自身累计耗时 0 s。相对 S0 的 expanded 标签净减少总和为 0，搜索 wall-clock 净减少总和为 0 s。图构建总耗时为 0 s。自然机制样例见 `mechanism_example.json`；若文件标记未找到，则没有人工删边制造样例。

## 6. 解释边界

下界结论只对每个 graph hash 对应的冻结有限图、离散 footprint membership 和 ON 段预算成立。未采样或 IK 未找到的边不代表物理不存在。最终结果是 sampled checker admission，不是连续区间证书。教师来自四个预注册有限路径族；失败场景继续保留在资格表。J_q 是关节路径长度，不解释为能耗、时间或力跟踪表现。

## 7. 本机研究判断

**实验前提或实现失败：没有场景取得 P 合同完整参考。** 该判断只描述本机 E08 原型，不是 RSS 判断，也不判决整个全局规划课题。没有启动 FM。
