#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"results/global_completion_bound_v1"
ARA=Path("/data/chocheng/worktrees/global-completion-bound-v1-ara")


def read_csv(path):
    if not path.exists(): return []
    with path.open(newline="") as handle:return list(csv.DictReader(handle))


def yes(value):return str(value).lower()=="true"
def number(value):
    try:return float(value)
    except (TypeError,ValueError):return None


def main():
    pre=json.loads((OUTPUT/"preflight.json").read_text()); validation=json.loads((OUTPUT/"bound_validation.json").read_text())
    qualify=read_csv(OUTPUT/"canonical_qualification.csv"); search=read_csv(OUTPUT/"search_results.csv"); prune=read_csv(OUTPUT/"prune_breakdown.csv"); verify=read_csv(OUTPUT/"final_verification.csv")
    graphs=json.loads((OUTPUT/"graph_manifest.json").read_text())["graphs"]
    ik=read_csv(OUTPUT/"ik_comparison.csv")
    qualified=sorted({(r["surface_id"],r["scene_id"]) for r in qualify if yes(r["reference_found"])})
    verified_pass=sum(yes(r.get("verification_overall_pass")) for r in verify)
    paired=[]
    for key in sorted({(r["surface_id"],r["scene_id"],r["k"]) for r in search}):
        values={r["arm"]:r for r in search if (r["surface_id"],r["scene_id"],r["k"])==key}
        if set(values)=={"S0","S1"}:
            paired.append((key,values))
    expanded_delta=[]; time_delta=[]; bound_prunes=0; bound_time=0.0
    for _,values in paired:
        expanded_delta.append(int(values["S0"]["expanded"])-int(values["S1"]["expanded"]))
        time_delta.append(float(values["S0"]["total_search_s"])-float(values["S1"]["total_search_s"]))
    for row in prune:
        if row["arm"]=="S1":
            bound_prunes+=int(row["completion_bound"]); bound_time+=float(row["bound_time_s"])
    if validation["false_prunes"] or validation["inadmissible_prefixes"]:
        category="实验前提或实现失败：下界正确性门未通过"
    elif not qualified:
        category="实验前提或实现失败：没有场景取得 P 合同完整参考"
    elif bound_prunes==0:
        category="正确但无额外信息"
    elif sum(time_delta)>0:
        category="有用：有限图上存在净搜索时间收益"
    else:
        category="信息有用但太贵：剪枝增加但总时间未下降"
    task=pre["task_freedom_audit"]
    lines=[
        "# E08 global completion bound v1 — evidence report","",
        "## 1. 当前进度","",
        f"Audit、qualification、bound validation、graph build、S0/S1 compare 和 independent verification 均已执行。共 {len(qualified)} 个场景取得 P 合同参考，构建 {sum(r['status']=='ready' for r in graphs)} 张冻结图；最终验证记录 {len(verify)} 条，其中 {verified_pass} 条在加密检查下通过。实验合同没有在结果后放松。","",
        "## 2. IK 诊断","",
        f"实际 XML 审计了 {len(pre['model_dependencies'])} 个已解析文件。q6 扰动的最大位置变化为 {task['maxima']['q6_perturb_position_change_m']:.6g} m，最大轴变化为 {task['maxima']['q6_perturb_axis_change_rad']:.6g} rad，碰撞状态变化 {task['collision_changes']} 次；据任务、范围和已建模碰撞，纯 q6 消去判定为 `{task['pure_q6_elimination_allowed']}`。六行与五行完整路径结果见 `ik_comparison.csv`；这是实现诊断，不计规划创新。轴反向的 cross residual 退化单独记录在 `preflight.json`。","",
        "## 3. 数学正确性","",
        f"确定性随机小图 {validation['random_graphs']} 张，检查 {validation['prefixes']} 个前缀，其中 {validation['completable_prefixes']} 个可完成。压缩状态与完整 visit-count oracle 不一致 {validation['oracle_state_mismatches']} 次；不可采纳下界 {validation['inadmissible_prefixes']} 次；误剪枝 {validation['false_prunes']} 次；S0/S1 与独立 oracle 搜索不一致 {validation['search_mismatches']} 次。固定 membership 的分段组合、活动状态、共享瓶颈、面积分位和 OFF 重构案例由单元测试覆盖。","",
        "## 4. 完整曲面主结果","",
        f"资格通过场景：{', '.join('/'.join(v) for v in qualified) if qualified else '无'}。主比较共 {len(search)} 次；加密最终检查通过 {verified_pass}/{len(verify) if verify else 0}。S0/S1 完整结束的配对若有差异，运行器会直接报错而不生成本报告。完整逐场景结果在 `search_results.csv`、`anytime.csv` 和 `final_verification.csv`。","",
        "## 5. 机制解释","",
        f"S1 自然产生 {bound_prunes} 次 completion-bound 剪枝，下界自身累计耗时 {bound_time:.6g} s。相对 S0 的 expanded 标签净减少总和为 {sum(expanded_delta)}，搜索 wall-clock 净减少总和为 {sum(time_delta):.6g} s。图构建总耗时为 {sum(float(r.get('build_s',0)) for r in graphs):.6g} s。自然机制样例见 `mechanism_example.json`；若文件标记未找到，则没有人工删边制造样例。","",
        "## 6. 解释边界","",
        "下界结论只对每个 graph hash 对应的冻结有限图、离散 footprint membership 和 ON 段预算成立。未采样或 IK 未找到的边不代表物理不存在。最终结果是 sampled checker admission，不是连续区间证书。教师来自四个预注册有限路径族；失败场景继续保留在资格表。J_q 是关节路径长度，不解释为能耗、时间或力跟踪表现。","",
        "## 7. 本机研究判断","",
        f"**{category}。** 该判断只描述本机 E08 原型，不是 RSS 判断，也不判决整个全局规划课题。没有启动 FM。","",
    ]
    text="\n".join(lines)+"\n"
    (OUTPUT/"summary.md").write_text(text)
    evidence=ARA/"evidence/global_completion_bound_v1_2026-09-16.md"
    if evidence.exists(): raise FileExistsError(f"refusing to overwrite append-only evidence: {evidence}")
    evidence.write_text(text)
    print(category)


if __name__=="__main__":main()
