#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/data/chocheng/.cache/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize E06-R2")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/riemannian_local_deformation_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/riemannian_local_deformation_v1")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key, "")
    return np.nan if value in (None, "", "None") else float(value)


def main():
    args = parse_args(); config = json.loads(args.config.read_text())
    baselines = {row["window_id"]: row for row in read_csv(args.output / "window_baselines.csv")}
    finals = read_csv(args.output / "final_window_results.csv")
    by_window: dict[str, dict[str, dict]] = {}
    for row in finals:
        by_window.setdefault(row["window_id"], {})[row["method"]] = row
    if len(by_window) != 60 or any(set(value) != {"M1", "M2"} for value in by_window.values()):
        raise RuntimeError("summary requires 60 complete paired windows")
    paired = []
    for window_id in sorted(by_window):
        baseline = baselines[window_id]; m1 = by_window[window_id]["M1"]; m2 = by_window[window_id]["M2"]
        delta_e = (number(baseline, "L_q0") - number(m1, "L_q")) / number(baseline, "L_q0")
        delta_r = (number(baseline, "L_q0") - number(m2, "L_q")) / number(baseline, "L_q0")
        paired.append({
            "window_id": window_id, "surface_id": baseline["surface_id"], "anisotropy_level": baseline["anisotropy_level"],
            "placement_id": baseline["placement_id"], "A_window": number(baseline, "A_window"),
            "L_q0": number(baseline, "L_q0"), "L_qE": number(m1, "L_q"), "L_qR": number(m2, "L_q"),
            "L_surface0": number(baseline, "L_surface0"), "L_surfaceE": number(m1, "L_surface"), "L_surfaceR": number(m2, "L_surface"),
            "C_q0": number(baseline, "L_q0") / number(baseline, "L_surface0"), "C_qE": number(m1, "C_q"), "C_qR": number(m2, "C_q"),
            "Delta_E": delta_e, "Delta_R": delta_r, "A_R": delta_r - delta_e, "M2_wins": number(m2, "L_q") < number(m1, "L_q"),
            "E_NUC0": number(baseline, "E_NUC0"), "E_NUCE": number(m1, "E_NUC"), "E_NUCR": number(m2, "E_NUC"),
            "E_miss0": number(baseline, "E_miss0"), "E_missE": number(m1, "E_miss"), "E_missR": number(m2, "E_miss"),
            "E_rep0": number(baseline, "E_rep0"), "E_repE": number(m1, "E_rep"), "E_repR": number(m2, "E_rep"),
            "surface_change_E": number(m1, "relative_surface_length_change"), "surface_change_R": number(m2, "relative_surface_length_change"),
            "terminal_E": number(m1, "terminal_q_mismatch"), "terminal_R": number(m2, "terminal_q_mismatch"),
            "sigma0": number(baseline, "min_sigma0"), "sigmaE": number(m1, "min_sigma_min_5"), "sigmaR": number(m2, "min_sigma_min_5"),
            "joint_margin0": number(baseline, "joint_margin0"), "joint_marginE": number(m1, "min_joint_limit_margin"), "joint_marginR": number(m2, "min_joint_limit_margin"),
            "baseline_reconstruction_error": number(baseline, "baseline_reconstruction_error"),
            "L_G0": number(baseline, "L_G0"), "L_GE": number(m1, "L_G_posthoc") if np.isfinite(number(m1, "L_G_posthoc")) else number(m1, "L_G"), "L_GR": number(m2, "L_G"),
            "accepted_E": int(m1["accepted_iterations"]), "accepted_R": int(m2["accepted_iterations"]),
        })
    write_csv(args.output / "paired_window_results.csv", paired)
    with gzip.open(args.output / "candidate_history.csv.gz", "rt", newline="") as handle:
        history = list(csv.DictReader(handle))
    summary = make_summary(paired, history, config)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_plots(paired, history, args.output)
    (args.output / "summary.md").write_text(summary_markdown(summary))
    commands = [
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_riemannian_local_deformation.py --stage freeze-windows",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/run_riemannian_local_deformation.py --stage run --jobs 16",
        "/data/chocheng/.venvs/coverage-fm/bin/python scripts/summarize_riemannian_local_deformation.py",
        "/data/chocheng/.venvs/coverage-fm/bin/python -m pytest -q",
    ]
    (args.output / "reproduction_commands.txt").write_text("\n".join(commands) + "\n")
    print(json.dumps(summary["primary"], indent=2))


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def bootstrap_median_ci(data, seed, samples):
    data = np.asarray(data, dtype=np.float64); rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(data, size=(samples, len(data)), replace=True), axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def subset_summary(rows, config):
    if not rows:
        return {"count": 0}
    seed = config["statistics"]["bootstrap_seed"]; samples = config["statistics"]["bootstrap_samples"]
    return {
        "count": len(rows), "median_Delta_E": float(np.median(values(rows, "Delta_E"))),
        "median_Delta_R": float(np.median(values(rows, "Delta_R"))), "median_A_R": float(np.median(values(rows, "A_R"))),
        "iqr_A_R": [float(np.quantile(values(rows, "A_R"), 0.25)), float(np.quantile(values(rows, "A_R"), 0.75))],
        "median_A_R_bootstrap_95ci": bootstrap_median_ci(values(rows, "A_R"), seed, samples),
        "M2_win_fraction": float(np.mean([row["M2_wins"] for row in rows])),
        "median_surface_change_E": float(np.median(values(rows, "surface_change_E"))),
        "median_surface_change_R": float(np.median(values(rows, "surface_change_R"))),
        "median_terminal_E": float(np.median(values(rows, "terminal_E"))), "median_terminal_R": float(np.median(values(rows, "terminal_R"))),
    }


def make_summary(paired, history, config):
    medium_high = [row for row in paired if row["anisotropy_level"] in ("P_mid", "P_high")]
    corr = spearmanr(values(paired, "A_window"), values(paired, "A_R")).statistic
    regression = linregress(values(paired, "A_window"), values(paired, "A_R"))
    rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    x_all, y_all = values(paired, "A_window"), values(paired, "A_R")
    slopes = []
    for _ in range(config["statistics"]["bootstrap_samples"]):
        index = rng.integers(0, len(paired), len(paired))
        slopes.append(linregress(x_all[index], y_all[index]).slope)
    slope_ci = [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))]
    sigma_cut = float(np.quantile(values(medium_high, "sigma0"), 0.25))
    sensitivities = {
        "remove_lowest_sigma_quartile": [row for row in medium_high if row["sigma0"] > sigma_cut],
        "surface_length_mismatch_le_1pct": [row for row in medium_high if max(row["surface_change_E"], row["surface_change_R"]) <= config["admission"]["maximum_relative_surface_length_change_sensitivity"]],
        "terminal_mismatch_le_0p025": [row for row in medium_high if max(row["terminal_E"], row["terminal_R"]) <= config["admission"]["maximum_terminal_q_mismatch_sensitivity_rad"]],
    }
    mh = subset_summary(medium_high, config); gate = config["gate"]
    checks = {
        "M2_win_fraction": bool(mh["M2_win_fraction"] >= gate["minimum_medium_high_M2_win_fraction"]),
        "median_incremental_benefit": bool(mh["median_A_R"] >= gate["minimum_medium_high_median_incremental_benefit"]),
        "median_riemannian_improvement": bool(mh["median_Delta_R"] >= gate["minimum_medium_high_median_riemannian_improvement"]),
        "anisotropy_interaction": bool(corr >= gate["minimum_anisotropy_spearman"] or slope_ci[0] > 0.0),
    }
    sensitivity_summaries = {name: subset_summary(rows, config) for name, rows in sensitivities.items()}
    robust_direction = all(value.get("median_A_R", -np.inf) > 0.0 and value.get("M2_win_fraction", 0.0) > 0.5 for value in sensitivity_summaries.values())
    checks["sensitivity_robustness"] = bool(robust_direction)
    candidate_admission = {}
    for method in ("M1", "M2"):
        selected = [row for row in history if row["method"] == method]
        candidate_admission[method] = {"evaluations": len(selected), "admitted": sum(row["admitted"] == "True" for row in selected), "admission_rate": float(np.mean([row["admitted"] == "True" for row in selected]))}
    controls = {
        "maximum_baseline_reconstruction_error": float(max(row["baseline_reconstruction_error"] for row in paired)),
        "maximum_surface_length_change": float(max(max(row["surface_change_E"], row["surface_change_R"]) for row in paired)),
        "maximum_terminal_q_mismatch": float(max(max(row["terminal_E"], row["terminal_R"]) for row in paired)),
        "maximum_absolute_E_NUC_change": float(max(max(abs(row["E_NUCE"] - row["E_NUC0"]), abs(row["E_NUCR"] - row["E_NUC0"])) for row in paired)),
        "maximum_absolute_E_miss_change": float(max(max(abs(row["E_missE"] - row["E_miss0"]), abs(row["E_missR"] - row["E_miss0"])) for row in paired)),
        "maximum_absolute_E_rep_change": float(max(max(abs(row["E_repE"] - row["E_rep0"]), abs(row["E_repR"] - row["E_rep0"])) for row in paired)),
        "minimum_sigma_after": float(min(min(row["sigmaE"], row["sigmaR"]) for row in paired)),
        "spearman_A_R_vs_sigma0": float(spearmanr(values(paired, "A_R"), values(paired, "sigma0")).statistic),
        "spearman_A_R_vs_joint_margin0": float(spearmanr(values(paired, "A_R"), values(paired, "joint_margin0")).statistic),
        "equal_candidate_budget": all(value["evaluations"] == 3600 for value in candidate_admission.values()),
        "baseline_window_length_range_m": [float(np.min(values(paired, "L_surface0"))), float(np.max(values(paired, "L_surface0")))],
    }
    m2_admitted = [row for row in history if row["method"] == "M2" and row["admitted"] == "True"]
    metric_candidate_pearson = float(np.corrcoef([float(row["L_G"]) for row in m2_admitted], [float(row["L_q"]) for row in m2_admitted])[0, 1])
    metric_candidate_spearman = float(spearmanr([float(row["L_G"]) for row in m2_admitted], [float(row["L_q"]) for row in m2_admitted]).statistic)
    oracle_reductions = {}
    for method in ("M1", "M2"):
        reductions = []
        for row in paired:
            candidates = [item for item in history if item["method"] == method and item["window_id"] == row["window_id"] and item.get("L_q") not in (None, "")]
            reductions.append((row["L_q0"] - min(float(item["L_q"]) for item in candidates)) / row["L_q0"])
        oracle_reductions[method] = float(np.median(reductions))
    optimizer_diagnostics = {
        "M2_admitted_candidate_LG_Lq_pearson": metric_candidate_pearson,
        "M2_admitted_candidate_LG_Lq_spearman": metric_candidate_spearman,
        "median_best_evaluated_actual_Lq_reduction_M1": oracle_reductions["M1"],
        "median_best_evaluated_actual_Lq_reduction_M2": oracle_reductions["M2"],
        "interpretation": "The best-actual values are diagnostic candidate-set oracles, not method outputs; neither objective may rank them first.",
    }
    return {
        "experiment": config["experiment"], "statistical_unit": "window", "windows": len(paired),
        "primary": {"all": subset_summary(paired, config), "medium_high": mh, "anisotropy_spearman": float(corr), "regression_alpha": float(regression.intercept), "regression_beta": float(regression.slope), "regression_beta_bootstrap_95ci": slope_ci, "regression_r_squared": float(regression.rvalue**2)},
        "by_level": {level: subset_summary([row for row in paired if row["anisotropy_level"] == level], config) for level in ("P_low", "P_mid", "P_high")},
        "by_surface": {surface: subset_summary([row for row in paired if row["surface_id"] == surface], config) for surface in ("saddle", "hemisphere")},
        "candidate_admission": candidate_admission, "controls": controls, "optimizer_diagnostics": optimizer_diagnostics, "sensitivity": sensitivity_summaries,
        "gate_checks": checks, "decision": "GO" if all(checks.values()) else "NO-GO",
    }


def make_plots(rows, history, output):
    figures = output / "figures"; sources = output / "figure_sources"; figures.mkdir(exist_ok=True); sources.mkdir(exist_ok=True)
    write_csv(sources / "paired_window_source.csv", rows)
    ids = np.arange(len(rows)); labels = [row["window_id"] for row in rows]
    paired_plot(ids, [values(rows, key) for key in ("L_q0", "L_qE", "L_qR")], ["M0", "M1", "M2"], "Actual witness $L_q$", figures / "paired_Lq.png")
    paired_plot(ids, [values(rows, key) for key in ("C_q0", "C_qE", "C_qR")], ["M0", "M1", "M2"], "$C_q=L_q/L_{surface}$", figures / "paired_Cq.png")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(values(rows, "A_R"), bins=16); ax.axvline(0, color="k", lw=1); ax.set(xlabel="$A_R$", ylabel="Windows"); save(fig, figures / "incremental_benefit_distribution.png")
    fig, ax = plt.subplots(figsize=(6, 4));
    for level, marker in zip(("P_low", "P_mid", "P_high"), ("o", "s", "^")):
        part = [row for row in rows if row["anisotropy_level"] == level]; ax.scatter(values(part, "A_window"), values(part, "A_R"), label=level, marker=marker)
    fit = linregress(values(rows, "A_window"), values(rows, "A_R")); x = np.linspace(min(values(rows, "A_window")), max(values(rows, "A_window")), 100); ax.plot(x, fit.intercept + fit.slope*x, "k--"); ax.axhline(0, color="0.5"); ax.set(xlabel="$A_{window}$", ylabel="$A_R$"); ax.legend(); save(fig, figures / "anisotropy_vs_incremental_benefit.png")
    fig, ax = plt.subplots(figsize=(6, 4)); grouped = [values([row for row in rows if row["anisotropy_level"] == level], "A_R") for level in ("P_low", "P_mid", "P_high")]; ax.boxplot(grouped, tick_labels=["low", "mid", "high"]); ax.axhline(0, color="0.5"); ax.set(ylabel="$A_R$"); save(fig, figures / "benefit_by_anisotropy.png")
    make_anytime(history, figures, sources)
    histogram_comparison(rows, ("surface_change_E", "surface_change_R"), ("M1", "M2"), "Relative surface-length change", figures / "surface_length_changes.png")
    make_coverage_plot(rows, figures)
    histogram_comparison(rows, ("terminal_E", "terminal_R"), ("M1", "M2"), "Terminal q mismatch [rad]", figures / "terminal_q_mismatch.png")
    paired_plot(ids, [values(rows, key) for key in ("sigma0", "sigmaE", "sigmaR")], ["M0", "M1", "M2"], "Minimum $sigma_{min,5}$", figures / "sigma_before_after.png")
    predicted = []
    for row in rows:
        for method, lg, lq in (("M1", row["L_GE"], row["L_qE"]), ("M2", row["L_GR"], row["L_qR"])):
            predicted.append({"window_id": row["window_id"], "method": method, "predicted_LG_reduction": (row["L_G0"]-lg)/row["L_G0"], "actual_Lq_reduction": (row["L_q0"]-lq)/row["L_q0"]})
    write_csv(sources / "predicted_vs_actual_source.csv", predicted)
    fig, ax = plt.subplots(figsize=(6, 4));
    for method in ("M1", "M2"):
        part=[row for row in predicted if row["method"]==method]; ax.scatter(values(part,"predicted_LG_reduction"),values(part,"actual_Lq_reduction"),label=method,alpha=.75)
    ax.set(xlabel="Predicted $L_G$ reduction",ylabel="Actual $L_q$ reduction"); ax.legend(); save(fig, figures / "predicted_vs_actual_reduction.png")
    make_representative_paths(rows, output, figures, sources)


def paired_plot(x, series, labels, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for index in range(len(x)):
        ax.plot([0,1,2], [values[index] for values in series], color="0.8", lw=.6)
    for position, values_, label in zip(range(3), series, labels):
        ax.scatter(np.full(len(values_), position), values_, s=10, label=label)
    ax.set(xticks=[0,1,2], xticklabels=labels, ylabel=ylabel); save(fig, path)


def histogram_comparison(rows, keys, labels, xlabel, path):
    fig, ax=plt.subplots(figsize=(6,4));
    for key,label in zip(keys,labels): ax.hist(values(rows,key),bins=14,alpha=.55,label=label)
    ax.set(xlabel=xlabel,ylabel="Windows"); ax.legend(); save(fig,path)


def make_anytime(history, figures, sources):
    source=[]
    for method in ("M1","M2"):
        selected=[row for row in history if row["method"]==method]
        for evaluation in range(1,61):
            vals=[float(row["best_feasible_actual_L_q"]) for row in selected if int(row["evaluation"])==evaluation]
            source.append({"method":method,"evaluation":evaluation,"median_best_actual_L_q":float(np.median(vals))})
    write_csv(sources/"anytime_source.csv",source)
    fig,ax=plt.subplots(figsize=(6,4))
    for method in ("M1","M2"):
        part=[row for row in source if row["method"]==method]; ax.plot(values(part,"evaluation"),values(part,"median_best_actual_L_q"),label=method)
    ax.set(xlabel="Strict candidate evaluation",ylabel="Median best feasible actual $L_q$"); ax.legend(); save(fig,figures/"anytime_actual_Lq.png")


def make_coverage_plot(rows, figures):
    source=[]
    for row in rows:
        for method,suffix in (("M0","0"),("M1","E"),("M2","R")):
            source.append({"window_id":row["window_id"],"method":method,"dE_miss":row[f"E_miss{suffix}"]-row["E_miss0"],"dE_rep":row[f"E_rep{suffix}"]-row["E_rep0"],"dE_NUC":row[f"E_NUC{suffix}"]-row["E_NUC0"]})
    fig,ax=plt.subplots(figsize=(7,4)); data=[[entry[metric] for entry in source if entry["method"]==method] for method in ("M1","M2") for metric in ("dE_miss","dE_rep")]; ax.boxplot(data,tick_labels=["M1 miss","M1 rep","M2 miss","M2 rep"]); ax.axhline(0,color="0.5"); ax.set(ylabel="Coverage metric change"); save(fig,figures/"coverage_metric_changes.png")


def make_representative_paths(rows, output, figures, sources):
    frozen=json.loads((output/"frozen_windows.json").read_text()); finals={row["window_id"]+"/"+row["method"]:row for row in read_csv(output/"final_window_results.csv")}
    selected=[]
    for surface in ("saddle","hemisphere"):
        candidates=[row for row in rows if row["surface_id"]==surface and row["anisotropy_level"]=="P_high"]
        selected.append(max(candidates,key=lambda row:abs(row["A_R"])))
    fig=plt.figure(figsize=(12,5)); source=[]
    from diffusion_coverage.diagnostics.e06_artifacts import load_e06_contract, make_e06_surface
    from diffusion_coverage.diagnostics.local_deformation import deform_surface_path
    from diffusion_coverage.geometry.robot_surface_metric import compute_robot_surface_metric, estimate_surface_contact_differential
    from diffusion_coverage.robot.task_kinematics import evaluate_task_kinematics_5d
    from diffusion_coverage.robot.ur5e_mujoco import UR5eKinematics
    archived,_,_=load_e06_contract(ROOT); config=json.loads((ROOT/"configs/riemannian_local_deformation_v1.json").read_text())
    robot_cfg=archived["config"]["robot"]
    robot=UR5eKinematics(robot_cfg["model"],site_name=robot_cfg["site_name"],tool_axis_index=robot_cfg["tool_axis_index"],tool_axis_sign=robot_cfg["tool_axis_sign"])
    for column,row in enumerate(selected,1):
        window=next(item for group in frozen["windows_by_scene"].values() for item in group if item["window_id"]==row["window_id"]); surface=make_e06_surface(archived["config"],row["surface_id"],1); witness=np.load(ROOT/window["witness_file"]); transform=np.asarray(window["transform_base_from_surface"]); points=(witness["desired_positions"][window["start_index"]:window["stop_index"]+1]-transform[:3,3])@transform[:3,:3]; points,_=__import__("diffusion_coverage.geometry.robot_surface_metric",fromlist=["smooth_surface_normal"]).smooth_surface_normal(surface,points); controls=np.asarray(window["control_indices_local"])
        paths={"M0":points}
        for method in ("M1","M2"):
            params=np.asarray(json.loads(finals[row["window_id"]+"/"+method]["parameters"])); paths[method]=deform_surface_path(surface,points,controls,params,maximum_displacement=config["window"]["maximum_control_displacement_m"],maximum_retraction_step=config["window"]["retraction_maximum_step_m"]).points
        ax=fig.add_subplot(1,2,column,projection="3d")
        for method,color in zip(("M0","M1","M2"),("black","tab:blue","tab:red")):
            p=paths[method]; ax.plot(p[:,0],p[:,1],p[:,2],label=method,color=color); [source.append({"window_id":row["window_id"],"method":method,"sample":i,"x":v[0],"y":v[1],"z":v[2]}) for i,v in enumerate(p)]
        q=witness["q"][window["start_index"]:window["stop_index"]+1]
        for index in np.unique(np.linspace(0,len(points)-1,5,dtype=int)):
            task=evaluate_task_kinematics_5d(robot,q[index],characteristic_length=config["robot_contract"]["characteristic_length_m"])
            contact=estimate_surface_contact_differential(surface,points[index],transform,task.axis_basis,characteristic_length=config["robot_contract"]["characteristic_length_m"],finite_difference_step=config["metric"]["finite_difference_step_m"])
            metric=compute_robot_surface_metric(task,contact.task_differential,minimum_singular_value=config["robot_contract"]["sigma_safe"])
            tangent_surface=transform[:3,:3].T@contact.tangent_basis
            for eigen_index,color,name in ((0,"tab:green","cheap_eigenvector"),(1,"tab:orange","expensive_eigenvector")):
                direction=tangent_surface@metric.eigenvectors[:,eigen_index]; scale=.004
                ax.quiver(*points[index],*(scale*direction),color=color,linewidth=1)
                source.append({"window_id":row["window_id"],"method":name,"sample":int(index),"x":points[index,0],"y":points[index,1],"z":points[index,2],"dx":scale*direction[0],"dy":scale*direction[1],"dz":scale*direction[2]})
        ax.set_title(row["window_id"]); ax.legend()
    write_csv(sources/"representative_paths_source.csv",source); save(fig,figures/"representative_deformed_windows.png")


def save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def write_csv(path, rows):
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summary_markdown(summary):
    p=summary["primary"]["medium_high"]
    return f"""# E06-R2 summary\n\nDecision: **{summary['decision']}**\n\n- Windows: {summary['windows']}\n- Medium/high M2 win fraction: {p['M2_win_fraction']:.4f}\n- Medium/high median Delta_E: {p['median_Delta_E']:.4f}\n- Medium/high median Delta_R: {p['median_Delta_R']:.4f}\n- Medium/high median A_R: {p['median_A_R']:.4f}\n- Spearman(A_window,A_R): {summary['primary']['anisotropy_spearman']:.4f}\n- Regression beta: {summary['primary']['regression_beta']:.4f}, bootstrap 95% CI {summary['primary']['regression_beta_bootstrap_95ci']}\n- Candidate admission: M1 {summary['candidate_admission']['M1']['admission_rate']:.4f}, M2 {summary['candidate_admission']['M2']['admission_rate']:.4f}\n- Maximum surface-length change: {summary['controls']['maximum_surface_length_change']:.6f}\n- Maximum terminal-q mismatch: {summary['controls']['maximum_terminal_q_mismatch']:.6f} rad\n- Maximum absolute E_NUC change: {summary['controls']['maximum_absolute_E_NUC_change']:.6f}\n- Minimum post-deformation sigma_min_5: {summary['controls']['minimum_sigma_after']:.6f}\n\nThis local numerical experiment does not establish whole-path NUC planning, global optimality, hardware performance, learned-planner benefit, cross-robot generality, or C-space topology.\n"""


if __name__ == "__main__":
    main()
