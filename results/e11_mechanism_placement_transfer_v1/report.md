# E11 mechanism attribution and frozen-policy placement transfer

Executed: 2026-09-17T18:41:42.325563+10:00

E11 froze E10 atomic method A, added the prefix-free A_root ablation and the equally informed one-continuation greedy control P, then built six prospectively registered placement graphs. A result is accepted only after the inherited achieved-FK T0/Q3, T0/Q4, T1/Q4 and Q4a checks plus robot, activity and same-sample composition checks.

## DEV attribution

| scene | F | A | A_root | P |
|---|---|---|---|---|
| T30 | failure_or_limit; queue_exhausted | new_global_plan; Jq 96.949 | failure_or_limit; retained_record_budget | new_global_plan; Jq 96.949 |
| T27 | fixed_route; Jq 90.011 | retained_F; Jq 90.011 | retained_F; Jq 90.011 | retained_F; Jq 90.011 |
| T33 | failure_or_limit; queue_exhausted | new_global_plan; Jq 96.153 | failure_or_limit; retained_record_budget | new_global_plan; Jq 96.153 |

T30 and T33: A and P returned the same accepted witness; A_root returned none accepted. T27: F supplied the accepted route and every global arm retained it. Thus prefix information mattered on the two E10 positive DEV cases, while retaining competing alternatives did not: P matched A with 677/735 expansions versus A's 36,910/34,705.

## TRANSFER — all 36 cells

| scene | k | F | P | A |
|---|---:|---|---|---|
| H00 | 1 | failure_or_limit | new_global_plan, Jq 93.654 | new_global_plan, Jq 93.654 |
| H00 | 2 | failure_or_limit | new_global_plan, Jq 94.106 | new_global_plan, Jq 93.654 |
| H01 | 1 | fixed_route, Jq 90.269 | retained_F, Jq 90.269 | improved_F, Jq 90.224 |
| H01 | 2 | fixed_route, Jq 90.269 | retained_F, Jq 90.269 | improved_F, Jq 90.224 |
| H02 | 1 | failure_or_limit | new_global_plan, Jq 88.148 | new_global_plan, Jq 88.146 |
| H02 | 2 | fixed_route, Jq 96.040 | improved_F, Jq 89.189 | improved_F, Jq 88.146 |
| H03 | 1 | fixed_route, Jq 86.329 | retained_F, Jq 86.329 | retained_F, Jq 86.329 |
| H03 | 2 | fixed_route, Jq 86.329 | retained_F, Jq 86.329 | retained_F, Jq 86.329 |
| H04 | 1 | failure_or_limit | new_global_plan, Jq 89.749 | new_global_plan, Jq 89.749 |
| H04 | 2 | failure_or_limit | new_global_plan, Jq 89.749 | new_global_plan, Jq 89.749 |
| H05 | 1 | fixed_route, Jq 92.519 | retained_F, Jq 92.519 | retained_F, Jq 92.519 |
| H05 | 2 | fixed_route, Jq 92.519 | retained_F, Jq 92.519 | retained_F, Jq 92.519 |

## Answers

1. **Q1 — prefix information.** Supported on DEV T30/T33: A passed and A_root did not. This is evidence for the deterministic F-prefix archive on these cases, not a general necessity result.
2. **Q2 — search complexity.** A's multi-label complexity was not necessary on the DEV positives. P reproduced A's exact accepted T30/T33 witnesses with far fewer expansions. The supported mechanism is therefore prefix-guided greedy recombination, while A remains the frozen transfer reference.
3. **Q3 — transfer.** A delivered independently accepted global benefit on H00, H01, H02 and H04 (four of six placements, identically for paired k rows): new routes where F had none on H00/H02-k1/H04, and lower Jq than F on H01 and H02-k2. H03/H05 only retained F. P delivered global benefit on H00, H02 and H04 and otherwise retained F. These are six deterministic pose perturbations in three anchor neighborhoods, not independent population samples or cross-surface generalization.
4. **Q4 — physical explanation in the frozen DEV graphs.** Of six saved cross-port decisions, 5 lacked the exact next fixed-family sampled transition at the actual prefix q state. The remaining decision's fixed suffix exhausted after 6 labels with maximum covered fraction 0.977358, below 0.98. This attributes the choices to finite-graph connection/coverage structure; it does not prove physical infeasibility.

## Graph and accounting

| scene | states | edges | IK calls | build s | root seed |
|---|---:|---:|---:|---:|---|
| H00 | 741 | 6109 | 982539 | 701.36 | anchor_root (first admitted) |
| H01 | 951 | 5330 | 976450 | 768.53 | anchor_root (first admitted) |
| H02 | 935 | 5329 | 984957 | 795.44 | anchor_root (first admitted) |
| H03 | 919 | 6098 | 963075 | 745.39 | anchor_root (first admitted) |
| H04 | 902 | 5979 | 977941 | 765.45 | anchor_root (first admitted) |
| H05 | 945 | 5391 | 964949 | 781.64 | anchor_root (first admitted) |

All six graphs were `ready`; every one covered all 238 ports and retained 741–951 actual states. Construction used 963,075–984,957 counted IK calls per scene. The large graph NPZ files remain local under the recorded retention paths and hashes.

Append-only accounting reconciled 20 DEV and 56 TRANSFER screen callbacks exactly with method counters. Full validation logs include fallback, novel finalist and cache-hit calls. The pre-fix final tables are retained because an initial selector omitted F fallback from A/P outputs; commit `01aa429` corrected only final selection and the searches were not rerun.

Focused suite: 65 passed. Literal suite exit 1: 219 passed, 1 skipped and 10 FileNotFoundError failures attributable individually to the two recorded historical E06 fixtures. The literal suite is not claimed fully passing.

## Boundaries

F is an internal fixed-library baseline. P is a transparent greedy control. A/P are bounded searches; retained-record or greedy exhaustion is not physical infeasibility or global optimality. Results concern pose transfer on the same hemisphere, not cross-surface generalization or published-planner superiority. Missing graph edges are sampled-construction limitations. Acceptance is refined finite sampling, not a continuous-time or hardware certificate. Collision checking is limited to the pinned MuJoCo model. Source-run B, G1, FM, hardware, non-spherical geometry, external-planner reimplementation and parameter tuning were NOT_RUN.

## Reproduction

Run the ordered commands in `reproduction_commands.txt`. Large graph recovery paths and hashes are recorded in `transfer_graph_manifest.json`; compact accepted witnesses and tables are in this directory.
