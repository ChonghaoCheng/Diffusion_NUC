# E09-R1 synchronized continuous routing repair

## Progress

All three repaired graphs, 24 cold-start method cells, and the fixed refined validation schedule completed. The graph/search results use `bb05ea9252a69687e435b3ca4458c8d5f21c13bd`; the subsequently versioned validation adapter is recorded in the manifest. Two unique new q witnesses were validated; one passed and one failed the unchanged coverage contract.

## Six-task four-method table

Cells show refined validation or absence of a graph plan, followed by search termination.

| scene | k | F | G0 | G1 | S |
|---|---:|---|---|---|---|
| T30 | 1 | NO_GRAPH_PLAN / queue_exhausted | NO_GRAPH_PLAN / memory_limit_projected | coverage_contract_failed_under_refined_checks / wall_time | NO_GRAPH_PLAN / memory_limit_projected |
| T30 | 2 | NO_GRAPH_PLAN / queue_exhausted | NO_GRAPH_PLAN / memory_limit_projected | NO_GRAPH_PLAN / wall_time | NO_GRAPH_PLAN / memory_limit_projected |
| T27 | 1 | accepted_under_E09_R1_refined_sampled_checks / queue_exhausted | NO_GRAPH_PLAN / memory_limit_projected | NO_GRAPH_PLAN / wall_time | NO_GRAPH_PLAN / memory_limit_projected |
| T27 | 2 | accepted_under_E09_R1_refined_sampled_checks / queue_exhausted | NO_GRAPH_PLAN / memory_limit_projected | NO_GRAPH_PLAN / wall_time | NO_GRAPH_PLAN / memory_limit_projected |
| T33 | 1 | NO_GRAPH_PLAN / queue_exhausted | NO_GRAPH_PLAN / memory_limit_projected | NO_GRAPH_PLAN / wall_time | NO_GRAPH_PLAN / memory_limit_projected |
| T33 | 2 | NO_GRAPH_PLAN / queue_exhausted | NO_GRAPH_PLAN / memory_limit_projected | NO_GRAPH_PLAN / wall_time | NO_GRAPH_PLAN / memory_limit_projected |

## Synchronized construction and graph capability

The former caller could pair a longer endpoint-targeted q trace with a shorter target trace and silently omit the terminal interval. `SynchronizedMotionTrace` now carries q, path parameter, target position/axis and activity on the same parameter. Admission checks exact stored endpoint q, endpoint activity and recomputed membership; densification rejects unequal shapes.

- T30: 865 states, 5870 edges, 237 multi-state ports, 1529 accepted cross-port ON edges (1118 root-reachable), 3511 OFF edges.
- T27: 897 states, 6091 edges, 238 multi-state ports, 1573 accepted cross-port ON edges (1120 root-reachable), 3732 OFF edges.
- T33: 724 states, 6056 edges, 238 multi-state ports, 1622 accepted cross-port ON edges (1120 root-reachable), 3491 OFF edges.

All 238 ports had multiple effective states in T27/T33 and 237 did in T30. The selected routes nevertheless used canonical rank 0 only, so availability was demonstrated but route-level benefit from extra states was not.

## Returned whole-plan witnesses

| scene | k/method | route | Jq (on/off/entry) | refined result | fine miss / repeat | motion extrema |
|---|---|---|---|---|---|---|
| T30 | 1/G1 | cross_family_ON_recombination; 88 edges, 9 cross ON | 95.837506 (92.197635/0.000000/3.639871) | coverage_contract_failed_under_refined_checks | 0.019344 / 0.092546 | sigma 0.082973; pos 5.18e-05 m; axis 0.0188 deg |
| T27 | 1/F | template_or_prefix; 77 edges, 0 cross ON | 90.011360 (89.946565/0.000000/0.064795) | accepted_under_E09_R1_refined_sampled_checks | 0.018332 / 0.019571 | sigma 0.109752; pos 3.91e-05 m; axis 0.0214 deg |
| T27 | 2/F | template_or_prefix; 77 edges, 0 cross ON | 90.011360 (89.946565/0.000000/0.064795) | accepted_under_E09_R1_refined_sampled_checks | 0.018332 / 0.019571 | sigma 0.109752; pos 3.91e-05 m; axis 0.0214 deg |

T27 k=1 and k=2 reference the same content-hashed witness: a 77-edge spiral template prefix with one ON segment. T30 k=1 G1 returned an 88-edge, 9-cross-port, cross-family ON route. Its motion checks passed, but T0/Q3 miss was 0.020274; because Q3-Q4 was stable within 0.002, the route is a coverage-contract failure, not an accepted result.

## Scientific questions

- **Q1 — valid global routing:** A complete refined-sampled plan was accepted for T27 at both budgets through F. No independently accepted globally recombined G0/G1 route was produced. T30's globally recombined route failed coverage; T33 returned no graph plan within the budgets.
- **Q2 — global route choice:** No measured G0 improvement over F. All G0 cells stopped at the 30,000-resident-label safeguard before finding a plan, while F exhaustively found the accepted T27 spiral prefix.
- **Q3 — state multiplicity:** The construction retained multiple effective q states, but all returned paths used rank 0. G0 and S were both budget-limited without a plan, so this run did not establish a finite-graph benefit from multi-state retention.
- **Q4 — bound utility:** G1 made 7772 prospective-repeat prunes, but spent 1747.7 of 1800.7 search seconds (97.1%) in the bound. It was not a net computational saving. Its sole graph solution appeared at 143.1 s and failed refined coverage.

The natural obstruction in `mechanism_example.json` has past repeat 0.099946 and future lower bound 0.013447, so total 0.113392 exceeds the 0.10 budget while ordinary reachability remains available.

## Validation and tests

Same-sample whole-trace episode counts equal composed edge summaries pointwise for both unique new witnesses (zero differing cells). T27's Q4-to-Q4a changes were below 0.002. The three accessible parent-E09 unique witnesses were retrospectively checked without replanning; all remain `numerically_unresolved` because repeat changed by more than 0.002 under Q4a. Their historical statuses remain unchanged.

Focused suite: 45 passed. Literal repository suite: 196 passed, 1 skipped, 10 failed. All 10 failures are individually recorded in `tests.txt` and come from two unavailable historical E06 inputs; the suite is dependency-limited, not reported as passing.

## Limits

F/G0 compares fixed-order and global route freedom on one repaired finite graph; S/G0 is the induced state-retention ablation; G0/G1 isolates the existing repeat bound. Most global cells ended at a resident-label or time budget, so they provide no graph infeasibility certificate. Missing numerical connections are not physical infeasibility. The accepted label is a refined sampled check, not a continuous-time certificate.

No hardware, saddle campaign, FM training, path-family tuning, threshold relaxation, Q5, or bound optimization ran. Collision statements cover only the pinned MuJoCo model; workpiece/tool-body geometry beyond it, environment and cables remain unmodeled.
