# E10 structured anytime global coverage routing

Executed: 2026-09-17T04:33:56.636275+10:00

## Scope and implementation

E10 reused the three frozen E09-R1 multi-state robot graphs without IK, connector, port, or geometry reconstruction. F is the corrected fixed-order reference. A uses bounded atomic-edge search; B changes only the action catalog by adding exact source-run compositions. Both retain a refined-validated F fallback and never call the prospective-repeat bound.

## Six-task results

| scene | k | F | A | B |
|---|---:|---|---|---|
| T30 | 1 | no_accepted_plan; queue_exhausted; 0.01s; exp 93 | new_global_plan; retained_record_budget; 141.26s; exp 36910 | new_global_plan; retained_record_budget; 145.64s; exp 36163 |
| T30 | 2 | no_accepted_plan; queue_exhausted; 0.04s; exp 486 | new_global_plan; retained_record_budget; 192.62s; exp 34652 | new_global_plan; retained_record_budget; 201.84s; exp 37741 |
| T27 | 1 | retained_F; queue_exhausted; 0.02s; exp 196 | retained_F; retained_record_budget; 72.08s; exp 39154 | retained_F; retained_record_budget; 74.08s; exp 39166 |
| T27 | 2 | retained_F; queue_exhausted; 0.05s; exp 196 | retained_F; retained_record_budget; 77.11s; exp 39154 | retained_F; retained_record_budget; 80.32s; exp 39166 |
| T33 | 1 | no_accepted_plan; queue_exhausted; 0.01s; exp 90 | new_global_plan; retained_record_budget; 134.95s; exp 34705 | new_global_plan; retained_record_budget; 177.47s; exp 43629 |
| T33 | 2 | no_accepted_plan; queue_exhausted; 0.13s; exp 1558 | new_global_plan; retained_record_budget; 219.83s; exp 34579 | new_global_plan; retained_record_budget; 240.12s; exp 35360 |

## Independently accepted unique plans

| scene | role | N_on | Jq (ON/OFF/entry) | T1/Q4a miss / repeat | sigma5 min | max position / axis | route |
|---|---|---:|---|---|---:|---|---|
| T30 | global recombination | 1 | 96.949149 (96.876117/0.000000/0.073032) | 0.015433 / 0.046278 | 0.082973 | 5.12e-05 m / 0.0188 deg | cross_family_ON_recombination, 3 cross-port ON |
| T27 | F fallback | 1 | 90.011360 (89.946565/0.000000/0.064795) | 0.018332 / 0.019571 | 0.109752 | 3.91e-05 m / 0.0214 deg | template_or_prefix, 0 cross-port ON |
| T33 | global recombination | 1 | 96.153376 (96.086162/0.000000/0.067213) | 0.018369 / 0.037363 | 0.083249 | 6.15e-05 m / 0.0307 deg | cross_family_ON_recombination, 3 cross-port ON |

## Answers to the experiment questions

1. **Practical global result:** yes on this frozen finite task set. A and B independently passed the refined sampled contract on T30 and T33, where exhaustive F returned no graph goal. T27 retained the already accepted F trajectory. The six budget rows reduce to three unique accepted physical trajectories because every accepted plan used one ON segment.
2. **Sweep proposal isolation:** B did not establish a general runtime benefit. It was slower than A in every paired cell. The accepted witness matched A in five tasks; at T33/k=2 B retained the lower-Jq 96.153376 route while A retained Jq 96.394518. This is a bounded beam-search outcome, not an optimality claim.
3. **Screening/resumption:** the implementation kept Q2 goals, screens, finalists and accepted incumbents separate. All 17 executed Q3 screens passed, so this run did not naturally exercise recovery after a Q3 rejection. Unscreened reservoir candidates were never promoted without final validation.

## Mechanism and cost

- Independently accepted novel global outputs: 8 method-task rows (3 unique witnesses). The accepted T30 and T33 routes are cross-family ON recombinations with no OFF relocation.
- Valid fixed-route fallback retained: 6 method-task rows. Retention is an engineered no-regression property.
- A expansions: 219154; B expansions: 231225. B evaluated 859891 source-run actions and more atomic-equivalent work, so grouping was not computationally free.
- All A/B cells stopped at the 30,000 retained ancestry+Pareto-record safeguard. Live OPEN peaks remained governed by the per-bucket quotas. This status is distinct from the old cumulative-admission shutdown.
- Frozen graph load benchmarks were about 3.1--3.5 s per scene. Inherited graph construction cost remained 703--723 s per scene and is reported in `graph_references.json`; reuse does not make that cost zero.
- Uncached-equivalent selected-witness refined validation cost was about 64--67 s. `global_results.csv` reports core time; `final_validation.csv` and `graph_load_benchmark.csv` keep validation and loading clocks separate.
- The prospective-repeat bound received zero calls, as preregistered.

## Numerical and correctness evidence

All 17 recorded same-sample whole-trace versus edge-summary comparisons had exactly equal pointwise episode counts. Each selected plan passed T0/Q3, T0/Q4, T1/Q4 and T1/Q4a under the unchanged miss/repeat limits and 0.002 stability rule, as well as sampled task, sigma5, joint-limit, collision, activity and Jq checks.

The focused suite passed 55/55 tests. The literal repository suite reported 209 passed, 1 skipped and 10 failed; each failure was a FileNotFoundError from one of the two absent historical E06 fixtures listed in `test_statuses.json`. The suite was therefore dependency-limited, not reported as fully passing.

## Reuse and limitations

A/B are beam-limited anytime searches. Retained-record termination is not finite-graph infeasibility or global optimality. Acceptance is under refined finite sampling, not continuous-time certification. The graph collision claim is limited to the pinned MuJoCo model. Unmodeled workpiece, tool-body extent beyond the XML, environment, cables, dynamics, force, and control performance remain outside scope. These three placements are development cases. FM, hardware, G1, the single-state ablation, IK, and graph construction were NOT_RUN.

## Files

See `global_results.csv`, `final_validation.csv`, `screening_results.csv`, `validation_resolution.csv`, `route_classification.csv`, compact witnesses, whole-surface plots, and `reproduction_commands.txt` in this directory.
