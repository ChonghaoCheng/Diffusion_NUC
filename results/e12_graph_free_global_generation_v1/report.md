# E12 graph-free global generation pilot

Executed through report at 2026-09-18T13:02:00.942930+10:00.

## Protocol and completed stages

All registered stages ran: preparation, E11 diagnostic repair, encoding, oracle re-lift, TRAIN/VALIDATION teacher collection, one-seed categorical/REG/FM training, validation pilot, sealed graph-free evaluation, P_lazy, P_graph, refined validation, plots, and reporting. The user-requested expansion to 40 TRAIN poses/15,000 updates was cancelled before any added graph or label completed; no expansion artifact entered a model or result. The primary contract remained 20 TRAIN, 4 VALIDATION, 8 SEALED_TEST and 5,000 updates per model.

Runtime model inputs contain the geometry-only SCAN/VIA/END library, the task transform, checked q0, robot model, and physical contract. They contain no robot graph, per-port q, stored successful connection, branch label, or future teacher q. The focused dependency test deliberately exposes forbidden artifacts and verifies that the candidate lifter has no graph-loader or teacher-q dependency.

## Interface and corpus

Of 15 indexed E11 accepted witness hashes, 12 were eligible single-ON programs. All 12/12 preserved the program interface and passed independent q0-only re-lift plus refined sampled checks. The three two-ON witnesses were indexed but remained outside this k=1 pilot. This establishes interface capability, not generated-plan success.

Teacher collection attempted all 24 TRAIN/VALIDATION tasks, found 27 qualified labels from 14 tasks (13 TRAIN tasks and one VALIDATION task), and retained every failed task. Offline teacher graph construction consumed 18282.9 CPU-s in aggregate. Training used one root seed, independent deterministic per-model streams, two RTX A5500 GPUs where concurrent, 5,000 updates per model, and 34.5 s critical-path / 96.8 GPU-s summed training time.

## Validation pilot

| task | RETRIEVE | REG | FM |
|---|---|---|---|
| VA00 | accepted_under_E12_refined_sampled_checks | no_accepted_candidate | no_accepted_candidate |
| VA01 | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate |
| VA02 | accepted_under_E12_refined_sampled_checks | no_accepted_candidate | no_accepted_candidate |
| VA03 | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate |

Accepted cells were RETRIEVE 2/4, REG 0/4, and FM 0/4. These diagnostics did not alter the sealed policy.

## Sealed test

| task | RETRIEVE | REG | FM | P_lazy | P_graph |
|---|---|---|---|---|---|
| TE00 | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate |
| TE01 | accepted (Jq 84.208) | accepted (Jq 83.893) | accepted (Jq 83.538) | accepted (Jq 83.547) | accepted (Jq 83.548) |
| TE02 | accepted (Jq 90.148) | no_accepted_candidate | no_accepted_candidate | accepted (Jq 93.503) | accepted (Jq 93.503) |
| TE03 | accepted (Jq 91.473) | no_accepted_candidate | no_accepted_candidate | accepted (Jq 91.473) | accepted (Jq 91.473) |
| TE04 | accepted (Jq 90.487) | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate |
| TE05 | accepted (Jq 88.182) | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate |
| TE06 | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate | no_accepted_candidate |
| TE07 | accepted (Jq 88.748) | no_accepted_candidate | no_accepted_candidate | accepted (Jq 88.124) | accepted (Jq 88.124) |

Accepted cells: RETRIEVE 6/8, REG 1/8, FM 1/8, P_lazy 4/8, P_graph 4/8. These are paired method rows over eight tasks, not independent trajectory samples.

All accepted routes passed actual-FK task residual, sigma5, joint-limit, modeled-collision, activity, exact same-sample episode-composition, T0/Q3, T0/Q4, T1/Q4 and Q4a checks. `accepted_route_metrics.csv` gives the exact extrema and coverage values. The four selected P_graph witnesses were independently replayed after result locking; all four again passed, with their complete composition and resolution records in `p_graph_validation_details.json`. The accepted witnesses had maximum reported resolution change below 0.00140. One P_lazy TE04 candidate was numerically unresolved (0.00298 change); REG/FM each produced a TE07 candidate with full coverage but repeat error above 1.10 and unstable resolution. None was accepted.

## Failure attribution

Across fixed slots, RETRIEVE had 17 IK-call-limit failures and six accepts; REG and FM each had 55 IK-call-limit failures, one accept and one numerically unresolved route. P_lazy produced four accepts and one numerically unresolved route; tasks without a complete P_lazy candidate remain explicit. `candidate_failure_taxonomy.csv` retains NOT_RUN slots separately. This separates proposal availability and candidate lifting from refined validation.

## Cost

P_graph built eight ready graphs using 7,693,304 IK calls. Graph construction alone cost 5827.3 s and full cold cells cost 6589.9 s (823.7 s/task). The eight P_lazy query phases cost 597.0 s and their candidate validation cells cost 349.4 s; including shared root checks gives 947.3 s (118.4 s/task). Thus P_lazy matched P_graph on the same four accepted tasks while using about 6.96x less measured cold time in this run. This ratio is descriptive for these tasks and implementations.

RETRIEVE evaluation cells summed 1041.6 s; REG 1545.7 s; FM 2511.2 s. Per-task candidate-generation/model-loading time was not separately captured by the freeze stage and is marked `NOT_CAPTURED` in `cold_cost_breakdown.csv`; offline teacher and training cost is reported above rather than assigned zero.

## Answers

- **Q1 — graph-free encoding and execution:** supported within this finite vocabulary. The oracle diagnostic passed 12/12, and graph-free RETRIEVE and P_lazy produced accepted routes on 6/8 and 4/8 sealed tasks without test robot graphs. This does not establish template-free or unseen-surface generation.
- **Q2 — FM:** FM generated one accepted sealed route, so the end-to-end path can work. FM matched REG at 1/8 and was worse than RETRIEVE (6/8) and P_lazy (4/8); stochastic flow modeling showed no incremental value in this one-seed pilot.
- **Q3 — controls:** training-only retrieval was strongest by accepted-task count. REG did not match retrieval. P_lazy exactly matched P_graph’s four-task accepted set and used much less measured cold time. The observed savings can therefore be explained by route reuse or on-demand kinematics without crediting FM.
- **Q4 — cold start:** full graph preprocessing dominated P_graph cost. Removing it yielded a measured P_lazy cold-time reduction, while RETRIEVE/REG/FM still depend on substantial offline teacher generation and training.

## Tests and boundaries

The focused E12 suite passed 26 tests with one warning. The literal repository suite had 245 passed, one skipped and ten failures, all `FileNotFoundError` cases from the two explicitly absent historical E06 fixtures listed in `tests.txt`; it is dependency-limited, not fully passing.

This is one training seed and pose transfer around three anchors on one hemisphere. It does not test exogenous coverage-history conditioning, RFM, non-spherical surfaces, force/contact control, hardware, or a faithful published planner. Acceptance is under a finite refined sampled checker, not a continuous-time certificate. Collision claims cover only the pinned MuJoCo model. Missing sampled IK connections and budget failures are not physical-infeasibility proofs.
