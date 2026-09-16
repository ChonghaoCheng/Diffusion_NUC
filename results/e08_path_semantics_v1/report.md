# E08-R1 path-semantics repair

## Progress and frozen scope

All five bounded stages completed. Exactly eight archived E08 geometries were regenerated once,
hashed, and evaluated without placement repetition or path tuning. The footprint radius and
miss/repeat limits remained 0.008 m, 0.02, and 0.10. Historical files and evaluator defaults were
not changed. Legacy replay warnings above 1e-8: 0.

## L/P/R geometry results

Values are E_miss/E_rep. R shows lower/upper sampled-reference bounds.

| candidate | L | P | R Q2 | geometry decision |
|---|---:|---:|---:|---|
| hemisphere/raster_u_phase_0.00 | 0.128660/0.589181 | 0.136591/0.031522 | [0.000000,0.000000] / [0.028614,0.028614] | accepted_under_reference_checks |
| hemisphere/raster_u_phase_0.25 | 0.127929/0.630148 | 0.136591/0.031699 | [0.000000,0.000000] / [0.028614,0.028614] | accepted_under_reference_checks |
| hemisphere/raster_v_phase_0.00 | 0.016274/1.235340 | 0.027621/0.386086 | [0.000000,0.000000] / [0.576860,0.576860] | unresolved |
| hemisphere/spiral_phase_0.00 | 0.147563/0.528890 | 0.156223/0.011734 | [0.012865,0.012865] / [0.020482,0.020482] | accepted_under_reference_checks |
| saddle/raster_u_phase_0.00 | 0.116651/0.511352 | 0.146340/0.125260 | [0.026564,0.026675] / [0.005929,0.005929] | unresolved |
| saddle/raster_u_phase_0.00_k2 | 0.117529/0.511352 | 0.147218/0.125260 | [0.026564,0.026675] / [0.005488,0.005488] | unresolved |
| saddle/raster_v_phase_0.00 | 0.151239/0.460590 | 0.142941/0.134586 | [0.026564,0.026675] / [0.005929,0.005929] | unresolved |
| saddle/spiral_phase_0.00 | 0.144951/0.471449 | 0.168640/0.012181 | [0.038599,0.038599] / [0.012259,0.012259] | unresolved |

L versus P changes only ordered trajectory reconstruction under the same mesh-distance backend.
L versus R additionally changes surface samples, area measure, and footprint distance, so it is
not a pure reconstruction attribution. Saddle uncertainty is retained with episode-count dynamic
programming. The final decisions also require Q1/Q2 changes no larger than 0.002. Geometry
acceptance, if any, does not imply robot qualification; robot requalification was NOT RUN.

The evidence separates two artifacts. Preserving the path reduced legacy reconstructed lengths
from 6.68--6.84 m to 3.89--4.02 m on saddle and from 14.68--22.55 m to 7.85--12.53 m on hemisphere;
repeat error fell in every candidate, showing that legacy route reconstruction created many extra
episodes. P still reported large miss fractions because it deliberately retained the mesh-distance
backend. With analytical surface distance and area, hemisphere raster-u phases had zero sampled
miss and 0.028614 repeat, while the spiral had 0.012865 miss and 0.020482 repeat; all three were
stable accepted geometries. Hemisphere raster-v retained 0.576860 repeat, supporting a genuine
repeat defect, but its Q1/Q2 change was 0.006384 and therefore its qualification is unresolved.
The saddle candidates retained Q2 miss lower bounds from 0.026564 to 0.038599, supporting genuine
uncovered gaps, but their final changes were 0.006019--0.010651, so all four remain unresolved.
No percentage of failure is assigned to either cause.

## Corrected execution semantics

The historical `complete_lift` field is interpreted only as `target_sequence_solved`. Dense
transition, coverage-contract, and overall execution checks are NOT_RUN. Sampled numeric failures
under the frozen sigma threshold include: hemisphere/T30/legacy6, hemisphere/T30/task5, hemisphere/T33/legacy6, hemisphere/T33/task5.
The full corrected view is in `corrected_ik_status.csv`.

Focused regressions passed 20 tests. The complete repository run passed 182 tests with one existing
skip and 14 warnings; no test failed. The planar regression measured the prescribed sqrt(8) mm
motion and separately exposed the legacy mesh-vertex detour.

## Search and graph readiness

Both search arms now use the same segment-budget-layered ordinary reachability check. Focused tests
separate segment-budget obstruction from a positive repeat-bound prune and compare both arms with
a bounded independent full-count oracle on a cyclic graph. No real graph or real S0/S1 campaign
was run. The old cross-chain edges are OFF reconfigurations, not task-preserving ON connections.
Future graph work still needs explicit node activity and independently checked ON connections.
Endpoint membership is recomputed and checked rather than overwritten. Collision claims remain
limited to modeled MuJoCo self-collision pairs; workpiece, full tool, and environment are absent.

## Interpretation boundary

These results concern sampled membership on the frozen eight paths. They do not certify continuous
coverage, robot safety, global infeasibility, planner superiority, FM benefit, or publication-level
novelty. Runtime and pruning benefit for E08 real-surface planning remain N/A because no case was
robot-qualified in the original run, this repair did not rerun robot qualification, and no such
comparison ran.
