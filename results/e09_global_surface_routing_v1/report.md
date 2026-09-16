# E09 full-surface robot-aware coverage routing

## 1. Progress, contract, and implementation repairs

E09 completed the frozen geometry-bank construction, three placement-specific robot graphs, all
18 scheduled F/G0/G1 search cells, and independent Q1/Q2 whole-plan checks. No saddle campaign,
FM training, hardware execution, dynamics, force, feed-rate, or dose experiment ran.

The immutable bank contains 238 physical ports, 470 directed source macro-arcs, and 1,120
deterministic nearby connector proposals (geometry hash
`01050368f57eca9a904a04106dc705aec89cab65248c508462ad232328976047`). Every E09 IK call uses
`backend="task5"`; a spy regression covers nested continuation. Historical defaults remain
`legacy6`. The pinned XML exposes a six-joint model, `attachment_site` on `wrist_3_link`, local TCP
offset `[0, 0.1, 0]` m, tool-axis index 2 and sign +1. E08's sampled terminal-q6 null-direction
audit justifies avoiding artificial roll bins; all actual unwrapped q values remain in witnesses.

The actual-execution checker uses ordered FK centerlines and exact spherical footprint distance.
OFF samples contribute joint cost and activity transitions, while only ON samples enter task
residual and sigma5 extrema. The cyclic unconstrained-reachability bug was repaired: the
unconstrained diagnostic now visits nodes, while finite segment budgets retain layered states.

Pre-comparison implementation attempts were invalidated and retained explicitly: short two-sample
entry handling, OFF residual filtering, graph deserialization, process isolation, and fixed-route
automaton start advancement. No invalidated output is mixed into the tables below.

## 2. Frozen robot graphs

| placement | states | verified edges | cross-port ON | OFF reconfigurations | IK calls | build s | graph hash prefix |
|---|---:|---:|---:|---:|---:|---:|---|
| T30 / P_low | 241 | 1,429 | 0 | 964 | 103,607 | 119.13 | `e8f97f3f` |
| T27 / P_mid | 238 | 1,423 | 0 | 952 | 102,646 | 117.11 | `e641af51` |
| T33 / P_high | 242 | 1,431 | 0 | 968 | 103,732 | 118.07 | `62c728dc` |

All three graphs are `recombination_limited`: independently checked source fragments and explicit
OFF relocations exist, but none of the frozen cross-port ON proposals produced an accepted
task-preserving robot edge. This is a construction/representation result, not physical
non-connectivity. Endpoint membership mismatches were rejected rather than overwritten.

## 3. Six-task result table

All searches ended by finite queue exhaustion. Values below are independently recomputed Q2
miss/repeat for plans; `unresolved` means Q1/Q2 changed by more than the frozen 0.002 rule.

| task | F | G0 | G1 |
|---|---|---|---|
| T30, k=1 | no graph plan | no graph plan | no graph plan |
| T30, k=2 | no graph plan | unresolved, 0.011565 / 0.081400, Jq 99.981 | unresolved, same plan |
| T27, k=1 | unresolved, 0.016922 / 0.026743, Jq 90.068 | unresolved, same plan | unresolved, same plan |
| T27, k=2 | unresolved, same one-segment plan | unresolved, same plan | unresolved, same plan |
| T33, k=1 | no graph plan | no graph plan | no graph plan |
| T33, k=2 | no graph plan | unresolved, 0.011565 / 0.081420, Jq 100.815 | unresolved, same plan |

The T27 witness is a 77-arc prefix of raster-u phase 0 and uses one ON segment. T30/T33 use the
same geometry mechanism with placement-specific q: 74 forward raster-u arcs, four reverse arcs,
one verified OFF relocation from port 70 to port 79, and four further reverse arcs. Their Jq
decompositions are 99.604 + 0.377 + 0 (ON/OFF/entry) for T30 and 100.450 + 0.365 + 0 for T33.
They are reconfiguration-plus-direction-reversal routes within one source family; no cross-family
ON connection was available.

All ten graph plans passed the denser sampled kinematic checks: maximum position error was at most
`6.21e-5` m, maximum axis error at most `0.00631` degrees, minimum sigma5 at least `0.08325`,
minimum normalized joint margin at least `0.08088`, and modeled collisions were absent. None is an
accepted execution because the Q1/Q2 changes were 0.002628 for T27 and about 0.00797--0.00799 for
T30/T33. No further resolution was authorized.

## 4. Search comparison and completion bound

| task | G0 expanded / s | G1 expanded / s | finite repeat prunes | outcome |
|---|---:|---:|---:|---|
| T30, k=1 | 1 / 0.005 | 1 / 0.005 | 0 | shared segment-reachability exhaustion |
| T30, k=2 | 2,325 / 5.097 | 1,472 / 29.142 | 72 | fewer labels, slower total |
| T27, k=1 | 487 / 0.194 | 338 / 5.922 | 70 | fewer labels, slower total |
| T27, k=2 | 487 / 0.774 | 474 / 36.666 | 42 | slightly fewer labels, slower total |
| T33, k=1 | 1 / 0.006 | 1 / 0.006 | 0 | shared segment-reachability exhaustion |
| T33, k=2 | 2,343 / 5.207 | 1,872 / 35.922 | 71 | fewer labels, slower total |

G0 and G1 returned identical feasibility and graph-optimal objectives in every exhausted cell.
G1 made 1,627 bound calls, recorded 255 prospective-repeat prunes, and reduced expansions in every
cell where the bound was active. Bound computation took 98.68 s in total, so the reduction did not
produce net search savings. Graph construction (~117--119 s per placement) also dominates G0
search and remains material in cold totals.

A natural T30/k2 mechanism prefix had R=0.093340, remained ordinary-reachable within the segment
budget, and had future repeat lower bound 0.015870; total 0.109209 exceeded the 0.10 budget. The
saved `mechanism_example.json` contains its exact edge prefix. No edge was removed to create it.

## 5. Answers to Q1--Q3

**Q1:** No independently accepted full-surface robot execution was established. Ten plans were
feasible on their frozen Q2 graphs and passed denser sampled robot checks, but every plan was
numerically unresolved under the preregistered Q1/Q2 stability rule.

**Q2:** The finite graph shows a routing-capability signal: at T30/T33 with k=2, G found a
direction-reversal plus OFF-reconfiguration plan where the exhausted fixed-template automaton did
not. That signal is not a validated execution advantage because those plans are resolution
unresolved, and the graph contains no verified cross-port ON recombination. At T27, F/G0/G1 all
returned the same fixed-template prefix, so no global gain was measured there.

**Q3:** The prospective repeat bound supplied independent finite positive information and reduced
expanded labels with 255 genuine prunes, including the saved real-scene mechanism. It increased
net search time substantially in every informative cell. Under this implementation and bank the
result is **information useful but too expensive**, not a speedup.

## 6. Tests and interpretation boundary

Focused E09/search/evaluator tests pass, including explicit task5 forwarding, exact sphere seam and
pole membership, OFF/ON semantics, endpoint composition, positive repeat obstruction, shared
bottleneck, and cyclic unconstrained reachability. The literal full-tree run has ten failures
caused only by two absent historical gitignored E06 inputs (`saddle_T17.npz` and
`nuc_robot_skeleton_coupling_v1/config.json`), recorded in `test_summary.txt`.

These are sampled finite-graph results, not continuous-time certificates. The collision claim is
limited to the pinned MuJoCo contacts; unmodeled workpiece, tool-body geometry beyond the XML,
environment, and cables are not guaranteed absent. Jq is neither energy nor execution time.
Failed numerical connection attempts do not establish physical infeasibility. F is an internal
fixed-library baseline. No planner-superiority, general-surface, RSS novelty, or FM claim is made.
