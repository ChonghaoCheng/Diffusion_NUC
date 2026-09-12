# E06-G G0 summary

Decision: **NO-GO**

Verified executions: **107/192**. G1 was not authorized.

| Scene | Eq. | Eq. verified | S_q (eq.) | All verified | S_q (all) | Delta_global | Length-matched gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| saddle P_low | 1 | 1 | n/a | 32 | 28.285% | 0.000% | 0.000% |
| saddle P_mid | 1 | 1 | n/a | 31 | 49.962% | 0.000% | 0.000% |
| saddle P_high | 1 | 1 | n/a | 31 | 38.928% | 0.000% | 0.000% |
| hemisphere P_low | 8 | 0 | n/a | 2 | 3.010% | n/a | n/a |
| hemisphere P_mid | 8 | 0 | n/a | 11 | 8.420% | n/a | n/a |
| hemisphere P_high | 8 | 0 | n/a | 0 | n/a | n/a | n/a |

Cost GO: `False`; feasibility GO: `False`; G1 authorized: `False`.

The negative gate is admission-limited: saddle has a singleton coverage-equivalent set, while hemisphere has eight equivalent layouts but no verified witness among them. Non-equivalent layouts exhibit robot-cost and liftability variation, but cannot support the registered claim.

This is a frozen finite-layout and finite-continuation capacity test, not a global optimum or C-space topology result.
