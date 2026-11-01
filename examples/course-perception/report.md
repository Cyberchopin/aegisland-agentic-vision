# Offline course perception report

Samples: 26. CONTROL OUTPUT=DISABLED; COMMAND EXECUTED=FALSE.

Heuristic scores are not calibrated probabilities. No real-flight validation.

| Input | Line / score | Opening / depth-supported | Depth valid | Recovery | Clearance at probe (m) | Pad / geometry suitable | Landing rejection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic:line | True / 1.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:maze | True / 1.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:opening | False / 0.000 | 1 / True | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:dim_recoverable | True / 0.610 | 0 / False | True | True | 2.1785714285714284 | False / False | no_outer_ring, low_light_raw_evidence |
| synthetic:dim_line | False / 0.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring, low_light_raw_evidence |
| synthetic:blackout | False / 0.000 | 0 / False | False | False | unknown | False / False | no_outer_ring, low_light_raw_evidence |
| synthetic:chicane | False / 0.000 | 1 / True | True | False | 1.0625 | False / False | no_outer_ring |
| synthetic:narrow_gap | False / 0.000 | 1 / False | True | False | 0.2857142857142857 | False / False | no_outer_ring |
| synthetic:blocked_opening | False / 0.000 | 1 / False | True | False | 0.0 | False / False | no_outer_ring |
| synthetic:unknown_opening | False / 0.000 | 1 / False | True | False | unknown | False / False | no_outer_ring |
| synthetic:pad | False / 0.000 | 0 / False | True | False | unknown | True / True |  |
| synthetic:intruded_pad | False / 0.000 | 0 / False | True | False | unknown | True / False | landing_footprint_too_small, landing_surface_not_planar |
| synthetic:solid_disk | False / 0.000 | 0 / False | True | False | unknown | False / False | no_outer_ring |
| synthetic:ellipse | False / 0.000 | 0 / False | True | False | unknown | False / False | no_outer_ring |
| synthetic:parallel_lines | False / 1.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:branch | False / 0.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:broken_line | False / 0.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:blank | False / 0.000 | 0 / False | True | False | 2.1785714285714284 | False / False | no_outer_ring |
| synthetic:tilted_pad | False / 0.000 | 0 / False | True | False | unknown | True / False | landing_footprint_too_small, landing_surface_tilted |
| synthetic:clipped_pad | True / 0.615 | 0 / False | True | False | unknown | False / False | no_outer_ring |
| synthetic:dark_pad | False / 0.000 | 0 / False | True | False | unknown | True / True |  |
| synthetic:two_pads | False / 0.000 | 0 / False | True | False | unknown | True / False | competing_outer_rings |
| synthetic:stale_depth | True / 1.000 | 0 / False | False | False | unknown | False / False | no_outer_ring |
| synthetic:unsynchronized_depth | True / 1.000 | 0 / False | False | False | unknown | False / False | no_outer_ring |
| synthetic:missing_depth | True / 1.000 | 0 / False | False | False | unknown | False / False | no_outer_ring |
| synthetic:blind_zone | True / 1.000 | 0 / False | False | False | unknown | False / False | no_outer_ring |

CPU observe latency: p50=5.579 ms, p95=11.974 ms, max=24.102 ms. One pass, cold first sample included, no disk I/O. Not an Orin benchmark.

- synthetic results are not field detection accuracy
- no hardware, Orin, flight or full-route validation
- no semantic exit/pad identity or temporal obstacle tracking
- clearance is projected at the configured camera-z probe plane; not a physical gate width or swept 3-D vehicle volume
- landing suitability covers observed geometry/depth only; motion, gravity alignment and full 3-D vehicle footprint remain unassessed
- window and return stretch goals are not implemented
