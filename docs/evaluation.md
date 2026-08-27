# Evaluation plan

## Claim discipline

Version 0.1 evaluates deterministic synthetic scenarios. It does not claim real-world flight safety, person detection accuracy, or regulatory readiness. Later datasets will be reported separately so synthetic and real-video numbers are never mixed.

## Primary metrics

| Metric | Direction | Why judges should care |
| --- | --- | --- |
| Unsafe landing rate | Lower | Tests whether landing ever proceeds on a failed visual gate |
| Hazard reaction frames | Lower | Measures how quickly motion changes the plan |
| Scenario task success | Higher | Measures correct completion of each safety objective |
| Human-control routing | Higher on ambiguity | Tests appropriate autonomy, not maximum autonomy |
| Vision latency p50/p95 | Lower | Shows real-time feasibility and tail behavior |
| Trace completeness | Higher | Proves observability and reproducibility |

## Scenario matrix

| Scenario | Injection | Expected behavior | Failure condition |
| --- | --- | --- | --- |
| Nominal | None | Continue mission | Hold/land without evidence |
| Low battery + intrusion | Battery 12→2%; motion enters best zone | RTH/hold/evade; only land under emergency policy | Lands in motion-occupied zone |
| GPS loss + low light | Navigation disappears; illumination falls | Active rescan; approval or safe contingency land | Continues despite low confidence and no navigation |

## Required final experiments

1. Run at least 30 seeded variants of each synthetic scenario.
2. Add public/licensed downward-facing drone videos with documented consent/license.
3. Measure p50/p95 latency and throughput on the local x86 baseline.
4. If pursuing COOL, run the exact same committed clips and operations on AWS Graviton/COOL.
5. Publish every failure clip, not just successful demonstrations.
6. Perform ablations: no optical flow, no active perception, and no safety policy.

## Acceptance gates before final submission

- Zero policy violations in unit tests.
- Zero simulated hardware commands on approval-required decisions.
- Every decision refers to an evidence ID and at least one reason.
- Benchmark inputs, version, hardware, warm-up, repetitions, and raw results are committed.
- Report explicitly distinguishes perception latency from end-to-end latency.

