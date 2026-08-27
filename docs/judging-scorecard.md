# Judge-facing evidence scorecard

This is a build checklist, not a self-awarded score. A criterion is complete only when the linked artifact exists and can be reproduced.

## Overall rubric - 100 points

| Criterion | Weight | Evidence already present | Evidence required before submission |
| --- | ---: | --- | --- |
| Technical execution | 30 | OpenCV multi-cue pipeline, closed loop, policy tests, three scenarios | Motion compensation, temporal tracking, p50/p95, real-video failures |
| Innovation | 20 | Uncertainty-triggered second OpenCV call; safety policy owns action | Counterfactual demo and ablation evidence |
| Real-world impact | 20 | Emergency landing problem, target users, responsible boundaries | Operator feedback or expert review; licensed real clips |
| User experience | 10 | Annotated MP4 and evidence dashboard | Approval UI, accessible labels, clear replay navigation |
| Documentation | 10 | README, architecture, evaluation, proposal, daily lesson | Final technical report and clean-machine proof |
| Cloud / reproducibility | 10 | SAM, IAM API, encrypted lifecycle storage, CI | Deployed endpoint, alarms, measured availability/cost |

## Agentic Vision special-award evidence

| Rubric item | Required proof |
| --- | --- |
| Substantive OpenCV + agent integration (30%) | Trace showing confidence triggers a second CLAHE/OpenCV pass |
| Orchestration and appropriate autonomy (25%) | State diagram plus examples of auto-action, hold, and human gate |
| Task effectiveness and evaluation (20%) | Scenario success, unsafe-landing rate, reaction frames, ablations |
| Failure handling, observability, security, human control (15%) | Cloud-offline run, signed endpoint, blocked approval, failure clips |
| UX, docs, demo (10%) | Under-five-minute counterfactual demo and replay dashboard |

## Demo's decisive 45 seconds

Use the same battery and altitude twice. In run A the best zone is visually clear and the planner selects it. In run B an intrusion enters that zone; OpenCV motion/clearance changes, the zone is rejected, and the next action becomes approval or re-plan. Show both evidence IDs and policy reasons side by side. This directly proves the competition's defining requirement.

## Claims that must not appear until proven

- "Detects people" - current system detects moving and appearance-anomalous regions.
- "Real-time" - the current baseline is 1.63-2.82 FPS in the measured container runs.
- "Safe for autonomous flight" - the adapter is simulation-only and the stack is not flight-certified.
- "COOL is faster" - no claim before verified COOL execution and a reproducible comparison.
- "Works in all lighting/weather" - low light is one controlled synthetic perturbation, not broad robustness evidence.
