# Skill Exercise: Experiment Run Evaluator (Lite)

## Goal
Build a skill that helps an agent choose a production candidate from offline experiment runs.

This is intentionally small and local-only (no external APIs), but still agentic:
- scripts do deterministic calculations,
- the skill decides **what to optimize for** based on user intent and **what to do when tradeoffs/conflicts appear**.

## Dataset
All input files are in `data/`:
- `experiments.csv`: candidate runs and metrics
- `baseline.csv`: current production baseline
- `policy.json`: hard constraints + soft target + strategy weights
- `scenario_requests.md`: sample user prompts

### Metrics
- `accuracy` and `factuality`: higher is better
- `latency_p95_ms` and `cost_per_1k_usd`: lower is better

## Core Requirements
Implement a skill that can:
1. Identify feasible runs under hard constraints.
2. Compare runs to baseline and quantify deltas.
3. Recommend one run according to user intent:
   - `latency_first`
   - `quality_first`
   - `balanced`
4. If no run is feasible, return no-ship and explain closest alternatives.

## Why This Is A Skill (Not Just Tools)
This workflow is not a fixed one-size-fits-all chain. The agent must make choices between script calls:
- infer strategy from user text (mobile = latency-first, high-trust = quality-first, default = balanced)
- decide which candidate runs to inspect in detail
- decide whether to prioritize soft target (`latency <= 1100ms`) once hard constraints pass
- produce a decision narrative with evidence and caveats

Scripts provide primitives; the skill provides orchestration policy.

## Implementation Output
Your implementation should produce artifacts in `artifacts/`:
- `shortlist.csv`
- `summary.json`
- `eval_<run_id>.json` (for inspected run IDs)
- `final_recommendation.md`
