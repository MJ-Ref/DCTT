# Hard Pivot Publication Strategy

This document commits the project to a non-ad hoc publication strategy after the final strict predictive package failed.

## 1) Decision Basis

Evidence source:
- `outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.md`
- Strict sweeps: `2026-02-24_06-43-58`, `2026-02-24_07-31-10`, `2026-02-24_07-34-45`

Core facts:
- 20 strict runs across 4 models
- Geometry-minus-baseline pooled mean: `-0.128153` (95% CI `[-0.168367, -0.087938]`)
- Full-minus-baseline pooled mean: `-0.011735` (95% CI `[-0.023119, -0.000351]`)
- Strict predictive gate verdict: `FAIL`

Conclusion:
- Positive predictive claim is falsified for the current endpoint/protocol.
- Mechanistic geometry intervention claim remains valid and should lead the paper.

## 2) Primary Paper Angle (Recommended)

## "Mechanistic positive + predictive negative"

Claim:
- Cluster-level intervention changes geometry in the intended direction with semantic preservation.
- Under strict confound controls, geometry features do not improve predictive performance over baseline confounds.

Why this is strong:
- It is honest, reproducible, and difficult to dismiss as underpowered noise.
- It contributes a practical anti-overclaim protocol for embedding-geometry work.
- It resolves apparent contradiction by separating mechanistic validity from predictive utility.

## 3) Secondary Angle (Keep in Support)

## "Evaluation protocol contribution"

Position DCTT not only as a metric/intervention toolkit but as a strict evaluation design:
- forced-token minimal-pair tests
- confound alignment
- explicit PASS/FAIL gate criteria
- protocol lock artifacts

This raises the paper above "negative result only" and provides reusable methodology.

## 4) Third Angle (Exploratory, not confirmatory)

## "Boundary conditions and failure modes"

Use limited exploratory analyses to generate hypotheses:
- model-family heterogeneity in effect size magnitude
- strata where geometry signal may be less negative
- interactions between geometry and confound features

Rule:
- Clearly label as exploratory and non-claim-bearing.

## 5) Manuscript Architecture

1. Introduction:
- State dual objective: mechanistic intervention and predictive utility test.
- Pre-commit that predictive utility is judged by strict gate criteria.

2. Methods:
- Define stress tests, confounds, and gate thresholds before results.
- Separate confirmatory and exploratory analyses by design.

3. Results A (Mechanistic):
- Cluster repair improvements vs placebo.
- Semantic preservation and movement constraints.

4. Results B (Predictive, confirmatory):
- Per-model and pooled strict gate results.
- Emphasize directionality consistency and CI behavior.

5. Results C (Exploratory):
- Optional strata/interactions only; no headline claims.

6. Discussion:
- Explain why mechanistic control does not imply predictive signal.
- Provide implications for geometry-based safety/diagnostic claims.

## 6) Reviewer Risk Register and Responses

Risk 1: "This is just a null result."
- Response: It is a directional negative result with pooled CI excluding zero for geometry-minus-baseline.

Risk 2: "Maybe underpowered."
- Response: 20 strict runs across 4 models, 5 seeds/model, consistent sign.

Risk 3: "Why trust the labels?"
- Response: Forced-token minimal-pair setup, explicit no-proxy confound policy, locked artifacts.

Risk 4: "Mechanistic and predictive claims conflict."
- Response: They test different properties; paper explicitly separates them.

## 7) Non-Negotiable Claim Boundaries

Do claim:
- mechanistic geometry intervention effect
- strict negative predictive result for current endpoint/protocol
- protocol-level contribution to anti-overclaim practice

Do not claim:
- geometry predicts failures generally
- repair improves downstream behavior
- universal model-family generalization

## 8) Execution Checklist (Paper Readiness)

1. Regenerate all main tables/figures from locked artifacts only.
2. Insert hard-pivot pooled table/figure into Results B.
3. Rewrite abstract and conclusion to reflect final claim set.
4. Add a short "registered gate criteria" subsection.
5. Tag exploratory analyses explicitly as hypothesis-generating.
