# Manuscript Reframing If Predictive Gate Fails Again

Use this framing now that the strict predictive gate is `FAIL` after the finalized rescue package.

## Positioning Shift

From:
- "Embedding geometry predicts token-level failures beyond confounds."

To:
- "Under strict controls, geometry-only predictors do not outperform confound baselines, while cluster-level geometric interventions produce robust mechanistic changes."

This is a publishable negative-result + mechanism paper, not a predictive-model paper.

For full manuscript strategy and reviewer-risk handling, see:
- `docs/design/hard_pivot_publication_strategy.md`

## Core Claims To Keep

1. **Mechanistic claim (keep, lead with):**
   Cluster-level repair improves local geometry relative to placebo with high semantic retention.
2. **Methodological claim (keep):**
   The strict protocol (forced-token minimal-pair + confound-aligned gating) prevents overclaiming and yields reproducible negative findings.
3. **Predictive claim (replace):**
   Geometry-only predictive signal is not robust beyond confounds in current tested settings.

## Claims To Drop

1. "Geometry metrics are predictive biomarkers for failure."
2. "Repair causally improves downstream behavior."
3. "Findings generalize across all model families."

## Abstract Skeleton (Fail-Case)

1. Present DCTT as a framework for diagnosing and intervening on token-local embedding pathologies.
2. Report strong mechanistic intervention evidence (cluster-level geometry improvement vs placebo).
3. Report strict negative predictive validity across seeds/models under confound-controlled real-label evaluation.
4. Conclude with a calibration message: geometry interventions are mechanistically real, but predictive utility is currently limited.

## Figure/Table Priorities

1. Keep mechanistic figures first (cluster repair + treatment/control geometry changes).
2. Keep predictive sweep aggregate as a negative-result figure with confidence intervals and explicit gate verdict.
3. Keep token examples only as diagnostics, not as proof of predictive utility.
4. Ensure all numbers are regenerated from locked sweep artifacts only.

## Reviewer-Resilient Language

Use:
- "strictly negative predictive result"
- "pre-specified gate criteria"
- "confound-aligned evaluation"
- "mechanistic effect without demonstrated behavioral gain"

Avoid:
- "failed approach" (too broad)
- "geometry is irrelevant" (unsupported overstatement)
- "repair improves performance" (not shown)

## Rapid Manuscript Edits Checklist

1. Update title/subtitle to foreground mechanistic evidence + strict negative prediction.
2. Rewrite RQ1 conclusion as falsification/constraint, not success.
3. Add a "Why this negative result matters" paragraph in Discussion.
4. Move predictive details into a registered gate section with PASS/FAIL criteria.
5. Explicitly separate mechanistic validity from behavioral validity.
