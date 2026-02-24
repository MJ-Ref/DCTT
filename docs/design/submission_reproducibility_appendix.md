# Submission Reproducibility Appendix

This appendix defines the claim-bearing artifact chain for the hard-pivot manuscript.

## A) Claim Scope (Final)

Allowed claims:
1. Cluster-level intervention improves local geometry relative to placebo with semantic preservation.
2. Under strict confound controls, predictive geometry signal is negative for the current endpoint/protocol.
3. The protocol contribution is an anti-overclaim evaluation framework (forced-token minimal pairs, strict confound alignment, explicit gate criteria).

Disallowed claims:
1. Geometry predicts failures generally.
2. Repair improves downstream behavior.
3. Universal model-family generalization.

## B) Registered Gate Criteria (Confirmatory)

Source of truth:
- `outputs/sweeps/predictive_validity/HARD_PIVOT_REPORT.json`
- `configs/paper/submission_package_lock.yaml`

Expected confirmatory values:
- strict gate verdict: `FAIL`
- total runs: `20`
- model count: `4`
- pooled geometry-minus-baseline mean: `-0.12815261735030198`
- pooled full-minus-baseline mean: `-0.011734773768573958`
- pooled geometry CI: `[-0.16836703533197275, -0.0879381993686312]`
- pooled full CI: `[-0.02311898678734968, -0.0003505607497982389]`

## C) Regenerate Camera-Ready Assets

```bash
python scripts/generate_paper_figures.py \
  --paper-lock configs/paper/publication_assets_lock.yaml \
  --pipeline-spec figures_src/pipeline_diagram_spec.yaml \
  --strict-lock
```

Outputs:
- `outputs/figures/fig0_pipeline_diagram.svg`
- `outputs/figures/fig1_predictive_validity.pdf`
- `outputs/figures/fig2_cluster_repair.pdf`
- `outputs/figures/fig3_causal_geometry.pdf`
- `outputs/figures/fig4_model_replication.pdf`
- `outputs/figures/table1_main_results.txt`
- `outputs/figures/table2_flagged_tokens.txt`
- `outputs/figures/table3_model_replication.txt`
- `outputs/figures/PUBLICATION_MANIFEST.json`

## D) Verify Package Integrity

```bash
python scripts/verify_submission_package.py \
  --lock configs/paper/submission_package_lock.yaml
```

Expected output:
- `PASS: submission package verified`

Verification covers:
1. Hash match for every lock artifact.
2. Claim check consistency against `HARD_PIVOT_REPORT.json`.
3. Internal hash consistency of `PUBLICATION_MANIFEST.json`.

One-command freeze (regenerate + refresh lock hashes + verify):

```bash
python scripts/freeze_submission_package.py
```

## E) Release Candidate Procedure

1. Ensure verification passes.
2. Commit and push manuscript/docs/code updates.
3. Create annotated tag:
   - `git tag -a submission-rcN -m "Hard pivot submission candidate"`
   - `git push origin submission-rcN`
4. Freeze claim-bearing files in lock and avoid edits after tag.
