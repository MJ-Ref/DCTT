# Documentation Verification Skill

Verify that all DCTT documentation is in sync after changes.

## Checklist

### 1. README.md
- [ ] Project status section reflects current milestone completion
- [ ] Experimental results tables have latest numbers
- [ ] Code examples match current API signatures
- [ ] Installation instructions are accurate

### 2. Feedback.md (Mock Review)
- [ ] Status assessment updated (weak reject → borderline → etc.)
- [ ] Addressed issues marked with ~~strikethrough~~ and ✅
- [ ] UPDATE timestamps reflect latest changes
- [ ] Remaining issues accurately listed

### 3. Plan File
- [ ] Status line in Executive Summary is current
- [ ] Completed steps marked ✅ COMPLETE with details
- [ ] Next Session Priority reflects actual next steps
- [ ] Statistics (module count, test count) are accurate

### 4. Config Files
- [ ] `configs/config.yaml` has all experiment sections
- [ ] Default values match what experiments actually use

### 5. Cross-Document Consistency
- [ ] Same numbers appear in README, Feedback.md, and plan
- [ ] Claim language is consistent (mechanistic vs behavioral)
- [ ] All new scripts mentioned in Commands Reference

## Verification Process

1. Read each document in parallel
2. Extract key metrics and claims
3. Compare for consistency
4. Report any discrepancies with specific line numbers
5. Suggest fixes if inconsistencies found

## Expected Output

```
Documentation Verification Report
=================================
README.md: ✓ In sync
Feedback.md: ✓ In sync
Plan file: ✓ In sync
Configs: ✓ In sync

Cross-document consistency: ✓ All metrics match
- Geometry AUC: 0.80 (3 docs)
- Cond reduction: 0.43 (3 docs)
- Treatment effect: -0.27 (3 docs)

No issues found.
```

Or if issues exist:

```
Documentation Verification Report
=================================
README.md: ⚠ Issue found
  Line 45: States "5 clusters" but plan says "69 clusters found"

Suggested fix: Update README line 45 to match plan

Action needed: 1 inconsistency to resolve
```
