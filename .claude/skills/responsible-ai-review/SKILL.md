---
name: responsible-ai-review
description: >-
  Review a project or codebase against the ten UK Government AI Principles for
  responsible AI use in public sector work, and produce a RAG (Red/Amber/Green)
  scorecard per principle plus a normalised overall percentage score. Use when
  asked to assess, audit, or score how well a project adopts responsible-AI
  principles, AI governance, or the UK Government AI Playbook / AI principles.
---

# Responsible AI Review

Assess how well a project or codebase adopts the **ten UK Government AI Principles**
(responsible AI use in public sector statistical work) and report a RAG scorecard with
a normalised overall score.

## Capabilities

- Evaluates a project against all ten principles (P1–P10) using a consistent rubric.
- Assigns each principle a RAG rating — 🟢 Green / 🟡 Amber / 🔴 Red — or ⚪ N/A.
- Grounds every rating in concrete evidence from the project (file paths, cells, config).
- Computes an overall adoption score, normalised to a percentage, with an overall RAG band.
- Produces a prioritised list of the highest-impact improvements.

## Workflow

### Phase 1: Load the rubric
1. Read `references/principles.md` (in this skill) for the full definition of each
   principle and the specific evidence to look for and RAG criteria. **Always score
   against that rubric — do not rely on memory of the principles.**

### Phase 2: Survey the project
1. Establish scope: identify the target project/codebase (default to the current repo
   unless the user names a path). Note languages, notebooks, data assets, and docs.
2. Gather evidence broadly but efficiently — READMEs, notebooks, source, config,
   comments/docstrings, table/asset metadata, tests, MLflow/logging, secrets handling,
   and any governance/lineage signals. Prefer search over reading everything.

### Phase 3: Score each principle
For each of P1–P10:
1. Match evidence against the rubric criteria in `references/principles.md`.
2. Assign a rating: Green / Amber / Red, or N/A (only when the principle genuinely does
   not apply — and justify why).
3. Record 1–3 concrete pieces of evidence (or note the *absence* of expected evidence).
4. Note the single most useful improvement for that principle.

### Phase 4: Compute the overall score
Map ratings to points, then normalise (see **Scoring** below). N/A principles are
excluded from both numerator and denominator so a project is never penalised for
principles outside its scope.

### Phase 5: Report
Emit the scorecard in the exact format under **Output format**.

## Scoring

Points per rating:

| Rating | Points |
|--------|--------|
| 🟢 Green | 2 |
| 🟡 Amber | 1 |
| 🔴 Red | 0 |
| ⚪ N/A | excluded |

Let `n` = number of principles rated Green/Amber/Red (i.e. not N/A).

```
overall_% = round( (sum of awarded points) / (2 × n) × 100 )
```

Overall RAG band:

| Overall % | Band |
|-----------|------|
| ≥ 75% | 🟢 Green — strong adoption |
| 40–74% | 🟡 Amber — partial adoption, address gaps |
| < 40% | 🔴 Red — significant gaps, prioritise remediation |

Worked example: 6 Green, 3 Amber, 1 Red, 0 N/A → points = 12 + 3 + 0 = 15;
n = 10; overall = 15 / 20 × 100 = **75% (🟢 Green)**.

## Output format

Produce exactly this structure:

```markdown
# Responsible AI Review — <project name>

**Overall score: <X>% — <🟢/🟡/🔴> <band label>**
(<Green count> Green · <Amber count> Amber · <Red count> Red · <N/A count> N/A)

## Scorecard

| # | Principle | RAG | Evidence | Top improvement |
|---|-----------|-----|----------|-----------------|
| P1 | You know what AI is and its limitations | 🟢/🟡/🔴/⚪ | <cited evidence> | <action> |
| P2 | You use AI lawfully, ethically and responsibly | … | … | … |
| … | … | … | … | … |
| P10 | You use these principles alongside org policy + assurance | … | … | … |

## Priority actions
1. <highest-impact fix, referencing the principle(s) it lifts>
2. …
3. …

## Notes
- <scope, assumptions, and justification for any ⚪ N/A ratings>
```

Rules:
- Every non-N/A rating **must** cite specific evidence (path / cell / snippet) or the
  specific absence of expected evidence. No unsupported ratings.
- Be calibrated and honest: Amber and Red findings are the useful ones. Do not inflate.
- Keep the score reproducible: show the counts so the percentage can be checked against
  the formula.

## Resources

### References
- `references/principles.md`: Full definition of each of the ten UK Government AI
  Principles, the concrete evidence to look for, per-principle RAG criteria, and the
  pipeline stage → principle mapping. Read this before scoring.

## Examples

### Example: Score the current repo
User says: "Review this project against the responsible AI principles and give me a score."
Result: The skill surveys the repo, rates P1–P10 with cited evidence, and returns the
scorecard with an overall normalised percentage and prioritised actions.

### Example: Score a specific subproject
User says: "How well does the fraud-narratives notebook adopt the AI principles?"
Result: Scope is narrowed to that notebook and its assets; principles that don't apply at
that scope are marked ⚪ N/A and excluded from the percentage.
