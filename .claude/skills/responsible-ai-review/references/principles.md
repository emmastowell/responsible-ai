# The Ten UK Government AI Principles — Review Rubric

Source: the ten principles for responsible AI use in public sector organisations, as
set out in the hackathon tutorial (`notebooks/hackathon_tutorial.py`) and the UK
Government's AI Playbook. Use this file as the scoring rubric when reviewing a project.

For each principle you assess, decide a RAG rating using the criteria below and cite
concrete evidence (file paths, notebook cells, comments, config, docs) from the project.

RAG meaning (consistent across all principles):
- 🟢 **Green** — clearly and demonstrably adopted; concrete evidence exists in the project.
- 🟡 **Amber** — partially adopted; started but with material gaps, or asserted without evidence.
- 🔴 **Red** — not adopted; no evidence, or the project actively works against the principle.
- ⚪ **N/A** — genuinely not applicable to this project's scope. Excluded from the score.
  Use sparingly and always justify why it does not apply.

---

## P1 — You know what AI is and what its limitations are

Understanding the strengths and limits of the AI tools used, and validating suitability
in low-risk settings before relying on outputs.

**Evidence to look for**
- Comments/docs explaining *why* a given AI technique or model was chosen and what it is
  and is not good at.
- Explicit acknowledgement of model limitations, failure modes, or uncertainty.
- Experimentation / comparison of approaches before committing (e.g. trying models, sanity
  checks on small samples).

**Green**: Limitations are documented and design choices are justified against them.
**Amber**: Some awareness shown but limitations not systematically addressed.
**Red**: AI outputs treated as ground truth with no discussion of limits.

## P2 — You use AI lawfully, ethically and responsibly

Knowing the data, its sensitivity, biases and legal constraints; evaluating model
performance against suitability criteria; and optimising for financial and environmental cost.

**Evidence to look for**
- Data provenance, licensing, and sensitivity assessment (personal / special-category data).
- Bias / representativeness considerations in the data or model.
- Evaluation against a defined suitability criteria or test set.
- Cost/efficiency awareness: model selection, prompt optimisation, batching, caching,
  choosing smaller models where adequate.

**Green**: Data legality/ethics assessed *and* model suitability evaluated *and* cost considered.
**Amber**: Only some of the above (e.g. cost considered but no bias/suitability evaluation).
**Red**: No consideration of data legality, bias, evaluation, or cost.

## P3 — You know how to use AI securely

Only using approved models/tools for the data and use case, and keeping sensitive data
within appropriate boundaries. (For public/hackathon work: only public data.)

**Evidence to look for**
- Use of approved / governed model endpoints rather than arbitrary external APIs.
- Secrets handling (no hard-coded keys/tokens; use of secret stores / env vars).
- Confirmation that only public / non-sensitive data leaves the trust boundary.
- Access controls on data and compute.

**Green**: Approved endpoints, no leaked secrets, appropriate data boundary respected.
**Amber**: Mostly sound but with gaps (e.g. a hard-coded token, or unclear data boundary).
**Red**: Secrets in code, sensitive data sent to unapproved services, or no security posture.

## P4 — You have meaningful human control at the right stages

Identifying the risk level of the use case and placing human review/validation at the
decision points that matter.

**Evidence to look for**
- Explicit risk-level assessment of the use case.
- Human-in-the-loop / review steps (e.g. domain-expert sense-checking of features or outputs
  before they are used downstream).
- Explainability aids (e.g. SHAP, feature importance, surfaced intermediate outputs).

**Green**: Risk assessed and human checkpoints exist at the stages that matter.
**Amber**: Some human review but ad hoc, or risk level not articulated.
**Red**: Fully automated with no human oversight and no risk framing.

## P5 — You understand how to manage the full AI life cycle

Monitoring the suitability and performance of both the tool and any underpinning foundation
model across the project life cycle.

**Evidence to look for**
- Experiment tracking / model logging (e.g. MLflow), versioning of models and data.
- Monitoring or re-evaluation plans as data or the underlying model changes.
- Reproducibility: pinned dependencies, recorded parameters, seeds.

**Green**: Lifecycle tracked (logging + versioning + monitoring/re-eval intent).
**Amber**: Some tracking (e.g. runs logged) but no monitoring/versioning story.
**Red**: No tracking, versioning, or monitoring.

## P6 — You use the right tool for the job

Matching technique to task — e.g. LLMs for semantic feature extraction from unstructured
text, conventional ML/statistics for modelling where they are more effective and cheaper.

**Evidence to look for**
- Deliberate division of labour between LLM and non-LLM techniques.
- Justification that the chosen tool fits the task (not "LLM for everything").
- Cost/effectiveness reasoning behind tool choices.

**Green**: Tools chosen deliberately and justified per task.
**Amber**: Reasonable choices but little justification, or some mismatched tooling.
**Red**: Wrong tool for the task (e.g. LLM used where simple stats/ML would be better) with no rationale.

## P7 — You are open and collaborative

Making datasets and outputs discoverable and understandable to others, and sharing knowledge.

**Evidence to look for**
- Metadata on created assets (table/column comments, tags, descriptions).
- Documentation: READMEs, docstrings, notebook narrative explaining what and why.
- Discoverability under governance (e.g. catalogued tables, clear naming).

**Green**: Assets and code are documented and discoverable by others.
**Amber**: Partial documentation; some assets lack metadata or explanation.
**Red**: No documentation or metadata; work is opaque to others.

## P8 — You work with commercial/business colleagues from the start

Ensuring the solution delivers real value to end users, not just good evaluation metrics.

**Evidence to look for**
- Stated user need / business value the work serves.
- Outputs surfaced for business users (dashboards, apps, reports).
- Success defined in terms of user benefit, not only technical metrics.

**Green**: Clear user value articulated and outputs delivered in a usable form.
**Amber**: Some user framing but weak link between the work and user benefit.
**Red**: Purely technical exercise with no stated user or value.

## P9 — You have the skills and expertise needed

The team has (or is building) the skills to implement and operate the solution responsibly.

**Evidence to look for**
- Clear, maintainable code and documented approach that others could operate.
- Notes on skills used / learning, or references to established libraries and patterns.
- Absence of copy-paste code the author evidently does not understand.

**Green**: Code and docs demonstrate competent, maintainable implementation.
**Amber**: Workable but fragile or under-explained in places.
**Red**: Code suggests the approach is not understood or is unmaintainable.

## P10 — You use these principles alongside your organisation's policies and have the right assurance in place

Determining the appropriate risk level and required assurance, and maintaining audit,
lineage and governance. (Where no internal policy exists, consider the EU AI Act framework.)

**Evidence to look for**
- Governance: assets under a catalog with ownership, lineage, and access records.
- Audit trail: who created what, when, from what source; reproducible pipeline.
- Reference to organisational policy or an external assurance framework (e.g. EU AI Act).

**Green**: Governed, auditable, with lineage and an assurance/policy reference.
**Amber**: Some governance (e.g. catalogued) but no assurance/policy framing or lineage.
**Red**: No governance, audit trail, or assurance consideration.

---

## Stage → principle mapping (from the tutorial)

Use this to sanity-check coverage across a typical LLM-feature statistical pipeline:

| Stage | Principles |
|-------|-----------|
| 1 — Ingest (load, clean, save as governed table) | P2, P4, P7, P10 |
| 2 — Feature engineering (LLM-derived features) | P1, P6, P9 |
| 3 — Human review (domain-expert sense check) | P4, P8 |
| 4 — Model (train, evaluate, log, explain) | P1, P5, P7 |
| 6 — Surface outputs for users (dashboard/app) | P6, P8 |
| 7 — Govern (audit, lineage, monitoring) | P5, P10 |
