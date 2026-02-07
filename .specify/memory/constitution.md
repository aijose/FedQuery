<!--
Sync Impact Report
==================
Version change: 0.0.0 → 1.0.0 (MAJOR — initial ratification)
Modified principles: N/A (first version)
Added sections:
  - Core Principles (5): Local-First, Open-Source Only,
    Retrieval Correctness, Citation Grounding, Simplicity
  - Technical Constraints
  - Development Workflow
  - Governance
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md — ✅ no updates needed
    (Constitution Check section is generic; will be filled per feature)
  - .specify/templates/spec-template.md — ✅ no updates needed
    (template is principle-agnostic)
  - .specify/templates/tasks-template.md — ✅ no updates needed
    (task phases are generic)
  - .specify/templates/commands/*.md — N/A (no files present)
Follow-up TODOs: None
-->

# FedQuery Constitution

## Core Principles

### I. Local-First

All components MUST run locally without external service
dependencies at runtime. The vector database, embedding model,
and agent orchestration MUST operate on a single machine.
LLMs accessible from Anthropic via API keys is permitted where needed.

### II. Open-Source Only

Every dependency MUST be open-source with a permissive or
copyleft license (MIT, Apache-2.0, BSD, GPL). No proprietary
SDKs, closed-source embedding models, or commercial vector
databases are permitted.

**Rationale**: The assignment explicitly requires open-source
tooling. This principle ensures the prototype can be freely
shared, reviewed, and reproduced by evaluators.

### III. Retrieval Correctness

The system MUST prioritize factual grounding over generative
creativity. Answers MUST be derived from retrieved FOMC
document passages. The agent MUST NOT fabricate information
absent from the corpus. When retrieval yields insufficient
evidence, the system MUST explicitly communicate uncertainty
rather than hallucinate an answer.

**Rationale**: The evaluation criteria prioritize retrieval
correctness and grounding over model creativity. This
principle enforces evidence-based responses.

### IV. Citation Grounding

Every claim in a generated answer MUST include a citation
referencing the source document and chunk or section. Citations
MUST be traceable — a reviewer MUST be able to locate the
original passage from the citation alone. Unsupported claims
MUST be flagged or omitted.

**Rationale**: The deliverables require answers with citations
(document + section/chunk). This principle makes grounding
verifiable and auditable.

### V. Simplicity

Prefer the simplest solution that satisfies requirements.
Avoid premature abstraction, over-engineering, and speculative
features. The prototype MUST be scoped to approximately 2-3
hours of work. Every added component MUST justify its
existence against the assignment's evaluation criteria.

**Rationale**: The assignment values engineering judgment and
design thinking over polish or scale. YAGNI applies strictly.

## Technical Constraints

Review key design choices with the user and justify rational 
for design before finalizing the design choice.

## Technical Constraints

- **Language**: Python 3.11+
- **Vector Database**: Open-source, local (e.g., Qdrant,
  FAISS, Chroma)
- **Index Comparison**: MUST demonstrate understanding of
  HNSW vs IVF trade-offs (implementation or benchmark)
- **MCP Integration**: Retrieval MUST be exposed via a local
  MCP server; the agent accesses data only through MCP tools
- **Data Source**: Federal Reserve FOMC press statements and
  meeting minutes (limited to a manageable subset)
- **Deliverables**: Runnable repo, README with design
  rationale, CLI or notebook demo

## Development Workflow

- **Branch Strategy**: Feature branches for all work; never
  commit directly to main
- **Commit Discipline**: Incremental commits with descriptive
  messages; commit after each logical unit of work
- **Testing**: Integration tests for MCP tool contracts and
  retrieval accuracy; unit tests for chunking and embedding
  logic where non-trivial
- **Documentation**: README MUST explain design choices
  (chunking, retrieval, agent behavior), grounding/citation
  enforcement, and HNSW vs IVF understanding
- **Code Review**: All changes MUST be reviewable; no opaque
  binary artifacts or unexplained magic numbers

## Governance

This constitution is the authoritative source of project
principles. All implementation decisions MUST comply with
the principles defined above. Deviations MUST be documented
with explicit justification in the relevant spec or plan
artifact.

**Amendment Procedure**: Amendments require documentation of
the change rationale, update to this file, version increment,
and propagation check across dependent templates.

**Versioning Policy**: MAJOR for principle removal or
redefinition, MINOR for new principles or material expansion,
PATCH for wording clarifications.

**Compliance Review**: Every spec and plan MUST include a
Constitution Check gate validating alignment with these
principles before implementation proceeds.

**Version**: 1.0.0 | **Ratified**: 2026-02-06 | **Last Amended**: 2026-02-06
