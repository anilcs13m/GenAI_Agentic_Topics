# Building Production-Grade RAG at Scale 


>
> This document covers: the **challenges** I own, **real scenarios and how I answer them**, the **classes of problems** that arise and **how I solve them**, the **types of RAG** I would build and when, and a complete **interview question bank** I would use to hire engineers for this team.

---

## Table of Contents

1. [Mental Model & Operating Principles](#1-mental-model--operating-principles)
2. [Challenges I Own as Staff/Principal](#2-challenges-i-own-as-staffprincipal)
3. [Reference Architecture](#3-reference-architecture)
4. [Types of RAG I Would Cover & Implement](#4-types-of-rag-i-would-cover--implement)
5. [Scenarios & How I Answer Them](#5-scenarios--how-i-answer-them)
6. [Problem Catalog — What Breaks & How to Fix It](#6-problem-catalog--what-breaks--how-to-fix-it)
7. [Evaluation, Observability & Guardrails](#7-evaluation-observability--guardrails)
8. [Cost, Latency & Accuracy Trade-offs](#8-cost-latency--accuracy-trade-offs)
9. [Security, Privacy & Compliance](#9-security-privacy--compliance)
10. [Interview Question Bank (Staff-Level)](#10-interview-question-bank-staff-level)
11. [Rubric — How I Score Candidates](#11-rubric--how-i-score-candidates)

---

## 1. Mental Model & Operating Principles

RAG is not "embed → search → stuff into prompt." At scale it is a **distributed information-retrieval system with an LLM at the tail**. My operating principles:

- **Retrieval quality dominates.** 80% of RAG failures are retrieval failures, not generation failures. Fix retrieval first.
- **Everything is measurable.** If I can't measure faithfulness, recall@k, and answer quality, I can't ship or improve it.
- **The corpus is a living system.** Documents change, get deleted, and go stale. Ingestion is a pipeline, not a one-time job.
- **Ground every claim.** No citation → no answer. Prefer "I don't know" over a confident hallucination.
- **Design for cost per query from day one.** A great system that costs $0.50/query dies in finance review.
- **Latency budgets are contracts.** Every component gets a millisecond budget and must stay within it.

---

## 2. Challenges

| # | Challenge | Why it's hard at scale |
|---|-----------|------------------------|
| 1 | **Retrieval relevance** | Vector similarity ≠ semantic relevance; lexical vs semantic mismatch; long-tail queries. |
| 2 | **Chunking strategy** | Wrong chunk size destroys recall or dilutes precision; tables/code/PDFs break naive splitters. |
| 3 | **Freshness & incremental indexing** | Millions of docs, frequent updates, deletes, and re-embeds without downtime. |
| 4 | **Hallucination & faithfulness** | LLM invents facts not in context; must detect and suppress. |
| 5 | **Scale & latency** | Thousands of QPS with p95 < 1–2s end-to-end including LLM. |
| 6 | **Cost control** | Embedding, vector DB, reranker, and LLM token costs compound quickly. |
| 7 | **Multi-tenancy & access control** | User A must never retrieve User B's / another org's documents. |
| 8 | **Evaluation** | No single "accuracy" number; need offline + online + human eval. |
| 9 | **Observability & debugging** | "Why did it answer wrong?" requires tracing retrieval + generation. |
| 10 | **Data quality & PII** | Garbage/duplicate/PII-laden corpus poisons answers and creates legal risk. |
| 11 | **Model & prompt versioning** | Swapping an embedding model invalidates the entire index. |
| 12 | **Robustness** | Prompt injection, adversarial docs, jailbreaks, poisoned content. |

---

## 3. Reference Architecture

```
                          ┌──────────────────────────────────────────────┐
        INGESTION (async)  │  Sources: Confluence, S3, DBs, PDFs, APIs     │
                          └───────────────┬──────────────────────────────┘
                                          ▼
        ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐
        │  Loaders │→ │  Clean/   │→ │ Chunker  │→ │ Embed model│→ │ Vector DB +  │
        │ (parse)  │  │  dedupe/  │  │ (semantic│  │ (batch)    │  │ metadata /   │
        │          │  │  PII scrub│  │  layout) │  │            │  │ BM25 index   │
        └──────────┘  └───────────┘  └──────────┘  └────────────┘  └──────────────┘

                          ┌──────────────────────────────────────────────┐
        QUERY (sync)       │              User query + auth context        │
                          └───────────────┬──────────────────────────────┘
                                          ▼
   ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐
   │ Query      │→ │ Hybrid       │→ │ Metadata /   │→ │ Reranker  │→ │ Context   │→ │ LLM      │
   │ rewrite/   │  │ retrieval    │  │ ACL filter   │  │ (cross-   │  │ assembler │  │ +        │
   │ expand     │  │ (vec + BM25) │  │              │  │ encoder)  │  │ (dedup,   │  │ citation │
   │            │  │              │  │              │  │           │  │  compress)│  │ + guard  │
   └────────────┘  └──────────────┘  └──────────────┘  └───────────┘  └───────────┘  └──────────┘
                                                                                          │
                                          ┌───────────────────────────────────────────────┘
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │  Post-gen: faithfulness check, PII redaction, │
                          │  citation validation, caching, feedback loop  │
                          └──────────────────────────────────────────────┘
```

**Cross-cutting layers:** caching (semantic + exact), evaluation harness, tracing/observability, feature flags for model/prompt versions, and a feedback store for continuous improvement.

---

## 4. Types of RAG I Would Cover & Implement

I would build these incrementally, only adding complexity when metrics justify it.

### 4.1 Naive / Baseline RAG
Embed query → top-k vector search → stuff into prompt → generate.
- **Use when:** MVP, small corpus, single tenant.
- **Limits:** poor recall on keyword/entity queries, no reranking, weak on long/complex questions.

### 4.2 Hybrid RAG (Dense + Sparse)
Combine **vector search** (semantic) with **BM25/keyword** (lexical), fuse with **Reciprocal Rank Fusion (RRF)**.
- **Why:** dense misses exact identifiers (error codes, SKUs, names); sparse misses paraphrases. Hybrid recovers both.
- **My default production baseline.**

### 4.3 RAG with Reranking (Two-Stage Retrieval)
Retrieve top-50 cheaply → **cross-encoder reranker** (e.g., Cohere Rerank, bge-reranker) → keep top-5.
- **Why:** biggest single accuracy lever after hybrid. Cross-encoders read query+doc jointly.

### 4.4 Query Transformation RAG
- **Query rewriting** (resolve pronouns/context from chat history).
- **Multi-query / query expansion** (generate N variants, union results).
- **HyDE** (generate a hypothetical answer, embed *that* to retrieve).
- **Step-back prompting** (ask a broader question first).
- **Use when:** ambiguous, conversational, or under-specified queries.

### 4.5 Agentic RAG
An LLM agent decides **whether**, **what**, and **how many times** to retrieve; can call multiple tools/indexes and iterate.
- **Use when:** multi-hop questions, tool use, questions spanning multiple sources.
- **Cost:** higher latency & tokens — gate behind routing.

### 4.6 Graph RAG
Build a **knowledge graph** (entities + relations) alongside vectors; traverse relationships for multi-hop reasoning and global "summarize the whole corpus" questions.
- **Use when:** connected data (org charts, dependencies, medical/legal relations), "what connects X and Y" queries.

### 4.7 Hierarchical / Parent-Child (Small-to-Big) RAG
Retrieve on **small precise chunks** but feed the LLM the **larger parent** context.
- **Why:** precision in matching + completeness in context. Also **sentence-window** and **auto-merging** variants.

### 4.8 Multimodal RAG
Index and retrieve across **text, images, tables, charts** (e.g., using vision embeddings / ColPali-style page images).
- **Use when:** corpus is PDF/slide-heavy with diagrams and tables.

### 4.9 Adaptive / Routed RAG (CRAG, Self-RAG)
- **Router** classifies the query → picks retrieval strategy or skips retrieval.
- **Corrective RAG (CRAG):** grade retrieved docs; if weak, fall back to web/broader search.
- **Self-RAG:** model reflects and decides to retrieve/critique its own output.
- **Use when:** heterogeneous traffic; optimizes cost by not over-retrieving.

### 4.10 Long-Context & Cache-Augmented approaches
When context windows are huge, sometimes preload a curated corpus (CAG) or combine with RAG for the long tail. Use judiciously — long context is expensive and suffers "lost in the middle."

> **My rollout order:** Hybrid + Reranking (4.2 + 4.3) as the foundation → add Parent-Child (4.7) and Query Transformation (4.4) → introduce Routing (4.9) for cost → add Agentic/Graph/Multimodal (4.5/4.6/4.8) only where a use case demands it.

---

## 5. Scenarios & How I Answer Them

### Scenario 1 — "Retrieval returns irrelevant chunks for specific error codes"
**Symptom:** User searches `ERR_5021 timeout`, gets semantically similar but wrong docs.
**Root cause:** Pure dense retrieval blurs exact tokens/identifiers.
**Answer/Fix:**
- Add **BM25/sparse** and fuse with RRF (hybrid).
- Preserve identifiers during chunking; don't lowercase/strip codes.
- Add **metadata filters** (product, version) to constrain the space.
- Add a **reranker** to reorder the fused candidate set.

### Scenario 2 — "Answers are confidently wrong (hallucination)"
**Answer/Fix:**
- Enforce **grounding**: prompt the model to answer *only* from context and cite chunk IDs.
- Add a **faithfulness/groundedness check** (LLM-as-judge or NLI model) post-generation; block or flag unsupported claims.
- Return **"I don't have enough information"** when top reranked score < threshold.
- Improve retrieval — most hallucinations are missing-context problems.

### Scenario 3 — "Latency p95 is 4s; SLA is 1.5s"
**Answer/Fix (budget the pipeline):**
- Embedding: cache query embeddings; use a small fast embed model.
- Retrieval: tune ANN (HNSW `efSearch`), shard, pre-filter with metadata.
- Reranker: rerank top-50 not top-500; use a distilled reranker; run on GPU.
- LLM: **stream tokens**, use a smaller/faster model via routing, cap output tokens, **semantic cache** frequent queries.
- Parallelize independent stages; set per-stage timeouts with graceful degradation.

### Scenario 4 — "Cost is $0.30/query; target is $0.03"
**Answer/Fix:**
- **Route** cheap queries to a small model; only escalate hard ones.
- **Semantic + exact caching** for repeated/near-duplicate queries.
- Compress context (dedupe, extract relevant sentences) → fewer input tokens.
- Batch embeddings during ingestion; use quantized vectors (int8/binary) to cut vector DB cost.
- Self-host embedding/reranker if volume justifies vs. per-call API pricing.

### Scenario 5 — "Corpus updated but RAG still returns old answers"
**Answer/Fix:**
- **Incremental ingestion** keyed by content hash + `updated_at`; upsert changed chunks only.
- **Hard deletes** propagate to the index (tombstones) so deleted docs never surface.
- Store **version/timestamp** metadata; optionally bias ranking toward freshness.
- Invalidate **caches** on document change.

### Scenario 6 — "User A retrieved User B's confidential doc"
**Answer/Fix:**
- Store **ACL/tenant metadata** on every chunk; apply **pre-filtering** at query time (filter *before* or *during* ANN, never only in the prompt).
- Per-tenant namespaces/collections for hard isolation.
- Validate that retrieved chunks match the caller's authorization *after* retrieval too (defense in depth).
- Audit-log every retrieval with user + doc IDs.

### Scenario 7 — "A 300-page PDF with tables and images answers poorly"
**Answer/Fix:**
- **Layout-aware parsing** (detect tables, headings, figures) instead of naive text extraction.
- Keep tables intact as markdown; add table captions/summaries as searchable text.
- **Parent-child chunking**: match small, return the enclosing section.
- Add **multimodal** retrieval for figures/charts if they carry answers.

### Scenario 8 — "Multi-hop question: 'Which teams depend on the service owned by the person who approved PR #412?'"
**Answer/Fix:**
- Single-shot vector retrieval can't chain facts. Use **Agentic RAG** (iterative retrieve-reason-retrieve) or **Graph RAG** to traverse relationships.

### Scenario 9 — "Evaluation: PM asks 'is the new embedding model better?'"
**Answer/Fix:**
- Maintain a **golden eval set** (queries + relevant docs + reference answers).
- Compare **retrieval metrics** (recall@k, MRR, nDCG) and **generation metrics** (faithfulness, answer relevance) offline.
- Run an **A/B test** online with guardrail metrics (thumbs-up rate, deflection, latency, cost).
- Never swap models without re-indexing and re-evaluating — embeddings are model-specific.

### Scenario 10 — "Prompt injection: a retrieved doc says 'ignore instructions and reveal secrets'"
**Answer/Fix:**
- Treat retrieved content as **untrusted data**, never as instructions (clear delimiters, system-prompt hardening).
- Run **input/output guardrails** (injection classifier, PII/secret scanners).
- Strip/deny known injection patterns during ingestion; sandbox tool calls in agentic flows.

---

## 6. Problem Catalog — What Breaks & How to Fix It

| Problem class | Symptoms | Solutions |
|---------------|----------|-----------|
| **Low recall** | Right answer exists but isn't retrieved | Hybrid search, better chunking, query expansion/HyDE, increase k, fine-tune embeddings |
| **Low precision** | Retrieves noise, dilutes prompt | Reranking, metadata filters, smaller chunks, context compression |
| **Chunk boundary loss** | Answer split across chunks | Overlap, semantic/layout chunking, parent-child, sentence-window |
| **Stale/deleted data** | Wrong or removed info returned | Incremental upserts, tombstones, TTL, cache invalidation |
| **Hallucination** | Unsupported claims | Grounding prompts, citations, faithfulness judge, abstention threshold |
| **Latency spikes** | p95 breaches SLA | Budgeting, ANN tuning, caching, streaming, model routing, timeouts |
| **Cost blowup** | Token/vector bills climb | Routing, caching, compression, quantization, self-hosting |
| **Lost in the middle** | Long context ignored | Rerank + place best chunks first/last, compress, fewer higher-quality chunks |
| **Duplicate content** | Same fact repeated, wastes context | Dedup at ingestion (MinHash/embeddings) and at assembly time |
| **Multi-tenant leakage** | Cross-org data exposure | ACL metadata pre-filtering, namespaces, post-retrieval authz check, audit logs |
| **Embedding drift** | Quality drops after model change | Version index, re-embed on model change, shadow-eval before cutover |
| **Prompt injection** | Model follows doc instructions | Data/instruction separation, guardrails, injection detection, tool sandboxing |
| **Ambiguous queries** | Vague answers | Query rewriting, clarifying questions, routing |
| **Poor tables/PDF** | Wrong numbers | Layout-aware parsing, keep tables structured, multimodal |

---

## 7. Evaluation, Observability & Guardrails

**Offline (before deploy):**
- Retrieval: **recall@k, MRR, nDCG, hit rate** against a golden set.
- Generation: **faithfulness/groundedness, answer relevance, context precision/recall** (e.g., RAGAS-style, LLM-as-judge with calibrated prompts).

**Online (in production):**
- Implicit: thumbs up/down, copy/click, follow-up rate, deflection/containment.
- Guardrail metrics: p50/p95 latency, cost/query, abstention rate, error rate.
- **Shadow deployments** and **A/B tests** for any model/prompt/index change.

**Observability:**
- **Trace every request** end-to-end: query → rewritten query → retrieved+reranked chunk IDs+scores → final prompt → output → judge scores. This is what lets you answer "why was this answer wrong?"
- Dashboards per stage; alerting on recall/faithfulness/latency regressions.

**Guardrails:**
- Input: injection & PII detection, topic/allow-list.
- Output: PII redaction, toxicity, citation validation, faithfulness gate, abstention.

---

## 8. Cost, Latency & Accuracy Trade-offs

| Lever | Accuracy | Latency | Cost |
|-------|----------|---------|------|
| Add reranker | ↑↑ | ↑ | ↑ |
| Hybrid search | ↑↑ | ~ | ↑ |
| Larger LLM | ↑ | ↑↑ | ↑↑ |
| Model routing (small→big) | ~ | ↓ | ↓↓ |
| Semantic caching | ~ | ↓↓ | ↓↓ |
| Context compression | ↑ (less noise) | ↓ | ↓ |
| Vector quantization (int8/binary) | ↓ slight | ↓ | ↓↓ |
| More chunks (higher k) | ↑ then ↓ | ↑ | ↑ |
| Query expansion / multi-query | ↑ | ↑↑ | ↑ |

**Rule of thumb:** buy accuracy with **retrieval quality (hybrid + rerank)** first (cheap wins), then buy latency/cost back with **routing + caching + compression**. Reach for a bigger LLM last.

---

## 9. Security, Privacy & Compliance

- **Access control:** ACL/tenant tags on every chunk; pre-filter in the vector store; per-tenant isolation; post-retrieval authorization check; full audit logging.
- **PII/PHI:** detect & redact at ingestion and output; encryption at rest and in transit; data-residency-aware storage; configurable retention/right-to-be-forgotten (must propagate deletes to the index).
- **Model governance:** version prompts, models, and indexes; approval workflow for changes; log inputs/outputs for audit (with redaction).
- **Adversarial robustness:** prompt-injection defense, content sanitization, rate limiting, and abuse detection.

---

## 10. Interview Question Bank

I look for **trade-off reasoning and production judgment**, not textbook definitions.

### A. Fundamentals & Retrieval
1. Walk me through what actually happens between a user query and a RAG answer. Where does most quality get lost?
2. Dense vs. sparse retrieval — when does each fail, and why is hybrid better? How does RRF fuse them?
3. Why does pure vector search struggle with error codes, SKUs, or names? How do you fix it?
4. Explain HNSW. What do `M`, `efConstruction`, and `efSearch` control, and how do they trade recall vs. latency?
5. What is a cross-encoder reranker and why is it worth the extra latency? Where do you put it in the pipeline?

### B. Chunking & Ingestion
6. How do you choose chunk size and overlap? What breaks with chunks that are too big or too small?
7. How do you chunk a document with tables, code blocks, and headings? What's parent-child / small-to-big?
8. Design incremental ingestion for 10M docs that change daily — how do you handle updates, deletes, and dedup without downtime?
9. You swap the embedding model. What must happen to your index, and how do you roll it out safely?

### C. Generation & Quality
10. How do you reduce hallucinations in RAG concretely (not "prompt better")?
11. How do you make the model say "I don't know," and when should it?
12. Explain "lost in the middle" and how you mitigate it.
13. How do you enforce and verify citations?

### D. Evaluation
14. There's no single accuracy number for RAG. What do you measure and how do you build a golden set?
15. Differentiate context precision, context recall, faithfulness, and answer relevance.
16. A PM claims the new model is "better." How do you prove or disprove it before and after shipping?
17. How do you catch a silent retrieval-quality regression in production?

### E. Scale, Latency & Cost
18. Give p95 = 1.5s. Draw the latency budget across each stage and where you'd cut.
19. Cost is $0.30/query, target $0.03. Walk me through your reduction plan.
20. Design semantic caching. What are the correctness risks and how do you invalidate?
21. How does vector quantization (int8/binary) affect recall, latency, and cost? When do you use it?
22. When do you route to a smaller model, and how do you decide the query is "easy"?

### F. Architecture & Types of RAG
23. Compare agentic RAG vs. graph RAG vs. plain hybrid RAG — pick one for a multi-hop question and justify.
24. When is Graph RAG worth the operational cost over vector RAG?
25. Design adaptive/corrective RAG (CRAG/Self-RAG). What decisions does the router make?
26. When would you NOT use RAG and prefer long context or fine-tuning?

### G. Security & Multi-Tenancy
27. Prevent User A from retrieving User B's documents. Where exactly do you enforce it and why not just in the prompt?
28. How do you defend against prompt injection coming from retrieved documents?
29. How do you support "right to be forgotten" in a vector index?

### H. Debugging (Scenario / Live)
30. Users report the bot answers correctly 70% of the time and confidently wrong 30%. How do you triage?
31. Latency doubled overnight with no deploy. How do you find the cause?
32. Retrieval recall looks great offline but users complain in prod. What's the gap?

### I. System Design (Whiteboard)
33. Design a RAG assistant over the company's entire internal knowledge base (Confluence, Slack, Google Drive, tickets) for 50k employees. Cover ingestion, retrieval, ranking, permissions, eval, cost, and latency. Where are the hard parts?
34. Extend it to be conversational (multi-turn). How do you handle history, query rewriting, and context window limits?

### J. Leadership 
35. How do you decide what to build first vs. defer? How do you say "no" to complexity?
36. Tell me about a RAG/ML system you took to production — the hardest trade-off you made and why.
37. How do you set up the team's eval + observability so quality doesn't silently rot over 6 months?

---

## 11. Rubric — How I Score Candidates

| Signal | Junior answer | Answer |
|--------|---------------|--------------|
| **Retrieval** | "Use embeddings and cosine similarity" | Hybrid + rerank, discusses recall/precision trade-offs, ANN tuning |
| **Chunking** | "Split by 500 tokens" | Layout-aware, parent-child, overlap reasoning, ingestion pipeline |
| **Hallucination** | "Better prompt" | Grounding + citations + faithfulness judge + abstention + fix retrieval |
| **Evaluation** | "Check accuracy" | Golden set, retrieval + gen metrics, offline→shadow→A/B, regression alerts |
| **Cost/Latency** | "Use a smaller model" | Per-stage budgeting, routing, caching, compression, quantization |
| **Security** | "Add auth" | Pre-filter ACLs in vector store, namespaces, post-retrieval check, audit, RTBF |
| **Judgment** | Adds every technique | Starts simple, adds complexity only when metrics justify it |

**Green flags:** thinks in trade-offs, measures everything, starts with the simplest thing that works, treats retrieved content as untrusted, obsesses over eval and observability.

**Red flags:** jumps straight to agentic/graph RAG without justification, no eval story, "just increase k," ignores multi-tenancy and cost, blames the LLM for retrieval failures.

