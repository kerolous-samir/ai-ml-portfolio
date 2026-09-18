# Medical Assistant — Retrieval-Augmented Generation for Clinical Q&A

> A low-resource retrieval prototype for clinical Q&A: a working TF-IDF retriever over a curated
> medical corpus, with ablations over prompt phrasing and retrieval depth — and an honest account
> of why the generation half is not yet wired to it.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20NearestNeighbors-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![Retrieval](https://img.shields.io/badge/Top--1%20retrieval-5%2F5%20questions-success)
![Status](https://img.shields.io/badge/Status-retriever%20working%20%C2%B7%20generation%20not%20conditioned-orange)

## Business Context

Clinicians need fast, reliable answers to point-of-care questions. A general language model can
produce fluent medical text, but without grounding it may state things no source supports — an
unacceptable failure mode in a clinical setting.

**Retrieval-augmented generation (RAG)** addresses this by retrieving relevant passages from a
trusted knowledge base and conditioning the generated answer on them, so every response traces back
to source material.

## Objective

Build and evaluate the components of a RAG system for clinical questions, and measure how
**prompt phrasing** and **retrieval depth** affect output quality.

**What this notebook actually delivers:** a functioning retriever and an evaluation harness. The
generation stage is a placeholder, and the retrieved context is never passed to it — so the
ablations below measure the retriever and the scaffolding, not end-to-end grounding. That gap is
documented precisely rather than papered over; see [Known Limitations](#known-limitations).

## Pipeline

```
Clinical question
       │
       ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│  TF-IDF vector  │────▶│  Knowledge base                  │
│    encoding     │     │  • 5 curated medical summaries   │
└─────────────────┘     │  • medical diagnosis manual (PDF)│
       │                └──────────────────────────────────┘
       ▼
┌─────────────────┐
│ NearestNeighbors│  retrieve top-k passages  (cosine)
│    retrieval    │
└────────┬────────┘
         │ question only — retrieved context is assembled, then discarded
         ▼
┌─────────────────┐
│    Generator    │  static keyword-matched answer
│   (fallback)    │
└─────────────────┘
```

That last arrow carries the question alone: `rag_answer()` (cell 9) builds
`context = ' '.join([d['text'] for d in docs])` and the very next line calls
`generate_response(question)` — without `context`. Retrieval and generation are fully decoupled.

## Knowledge Base

| Source | What is in it | Verified in notebook |
|--------|---------------|:--------------------:|
| Curated summaries | 5 hand-written snippets: `doc_sepsis`, `doc_appendicitis`, `doc_hair`, `doc_brain`, `doc_fracture` | Yes — cell 7 source |
| Medical diagnosis manual | Up to the first five pages (`min(5, doc.page_count)`), extracted with **PyMuPDF** | No — see below |

The manual ingestion is wrapped in a bare `try/except` that silently sets `manual_docs = []` on any
failure, and the notebook never prints `len(corpus)`. The only evidence it produced is the line
`PDF path: /content/manual_extract/medical_diagnosis_manual.pdf`, which shows the zip was extracted,
not that page text entered the index. No `manual_page_*` id appears in any output. **Corpus size is
therefore between 5 and 10 documents, and the notebook does not establish which.**

Index: `TfidfVectorizer(stop_words='english')` → `NearestNeighbors(metric='cosine')`.

## Approach

1. **Baseline answers** — generate responses with no retrieval, to establish what the generator produces unaided (cell 3).
2. **Prompt engineering** — build 5 prompt templates, each prepending a different instruction, across the 5 questions (25 runs). **Note:** cell 5 formats the prompt into a `prompt` variable and then calls `generate_response(question)`, so the templates are never passed to the generator.
3. **Corpus construction** — build a knowledge base from 5 curated medical summaries plus text from the first five pages of a medical diagnosis manual (**PyMuPDF**), then index it with **TF-IDF**.
4. **RAG question answering** — retrieve top-`k` passages and sweep `k ∈ {1, 2, 3}` to measure retrieval depth. **Conditioning is not wired up:** the retrieved context is assembled but not passed to the generator, so this step measures the retriever, not end-to-end grounding.
5. **Evaluation** — score all 15 question×`k` runs on **groundedness** (were any documents retrieved?) and **relevance** (do the expected literal keywords appear in the answer?).

### The generator

`generate_response()` (cell 1) tries three paths in order:

| Order | Path | Outcome in the executed run |
|:-----:|------|------------------------------|
| 1 | `llama_cpp` → `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` (`mistral-7b-instruct-v0.2.Q6_K.gguf`) | Not reached — no model loaded |
| 2 | `transformers` → `AutoModelForCausalLM('distilbert-base-uncased')` | Not reached |
| 3 | A hardcoded 5-entry `dummy_answers` dict, substring-matched on `sepsis` / `appendicitis` / `hair` / `brain` / `fracture`, else `"I'm sorry, I don't have information on that."` | **This is the path that ran** |

Cell 3's printed baseline answers are the dictionary's strings verbatim, which confirms path 3.
Every answer in this project comes from that lookup.

## Evaluation Questions

Five clinical questions spanning emergency medicine, surgery, dermatology and trauma (exact wording
from cell 3):

| # | Question |
|:-:|----------|
| 1 | What is the protocol for managing sepsis in a critical care unit? |
| 2 | What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure is recommended? |
| 3 | A patient exhibits patchy hair loss. What are possible causes and treatments? |
| 4 | What are the treatment options for a traumatic brain injury (TBI)? |
| 5 | How should a broken leg be managed both in a hospital and in the wilderness? |

**Retrieval succeeded on all five.** The correct supporting document was ranked first at every `k`:
Q1→`doc_sepsis`, Q2→`doc_appendicitis`, Q3→`doc_hair`, Q4→`doc_brain`, Q5→`doc_fracture`.

**Generation did not use it.** Because `rag_answer()` drops the context, every RAG answer is
byte-identical to the corresponding no-retrieval baseline and unchanged across `k = 1, 2, 3`.

Question 5 returned the fallback string *"I'm sorry, I don't have information on that."* This is a
**generator failure, not a desirable refusal.** The wilderness scenario is in the corpus —
`doc_fracture` reads *"In wilderness settings, immobilize the limb using a rigid, padded splint,
control bleeding, assess circulation and nerve function, and keep the patient warm while awaiting
rescue"* — and that document was retrieved first at every `k`. The refusal fires only because the
literal token `fracture` does not occur in the words "broken leg". Relevant context was retrieved
and then ignored.

## Findings

| Dimension | Finding |
|-----------|---------|
| **Baseline (no retrieval)** | Keyword-driven, high-level summaries; no source citation and little clinical nuance |
| **Prompt engineering** | All 25 runs (5 templates × 5 questions) produced exactly one distinct answer per question. This is not evidence that phrasing does not matter: the formatted prompt is never passed to the generator (cell 5), and the generator keys only on question substrings |
| **Corpus design** | Every retrieval — all `k`, all 5 questions, 15/15 rows — returned only curated snippets. No manual page was ever retrieved, so the manual's contribution to coverage is untested |
| **Retrieval depth (`k`)** | Retrieval sets grow as expected (`k=1` → correct doc only; `k=3` → correct doc plus two lexical neighbours). Answers were identical at every `k` — structurally guaranteed, since the generator never sees the retrieved text. This ablation therefore says nothing about retrieval depth |
| **Retriever quality** | Top-1 document correct for 5/5 questions, including the one the generator refused. TF-IDF + cosine is adequate for a corpus this small and this lexically distinct |
| **Groundedness / relevance** | Groundedness = True in 15/15 rows, but the check is `len(retrieved) > 0`, which cannot fail. The keyword relevance heuristic passed 9/15: Q1–Q3 at every `k`; Q4 failed at every `k` because the answer says "TBI" where the heuristic requires the literal terms "brain" and "injury"; Q5 failed at every `k` because of the fallback refusal |

## Known Limitations

These are defects in the current notebook, not caveats about scope. Nothing below has been fixed —
this section documents the code as committed.

- **The pipeline does not perform RAG.** `rag_answer()` (cell 9) computes `context` from the
  retrieved documents and then calls `generate_response(question)` without it. Retrieval runs and
  works; generation is a static lookup that is independent of what was retrieved. The in-code
  comment concedes this: *"here we reuse the baseline generator."*
- **The `k` ablation cannot produce a finding.** Since the generator never receives the context,
  identical answers across `k = 1, 2, 3` were structurally guaranteed before the experiment ran.
  Reporting this as a result about retrieval depth would be circular.
- **The prompt-engineering ablation has the same defect.** Cell 5 builds
  `prompt = variant.format(question)` and then discards it, calling `generate_response(question)`.
  The 25-row table measures nothing.
- **The answer key is the answer source.** The `dummy_answers` dict (cell 1) was hand-written
  against these five specific questions. Any apparent answer quality is authored in, not produced —
  a construction-by-leakage that makes the baseline and RAG numbers uninformative about model
  capability.
- **Groundedness is a vacuous metric.** It is defined as `len(row['Retrieved Docs']) > 0` (cell 11),
  and `retrieve()` always returns `k ≥ 1` documents, so it returns `True` by construction in every
  row. It measures nothing about whether the answer is supported.
- **Relevance is literal substring matching.** Q4 scores `False` at every `k` because the answer
  uses "TBI" while the keyword list demands "brain" and "injury" — a false negative on a clinically
  correct answer. The metric is a string test, not a correctness test; no factual correctness,
  hallucination or citation check exists anywhere in the notebook.
- **The committed notebook does not run as-is.** Cell 7 contains an unterminated string literal
  (`zip_path = '/content/` followed by a line break), which is a `SyntaxError` at parse time. The
  stored outputs come from an earlier session whose path has since been broken.
- **Colab-only and unpinned.** Cell 7 calls `google.colab.drive.mount` and hardcodes `/content/…`
  paths; there is no `pip install` cell, so PyMuPDF, `llama-cpp-python` and Transformers are assumed
  present rather than declared. The PyMuPDF block's bare `except Exception: manual_docs = []` hides
  any ingestion failure without a warning.
- **The corpus is 5 curated documents (plus at most 5 unverified manual pages).** Coverage, not
  reasoning, is the binding constraint, and the retrieval task is easy: each question maps to a
  distinct document with near-zero lexical overlap with the others.

## Next Steps

In priority order — the first item is a prerequisite for every experiment in this notebook being
meaningful.

1. **Wire the context into generation** — pass `context` into the prompt in `rag_answer()`, and pass
   the formatted prompt in the prompt-engineering cell. Until then the two ablations are no-ops.
2. **Fix and pin the environment** — repair the cell 7 string literal, add an install cell, and
   replace the Colab-mounted path with a repo-relative one so the notebook runs end to end.
3. **Real generator** — swap the fallback dict for a medical instruction-tuned LLM, then re-run the
   prompt and `k` ablations, which only then measure anything.
4. **Real evaluation** — replace `len(docs) > 0` with a check that the answer's claims appear in the
   retrieved text, and replace keyword matching with an LLM judge scoring factual correctness and
   citation. Add a held-out question set not used to author the corpus.
5. **Expand the corpus and go dense** — ingest the full manual plus external clinical guidelines,
   verify ingestion with an asserted corpus count, and replace TF-IDF with sentence embeddings
   (e.g. SBERT) for semantic rather than lexical matching.
6. **Citation surfacing** — return the supporting passage alongside each answer so a clinician can
   verify before acting.

## Tech Stack

`Python 3.10` · `scikit-learn` (TF-IDF, NearestNeighbors) · `pandas`

*Present in the code but not exercised in the stored run:* `Transformers` / `PyTorch` and
`llama-cpp-python` (both fallback paths were skipped), and `PyMuPDF` (ingestion produced no
observable output).

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full RAG pipeline with ablations, as executed — see Known Limitations before reading the results |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/07-medical-rag-assistant/report.html)** |
