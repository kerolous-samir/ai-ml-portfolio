# Medical Assistant — Retrieval-Augmented Generation for Clinical Q&A

> A low-resource RAG pipeline that grounds clinical answers in a curated medical corpus, with
> systematic ablations over prompt phrasing and retrieval depth.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-NLP-FFD21E)](https://huggingface.co/docs/transformers)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20Retrieval-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
![Type](https://img.shields.io/badge/Type-RAG%20%2F%20NLP-blue)

## Business Context

Clinicians need fast, reliable answers to point-of-care questions. A general language model can
produce fluent medical text, but without grounding it may state things no source supports — an
unacceptable failure mode in a clinical setting.

**Retrieval-augmented generation (RAG)** addresses this by retrieving relevant passages from a
trusted knowledge base and conditioning the generated answer on them, so every response traces back
to source material.

## Objective

Build and evaluate a RAG system that answers clinical questions grounded in a medical knowledge
base, and measure how **prompt phrasing** and **retrieval depth** affect output quality.

## Pipeline

```
Clinical question
       │
       ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│  TF-IDF vector  │────▶│  Knowledge base                  │
│    encoding     │     │  • curated medical summaries     │
└─────────────────┘     │  • medical diagnosis manual (PDF)│
       │                └──────────────────────────────────┘
       ▼
┌─────────────────┐
│ NearestNeighbors│  retrieve top-k passages
│    retrieval    │
└────────┬────────┘
         │ question + retrieved context
         ▼
┌─────────────────┐
│    Generator    │  grounded answer
└─────────────────┘
```

## Approach

1. **Baseline answers** — generate responses with no retrieval, to establish what the model produces unaided.
2. **Prompt engineering** — test several prompt templates, each prepending a different instruction, to isolate the effect of phrasing.
3. **Corpus construction** — build a knowledge base from credible medical summaries plus text extracted from a medical diagnosis manual (**PyMuPDF**), then index it with **TF-IDF**.
4. **RAG question answering** — retrieve top-`k` passages and condition generation on them, sweeping `k` to measure the effect of retrieval depth.
5. **Evaluation** — score outputs on **groundedness** (were supporting documents retrieved?) and **relevance** (do expected clinical terms appear in the answer?).

## Evaluation Questions

The system was assessed on five clinical questions spanning emergency medicine, surgery,
dermatology and trauma:

| # | Question |
|:-:|----------|
| 1 | What is the protocol for managing sepsis in a critical care unit? |
| 2 | What are the common symptoms of appendicitis, and can it be cured with medicine? If not, what surgical procedure is recommended? |
| 3 | A patient exhibits patchy hair loss. What are the possible causes and treatments? |
| 4 | What are the treatment options for a traumatic brain injury (TBI)? |
| 5 | How should a broken leg be managed both in a hospital and in the wilderness? |

Questions 1–4 returned grounded, clinically coherent answers from the corpus. **Question 5
correctly returned "I don't have information on that"** — a *desirable* outcome: the wilderness
management scenario was outside the corpus, and declining to answer is the right behaviour for a
medical assistant. Refusal-when-ungrounded is a safety property, not a failure.

## Findings

| Dimension | Finding |
|-----------|---------|
| **Baseline (no retrieval)** | Keyword-driven, high-level summaries; no source citation and little clinical nuance |
| **Prompt engineering** | Prompt phrasing produced no output change in this lightweight implementation, because the fallback generator ignores instruction context. With an instruction-tuned model, prompt tuning drives style and completeness |
| **Corpus design** | Combining curated summaries with manual extracts widened retrievable coverage across both general knowledge and specific detail |
| **Retrieval depth (`k`)** | Larger `k` retrieves more context at higher compute cost; in this static-generator setup the final answer was unchanged, so `k` tuning matters most once a genuinely context-sensitive generator is in place |
| **Groundedness / relevance** | Documents were retrieved for every in-corpus query, and answers contained the expected clinical key terms |

## Honest Limitations

This is a deliberately **low-resource** implementation, and the evaluation reflects that:

- The generator is lightweight, so the prompt-engineering and `k`-ablation results measure the *pipeline*, not a frontier model's sensitivity to either.
- Groundedness and relevance are **heuristic** checks — retrieval success and keyword presence — not assessments of factual correctness.
- The corpus is small. Coverage, not reasoning, is the current bottleneck.

## Next Steps

1. **Domain-specific model** — swap in a medical instruction-tuned LLM for more nuanced generation.
2. **Expand the corpus** — more of the manual plus external clinical guidelines.
3. **Dense retrieval** — replace TF-IDF with sentence embeddings (e.g. SBERT) for semantic rather than lexical matching.
4. **Model-based evaluation** — use an LLM judge to score factual correctness and source citation, replacing the keyword heuristics.
5. **Citation surfacing** — return the supporting passage alongside each answer so clinicians can verify before acting.

## Tech Stack

`Python` · `Transformers` · `PyTorch` · `scikit-learn` (TF-IDF, NearestNeighbors) · `llama-cpp-python` · `PyMuPDF` · `pandas`

## Files

| File | Description |
|------|-------------|
| [`notebook.ipynb`](notebook.ipynb) | Full executed RAG pipeline with ablations |
| [`report.html`](report.html) | Rendered HTML report — **[view live](https://kerolous-samir.github.io/ai-ml-portfolio/07-medical-rag-assistant/report.html)** |
