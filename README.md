# OSHA Advanced Local RAG

A local retrieval-augmented question-answering application over exactly two indexed OSHA PDFs:

- `OSHA_3071_Job_Hazard_Analysis.pdf`
- `OSHA_3886_Construction_Safety.pdf`

Stack: Python, ChromaDB, Ollama, `nomic-embed-text`, `llama3.2`, and Streamlit.

## Current capabilities

- Persistent ChromaDB retrieval; evaluation and chat do **not** ingest or rebuild the database.
- Intent-aware source/page routing for known source-grounded question types.
- Hybrid retrieval combining semantic search and IDF-weighted lexical relevance.
- Deterministic, domain-aware query expansion with a configurable toggle.
- Optional local Ollama reranking of the hybrid candidate shortlist, with deterministic fallback if the reranker errors or returns invalid JSON.
- Configurable top-k values, context size, reranking, and relevance thresholds.
- Source/page citations and a final citation guard that rejects citations absent from retrieved context.
- Conservative abstention for unsupported, hazard-specific questions.
- Streamlit and terminal interfaces share the same retrieval and answer pipeline.
- File logging to `logs/rag.log`.
- Regression tests and a benchmark evaluation script with retrieval, citation, abstention, latency, and optional local LLM-judge diagnostics.

## Setup

Use Python 3.11 or 3.12 if possible, then activate your existing virtual environment. Install dependencies only if needed:

```powershell
pip install -r requirements.txt
```

Ensure Ollama is running and these models are installed:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

The optional local reranker uses `RAG_RERANK_MODEL`, defaulting to `llama3.2`, so no extra model is required.

## Run the application

Streamlit UI:

```powershell
streamlit run app.py
```

Terminal UI:

```powershell
python chat.py
```

## Tests and evaluation

```powershell
python -m unittest -v test_chat.py
python evaluate_rag.py
python evaluate_rag.py --llm-judge

# Run the 100-question real-world draft benchmark (80 development questions)
python evaluate_rag.py --benchmark real_world_eval.json --split development

# Final check on the separate 20-question holdout set
python evaluate_rag.py --benchmark real_world_eval.json --split holdout
```

The local LLM judge is a diagnostic, not ground truth. Review its flagged claims manually. Expand the benchmark with manually verified expected pages as new query patterns are introduced.

## Runtime safeguards and diagnostics

Ollama model calls now have configurable timeouts so a stalled local model can fall back
instead of blocking a request indefinitely. Reranker failures/timeouts fall back to the
existing hybrid ranking; no safety threshold is automatically lowered.

```powershell
$env:RAG_OLLAMA_TIMEOUT_SECONDS="30"
$env:RAG_RERANK_TIMEOUT_SECONDS="12"
$env:RAG_LOG_LEVEL="INFO"
```

The log now records targeted/general retrieval duration and reranker duration/fallback.
Inspect `logs/rag.log` when a query is unexpectedly slow or abstains. Benchmark page labels
should be manually audited before interpreting retrieval precision/recall as correctness.

## Configuration

Copy `.env.example` as a reference. The application reads OS environment variables directly; it does not automatically parse a `.env` file. For a one-session PowerShell configuration, for example:

```powershell
$env:RAG_RERANKER_ENABLED = "true"
$env:RAG_QUERY_EXPANSION_ENABLED = "true"
$env:RAG_MIN_KEYWORD_RELEVANCE = "0.20"
$env:RAG_MIN_COMBINED_RELEVANCE = "0.0"
```

Tune thresholds only after reviewing benchmark regressions. Higher thresholds can reduce irrelevant chunks but may also exclude relevant paraphrased passages. Local reranking can improve ordering but adds latency; if the reranker fails, hybrid scores are used instead.

## Important constraints

- Keep the existing `chroma_db` folder.
- Do **not** run `ingest.py` as part of tests or evaluation.
- Do not add extra documents without explicitly changing the project scope.
- PDF page references in citations refer to the page metadata indexed in ChromaDB, not necessarily the printed page number in the PDF.
- A good benchmark score does not guarantee that every answer is correct. Safety-related answers require source verification.

## Evaluation outputs

`evaluate_rag.py` writes `evaluation_results.csv` and `evaluation_results.json` into the project directory. These are generated results and are overwritten by subsequent runs.
