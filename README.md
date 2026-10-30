# OSHA Advanced RAG Assistant

A locally hosted **Retrieval-Augmented Generation (RAG) application** that answers construction safety questions using two official OSHA publications. It combines hybrid document retrieval, source-grounded answers, page-level citations, conversational interaction, and temporary PDF uploads.

The application provides both a Streamlit user interface and a FastAPI backend, with Docker Compose support for running both services.

## Features

### RAG and Document Retrieval
- Indexed knowledge base containing exactly two OSHA publications:
  - `OSHA_3071_Job_Hazard_Analysis.pdf`
  - `OSHA_3886_Construction_Safety.pdf`
- Persistent ChromaDB vector storage.
- Hybrid retrieval combining semantic search and IDF-weighted lexical relevance.
- Intent-aware source and page routing for known question types.
- Domain-aware query expansion.
- Optional local Ollama reranking with deterministic fallback.
- Configurable retrieval parameters, context size, relevance thresholds and timeouts.

### Grounded Answers and Citations
- Answers generated using retrieved OSHA document context.
- Source and PDF-page citations.
- Citation validation to reject references absent from retrieved context.
- Conservative abstention when sufficient supporting evidence is unavailable.
- Conversational handling for greetings and general interactions.

### PDF Upload and Question Answering
- Upload additional construction-related PDF documents.
- Extract and chunk PDF content.
- Validate document relevance before accepting uploads.
- Create isolated, temporary ChromaDB collections for uploaded documents.
- Ask questions against uploaded documents without mixing them with the original OSHA knowledge base.
- Temporary upload sessions expire after one hour of inactivity.
- Uploaded collections are cleared when the API process restarts.

### Interfaces and Deployment
- Streamlit frontend for interactive question answering.
- FastAPI backend exposing REST endpoints.
- Swagger UI for API exploration and testing.
- Docker Compose deployment for Streamlit and FastAPI.
- Ollama runs on the host machine and is accessed by Docker containers.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Vector Database | ChromaDB |
| Embedding Model | `nomic-embed-text` |
| Language Model | `llama3.2` |
| Local Model Runtime | Ollama |
| Document Processing | PyMuPDF |
| Retrieval | Semantic + lexical hybrid search |
| Containerization | Docker and Docker Compose |
| Testing | Python unittest |

## System Architecture

```text
                 User
                  |
          +-------+-------+
          |               |
      Streamlit         FastAPI
       Frontend         Backend
          |               |
          +-------+-------+
                  |
             RAG Pipeline
                  |
       +----------+----------+
       |                     |
  Hybrid Retrieval      Query Processing
       |                     |
       +----------+----------+
                  |
              ChromaDB
                  |
          Retrieved Context
                  |
              Ollama
                  |
          llama3.2 LLM
                  |
       Answer + Citations
```

**Uploaded document workflow:** Uploaded PDFs are processed separately and stored in temporary, isolated collections. They do not modify the original OSHA knowledge base.

**Docker database isolation:** The API and Streamlit services use separate persistent ChromaDB directories to avoid having both processes open the same SQLite database.

## Project Structure

```text
osha-rag/
│
├── data/
│   ├── OSHA_3071_Job_Hazard_Analysis.pdf
│   └── OSHA_3886_Construction_Safety.pdf
│
├── chroma_db/                  # Original local database
├── chroma_db_docker_api/       # Docker API database
├── chroma_db_docker_ui/        # Docker Streamlit database
│
├── app.py                      # Streamlit application
├── chat.py                     # Core RAG pipeline
├── api.py                      # FastAPI application
├── upload_rag.py               # Temporary PDF upload processing
├── page_context.py             # Page-boundary context expansion
├── visual_routes.py            # Visual question routing
│
├── ingest.py                   # Original document ingestion
├── evaluate_rag.py             # RAG evaluation
├── real_world_eval.json        # Evaluation benchmark
├── evaluation_questions.json
│
├── test_chat.py
├── test_api.py
├── test_upload_rag.py
├── test_visual_routes.py
├── test_page_boundary_expansion.py
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
│
└── README.md
```

The ChromaDB directories are local data and are not included in Git.

## Installation and Local Setup

### Prerequisites

- Python 3.11 or 3.12
- Ollama
- Docker Desktop (for Docker deployment)
- Git

### 1. Clone the repository

```powershell
git clone https://github.com/Ayushlokre/Construction---safety---rag.git

cd Construction---safety---rag
```

### 2. Create a virtual environment

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Ollama models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 5. Prepare the indexed knowledge base

The application expects an existing ChromaDB collection named `osha_documents`.

The database is not included in the Git repository. You must prepare the indexed database before running the application.

For an existing setup, preserve the original `chroma_db` directory. Do not run ingestion unnecessarily because it rebuilds the collection.

## Running Locally

Start Ollama if it is not already running:

```powershell
ollama serve
```

Start the Streamlit interface:

```powershell
streamlit run app.py
```

Alternatively, use the terminal interface:

```powershell
python chat.py
```

## Docker Deployment

Docker Compose runs the Streamlit frontend and FastAPI backend together.

### Prerequisites

- Docker Desktop installed and running.
- Ollama running on the Windows host.
- Required Ollama models downloaded.
- Existing indexed ChromaDB collection.

### 1. Prepare separate Docker database copies

Create the two database directories by copying your existing, indexed `chroma_db` directory:

```powershell
Copy-Item -Recurse chroma_db chroma_db_docker_api

Copy-Item -Recurse chroma_db chroma_db_docker_ui
```

Both copies should contain the `osha_documents` collection.

These separate directories prevent the API and Streamlit containers from concurrently opening the same persistent ChromaDB database.

Do not run ingestion to prepare these copies if the original database is already indexed.

### 2. Build Docker images

```powershell
docker compose build
```

### 3. Start the application

```powershell
docker compose up -d
```

### 4. Access the application

| Service | URL |
|---|---|
| Streamlit frontend | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger API documentation | http://localhost:8000/docs |
| API health check | http://localhost:8000/health |

### 5. Check running containers

```powershell
docker compose ps
```

### 6. View application logs

```powershell
docker compose logs -f
```

### 7. Stop the application

```powershell
docker compose down
```

Stopping Docker Compose does not delete the Docker images, persistent database directories, or downloaded Ollama models.

## FastAPI Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API root |
| GET | `/health` | Application health and configuration |
| POST | `/ask` | Ask questions using the original OSHA knowledge base |
| POST | `/uploads` | Upload a construction-related PDF |
| POST | `/uploads/{upload_id}/ask` | Ask questions about an uploaded PDF |
| DELETE | `/uploads/{upload_id}` | Delete a temporary upload session |

Interactive endpoint documentation:

http://localhost:8000/docs

### Upload Session Behaviour

- Uploaded documents are stored in isolated in-memory ChromaDB collections.
- Sessions expire after one hour of inactivity.
- Temporary uploaded collections are cleared when the API process restarts.
- The original OSHA collection is not modified by the upload workflow.

**Deployment limitation:** The API currently has no authentication. It is intended for local development and demonstration, not unrestricted public deployment.

## Configuration

The application reads configuration directly from operating-system environment variables.

Example PowerShell configuration:

```powershell
$env:RAG_RERANKER_ENABLED = "false"
$env:RAG_QUERY_EXPANSION_ENABLED = "true"
$env:RAG_OLLAMA_TIMEOUT_SECONDS = "30"
$env:RAG_RERANK_TIMEOUT_SECONDS = "12"
$env:RAG_LOG_LEVEL = "INFO"
```

The optional reranker uses `RAG_RERANK_MODEL`, defaulting to `llama3.2`.

The application also supports configurable retrieval parameters and relevance thresholds. Changes should be evaluated against the benchmark before being used.

For Docker, the database directory is configured separately for each service through `RAG_DB_DIR`.

## Testing

Run the complete regression test suite:

```powershell
python -m unittest -v test_chat.py test_api.py test_upload_rag.py test_visual_routes.py
```

The latest recorded regression run completed with:

**45 tests passed, 0 failures, 0 errors.**

The page-boundary expansion tests can also be run separately:

```powershell
python -m unittest -v test_page_boundary_expansion.py
```

## Evaluation

Run the standard evaluation:

```powershell
python evaluate_rag.py
```

Run with the optional local LLM judge:

```powershell
python evaluate_rag.py --llm-judge
```

Run the development benchmark:

```powershell
python evaluate_rag.py --benchmark real_world_eval.json --split development
```

Run the separate holdout benchmark:

```powershell
python evaluate_rag.py --benchmark real_world_eval.json --split holdout
```

The evaluation framework measures:

- Retrieval precision and recall.
- Citation precision and recall.
- Citation support.
- Abstention accuracy.
- Supported-answer rate.
- Response latency.
- Optional LLM-judge diagnostics.

The LLM judge is a diagnostic tool, not ground truth. Benchmark page labels and flagged claims should be manually reviewed. Evaluation scores do not guarantee that every generated answer is correct.

Generated evaluation files include:

```text
evaluation_results.csv
evaluation_results.json
```

These files are overwritten by subsequent evaluation runs.

## Logging and Diagnostics

Application logs are written to:

```text
logs/rag.log
```

Logs contain retrieval timing, reranker timing and fallback information.

Inspect these logs when investigating slow responses, retrieval issues or unexpected abstentions.

## Important Safety and Data Constraints

- The original indexed knowledge base contains exactly two OSHA publications.
- Tests and evaluation must not rebuild the production ChromaDB collection.
- Do not run `ingest.py` unless you explicitly intend to rebuild the original collection.
- Uploaded PDFs are processed separately from the original OSHA collection.
- Citation page numbers refer to indexed PDF page metadata, not necessarily printed page numbers.
- Answers should be checked against the cited OSHA source, particularly for safety-critical decisions.
- A successful benchmark or test run does not guarantee that every answer is correct.

## Future Improvements

- Authentication and authorization for API access.
- Production-grade deployment and persistent upload-session management.
- Improved retrieval precision and citation coverage.
- Expanded manually verified evaluation datasets.
- Monitoring and performance optimization.

---

**Repository:** [OSHA Advanced RAG Assistant](https://github.com/Ayushlokre/Construction---safety---rag)

**Author:** Ayush Lokre