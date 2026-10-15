# 🦺 OSHA Safety Assistant — Local RAG Chatbot

A local Retrieval-Augmented Generation (RAG) application that answers questions from two OSHA publications:

- `OSHA_3071_Job_Hazard_Analysis.pdf`
- `OSHA_3886_Construction_Safety.pdf`

The application uses local embeddings, a persistent vector database, and a local language model. It returns answers grounded in retrieved document passages and includes source filename/PDF page citations.

> **Safety note:** This is a document Q&A and learning project, not a substitute for official OSHA standards, site-specific procedures, or a qualified safety professional. Verify safety-critical answers against the original publication and applicable regulations.

## Features

- Extracts text from PDFs and preserves source filename and PDF page metadata.
- Splits document text into chunks for retrieval.
- Creates local embeddings with Ollama `nomic-embed-text`.
- Stores and searches chunks in ChromaDB.
- Uses semantic retrieval plus keyword matching.
- Generates answers locally with Ollama `llama3.2`.
- Displays source citations in answers and a Streamlit chat interface.
- Includes targeted handlers for selected frequently tested JHA lists.
- Abstains for selected unsupported machinery questions.

## Architecture

```text
OSHA PDF files
    |
    v
PyMuPDF text extraction
    |
    v
Recursive text chunking
    |
    v
Ollama embedding model: nomic-embed-text
    |
    v
ChromaDB persistent collection: osha_documents
    |
    v
User question -> query embedding + keyword matching
    |
    v
Relevant chunks/pages assembled as context
    |
    v
Ollama LLM: llama3.2
    |
    v
Answer with source citations (or safe abstention)
    |
    v
Streamlit chat interface
```

## Technology stack

- Python
- Streamlit
- Ollama
- `nomic-embed-text` embeddings
- `llama3.2` language model
- ChromaDB
- PyMuPDF
- LangChain text splitters

## Project structure

```text
osha-rag/
├── data/
│   ├── OSHA_3071_Job_Hazard_Analysis.pdf
│   └── OSHA_3886_Construction_Safety.pdf
├── chroma_db/                 # Generated local vector database; do not commit
├── .venv/                     # Python virtual environment; do not commit
├── app.py                     # Streamlit interface
├── chat.py                    # Terminal chatbot
├── ingest.py                  # PDF ingestion and indexing
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

1. Python installed.
2. Ollama installed and running.
3. The two OSHA PDFs placed in the `data/` folder.
4. Models downloaded locally:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Setup on Windows PowerShell

Open PowerShell in the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, you can run the venv's Python directly with `.\.venv\Scripts\python.exe`.

Index the PDFs:

```powershell
python ingest.py
```

Start the terminal chatbot:

```powershell
python chat.py
```

Or start the web interface:

```powershell
streamlit run app.py
```

Streamlit usually serves the app at `http://localhost:8501`.

## Example questions

- What is a Job Hazard Analysis (JHA)?
- What are the five criteria for selecting jobs to be analyzed using a JHA?
- What questions should be asked to identify workplace hazards during a JHA?
- How should hazards identified during a JHA be corrected or prevented?
- What are the seven core elements of safety and health programs in construction?
- What are the five levels in the hierarchy of controls?
- What is the safest operating speed for an industrial press machine?

The final question should be refused when the indexed documents do not establish a safe speed for the specific machine.

## Evaluation

The project has been manually tested on selected questions, including:

- JHA definition and steps.
- Five criteria for selecting JHA jobs.
- Hazard-identification questions.
- JHA hazard-correction guidance.
- Seven core elements of construction safety programs.
- Five levels in the hierarchy of controls.
- Abstention for an unsupported industrial press speed.

These are selected functional checks, not a comprehensive accuracy guarantee. Use the evaluation checklist to record further results. Check that each citation points to a page that actually supports the associated claim.

## Known limitations

- Retrieval can include related but unnecessary passages.
- Citation formatting and answer completeness should be checked for new question types.
- Some common questions use targeted handlers; not every question has a deterministic answer.
- Scanned PDFs with no extractable text may require OCR.
- The local model can still make mistakes. Do not use this application as the sole basis for safety-critical decisions.
- The app requires Ollama and the local models to be available.

## Privacy

The intended workflow runs embeddings and generation locally through Ollama. Keep the PDFs and local database out of public repositories unless you have confirmed the documents' redistribution terms.

## Future improvements

- Automated retrieval and answer evaluation.
- Better reranking and relevance thresholds.
- Unit tests for citation extraction and deterministic handlers.
- Source cards with page references.
- Better out-of-scope question detection.
- Optional OCR for scanned documents.
- Packaging and deployment documentation.

## License

Add a license only after deciding what license applies to your original code and confirming the rights and terms for any bundled source documents. Do not redistribute OSHA PDFs without checking their applicable terms.
