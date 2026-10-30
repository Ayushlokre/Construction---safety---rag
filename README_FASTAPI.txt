OSHA FastAPI add-on

Files:
- api.py: replaces the current API wrapper; preserves GET /, GET /health and POST /ask.
- requirements.txt: existing project requirements plus FastAPI, Uvicorn, python-multipart and httpx.
- test_api.py: focused endpoint tests using mocks (imports the included api.py).

Keep chat.py, upload_rag.py, page_context.py, data/, and chroma_db/ unchanged.
Do not run ingest.py.

Install dependencies in the existing virtual environment:
    pip install -r requirements.txt

Run the API:
    uvicorn api:app --host 127.0.0.1 --port 8000

Open interactive API docs:
    http://127.0.0.1:8000/docs

Temporary upload API:
    POST /uploads                 multipart field: file
    POST /uploads/{upload_id}/ask JSON: {"question": "..."}
    DELETE /uploads/{upload_id}

Uploaded files are held in process memory, are not written to the persistent OSHA
ChromaDB, expire after one hour of inactivity (checked on API requests), and can
be explicitly deleted. Restarting the API process clears all temporary uploads.
This is a local-development API; it has no authentication and should not be
exposed publicly as-is.
