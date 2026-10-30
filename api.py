"""FastAPI wrapper around the existing OSHA RAG pipeline.

Run locally with:
    uvicorn api:app --host 127.0.0.1 --port 8000

The original OSHA ChromaDB is read-only here. Uploaded PDFs use separate,
in-memory Chroma clients and are never added to the persistent OSHA collection.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import chat as rag
import upload_rag

LOGGER = logging.getLogger("osha_api")
UPLOAD_TTL_SECONDS = 60 * 60  # In-memory uploads expire after one hour of inactivity.

app = FastAPI(
    title="OSHA Safety Assistant API",
    version="1.1.0",
    description=(
        "Document-grounded Q&A over the indexed OSHA PDFs, with temporary "
        "construction-PDF upload support."
    ),
)

SOURCE_RE = re.compile(
    r"\[Source:\s*([^,\]]+),\s*PDF page\s*(\d+)\]",
    flags=re.IGNORECASE,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class UploadedAskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class SourceRef(BaseModel):
    filename: str
    pdf_page: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    chunks: int
    message: str


@dataclass
class UploadSession:
    client: object
    collection: object
    filename: str
    last_access: float


# Process-local, temporary upload registry. It is intentionally not persisted.
UPLOAD_SESSIONS: dict[str, UploadSession] = {}


def extract_sources(answer: str) -> list[SourceRef]:
    sources: list[SourceRef] = []
    seen: set[tuple[str, int]] = set()
    for match in SOURCE_RE.finditer(answer or ""):
        item = (match.group(1).strip(), int(match.group(2)))
        if item not in seen:
            seen.add(item)
            sources.append(SourceRef(filename=item[0], pdf_page=item[1]))
    return sources


def _delete_upload(upload_id: str) -> bool:
    session = UPLOAD_SESSIONS.pop(upload_id, None)
    if session is None:
        return False
    try:
        session.client.delete_collection(name=session.collection.name)
    except Exception:
        # Dropping the only references still releases this in-memory client when
        # the session object is collected. Log, but don't expose internals.
        LOGGER.exception("Could not explicitly delete temporary upload collection")
    return True


def _expire_uploads() -> None:
    now = time.monotonic()
    expired = [
        upload_id
        for upload_id, session in UPLOAD_SESSIONS.items()
        if now - session.last_access > UPLOAD_TTL_SECONDS
    ]
    for upload_id in expired:
        _delete_upload(upload_id)


def _get_upload(upload_id: str) -> UploadSession:
    _expire_uploads()
    session = UPLOAD_SESSIONS.get(upload_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Uploaded document was not found or has expired. Upload it again.",
        )
    session.last_access = time.monotonic()
    return session


@lru_cache(maxsize=1)
def get_collection():
    try:
        return rag.load_collection()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base unavailable: {exc}",
        ) from exc


@app.get("/")
def root():
    return {
        "name": "OSHA Safety Assistant API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "original_documents": "POST /ask",
            "upload_pdf": "POST /uploads",
            "uploaded_document_question": "POST /uploads/{upload_id}/ask",
            "remove_upload": "DELETE /uploads/{upload_id}",
        },
    }


@app.get("/health")
def health():
    collection = get_collection()
    _expire_uploads()
    return {
        "status": "ok",
        "collection": rag.COLLECTION_NAME,
        "chunks": collection.count(),
        "embedding_model": rag.EMBED_MODEL,
        "answer_model": rag.LLM_MODEL,
        "active_temporary_uploads": len(UPLOAD_SESSIONS),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    collection = get_collection()
    try:
        if rag.is_greeting(question):
            answer = rag.greeting_response(question)
        else:
            context = rag.retrieve_documents(question, collection)
            answer = rag.generate_answer(question, context)
    except Exception as exc:
        rag.LOGGER.exception("API question failed")
        raise HTTPException(status_code=502, detail="RAG request failed. Check the API logs.") from exc

    return AskResponse(
        question=question,
        answer=answer,
        sources=extract_sources(answer),
    )


@app.post("/uploads", response_model=UploadResponse, status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    _expire_uploads()
    filename = file.filename or "uploaded.pdf"
    try:
        file_bytes = await file.read(upload_rag.MAX_FILE_BYTES + 1)
    finally:
        await file.close()

    if len(file_bytes) > upload_rag.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="The PDF exceeds the 20 MB upload limit.")

    try:
        client, collection, chunk_count, message = upload_rag.create_session_collection(
            filename, file_bytes, rag.EMBED_MODEL
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Temporary PDF upload/indexing failed")
        raise HTTPException(
            status_code=502,
            detail="The PDF could not be indexed. Check that Ollama and the embedding model are available.",
        ) from exc

    upload_id = uuid.uuid4().hex
    UPLOAD_SESSIONS[upload_id] = UploadSession(
        client=client,
        collection=collection,
        filename=filename,
        last_access=time.monotonic(),
    )
    return UploadResponse(
        upload_id=upload_id,
        filename=filename,
        chunks=chunk_count,
        message=message,
    )


@app.post("/uploads/{upload_id}/ask", response_model=AskResponse)
def ask_uploaded(upload_id: str, request: UploadedAskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    session = _get_upload(upload_id)
    try:
        if rag.is_greeting(question):
            answer = rag.greeting_response(question)
        else:
            context = upload_rag.retrieve_uploaded_context(
                question, session.collection, rag.EMBED_MODEL
            )
            answer = upload_rag.answer_uploaded_question(
                question, context, rag.LLM_MODEL
            )
    except Exception as exc:
        LOGGER.exception("Uploaded-document question failed")
        raise HTTPException(
            status_code=502,
            detail="Uploaded-document question failed. Check the API logs.",
        ) from exc

    return AskResponse(
        question=question,
        answer=answer,
        sources=extract_sources(answer),
    )


@app.delete("/uploads/{upload_id}")
def remove_upload(upload_id: str):
    _expire_uploads()
    if not _delete_upload(upload_id):
        raise HTTPException(status_code=404, detail="Uploaded document was not found or has expired.")
    return {"status": "deleted", "upload_id": upload_id}
