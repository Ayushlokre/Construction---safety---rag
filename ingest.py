from pathlib import Path

import chromadb
import ollama
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "osha_documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 32


# ---------------- PDF EXTRACTION ----------------

def load_pdf_pages():
    """Extract text from every PDF page and preserve source/page metadata."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {DATA_DIR}")

    pages = []

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        with pymupdf.open(pdf_path) as pdf:
            total_pages = pdf.page_count
            extracted_pages = 0

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    print(f"  Warning: no extractable text on PDF page {page_number}")
                    continue

                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                })
                extracted_pages += 1

        print(f"  Total PDF pages: {total_pages}")
        print(f"  Pages with extractable text: {extracted_pages}")

    if not pages:
        raise ValueError(
            "No PDF text was extracted. If the PDFs are scanned images, OCR is required."
        )

    return pages


# ---------------- CHUNKING ----------------

def create_chunks(pages):
    """Split each page independently so citations retain the original PDF page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for page in pages:
        for chunk_number, chunk_text in enumerate(splitter.split_text(page["text"]), start=1):
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": chunk_number,
                })
    return chunks


# ---------------- VECTOR DATABASE ----------------

def rebuild_collection(chunks):
    """Rebuild the collection from scratch to avoid stale or duplicate chunks."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("\nRemoved previous collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "OSHA Construction Safety and Job Hazard Analysis documents"},
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [item["text"] for item in batch]

        response = ollama.embed(model=EMBED_MODEL, input=texts)
        embeddings = response.embeddings
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding count does not match the number of chunks.")

        ids = [f"chunk_{start + i:06d}" for i in range(len(batch))]
        metadatas = [{
            "source": item["source"],
            "page": item["page"],
            "chunk": item["chunk"],
        } for item in batch]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"Indexed {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")

    return collection


def main():
    print("Starting OSHA RAG ingestion...")
    print(f"PDF directory: {DATA_DIR}")
    print(f"Embedding model: {EMBED_MODEL}")

    try:
        # Confirm the embedding model is available before rebuilding the database.
        ollama.embed(model=EMBED_MODEL, input="OSHA safety document indexing test")

        pages = load_pdf_pages()
        chunks = create_chunks(pages)

        print(f"\nPages with extractable text: {len(pages)}")
        print(f"Total chunks created: {len(chunks)}")

        if not chunks:
            raise ValueError("No chunks were created.")

        # Show the files that are actually being indexed.
        source_counts = {}
        for page in pages:
            source_counts[page["source"]] = source_counts.get(page["source"], 0) + 1
        print("\nPDF files found and extracted:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} pages")

        collection = rebuild_collection(chunks)
        print("\nIngestion completed successfully.")
        print(f"Database: {DB_DIR}")
        print(f"Stored chunks: {collection.count()}")

    except Exception as exc:
        print(f"\nIngestion failed: {exc}")
        print(
            f"Check Ollama is running, '{EMBED_MODEL}' is installed, "
            "and both PDFs are inside the data/ folder."
        )
        raise


if __name__ == "__main__":
    main()
