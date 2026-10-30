"""Page-aware context expansion for PDF RAG retrieval."""

def expand_page_boundary_items(selected_items, all_ids, all_documents, all_metadatas):
    """Add the immediately adjacent page's boundary chunk when a selected
    chunk touches the start/end of its PDF page.

    This handles passages split by PDF page breaks without re-ingestion.
    Expansion is restricted to the same source PDF and preserves each
    chunk's original page/chunk metadata for citation accuracy.
    """
    page_chunks = {}
    by_id = {}

    for doc_id, doc, metadata in zip(all_ids, all_documents, all_metadatas):
        if not doc or not metadata:
            continue
        source = metadata.get("source")
        try:
            page = int(metadata.get("page"))
            chunk_number = int(metadata.get("chunk"))
        except (TypeError, ValueError):
            continue
        if not source:
            continue

        item = {
            "id": doc_id,
            "text": doc,
            "metadata": metadata,
            "keyword_score": 0.0,
        }
        page_chunks.setdefault((source, page), []).append(item)
        by_id[doc_id] = item

    for items in page_chunks.values():
        items.sort(key=lambda item: int(item["metadata"].get("chunk", 0) or 0))

    expanded = {item["id"]: item for item in selected_items if item.get("id") is not None}

    for selected in selected_items:
        metadata = selected.get("metadata") or {}
        source = metadata.get("source")
        try:
            page = int(metadata.get("page"))
            chunk_number = int(metadata.get("chunk"))
        except (TypeError, ValueError):
            continue

        current_page_chunks = page_chunks.get((source, page), [])
        if not current_page_chunks:
            continue

        chunk_numbers = [
            int(item["metadata"].get("chunk", 0) or 0)
            for item in current_page_chunks
        ]
        first_chunk = min(chunk_numbers)
        last_chunk = max(chunk_numbers)

        # A selected last chunk may continue at the beginning of the next PDF page.
        if chunk_number == last_chunk:
            next_page_chunks = page_chunks.get((source, page + 1), [])
            if next_page_chunks:
                first_neighbor = next_page_chunks[0]
                expanded[first_neighbor["id"]] = first_neighbor

        # A selected first chunk may continue from the end of the previous PDF page.
        if chunk_number == first_chunk:
            previous_page_chunks = page_chunks.get((source, page - 1), [])
            if previous_page_chunks:
                last_neighbor = previous_page_chunks[-1]
                expanded[last_neighbor["id"]] = last_neighbor

    return sorted(
        expanded.values(),
        key=lambda item: (
            (item.get("metadata") or {}).get("source", ""),
            int((item.get("metadata") or {}).get("page", 0) or 0),
            int((item.get("metadata") or {}).get("chunk", 0) or 0),
        ),
    )
