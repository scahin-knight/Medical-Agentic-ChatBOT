
import argparse
import hashlib
import json
import time
from pathlib import Path

import pymupdf

from app.knowledge import (
    ensure_index,
    embed_documents,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    OLLAMA_INDEX_NAME,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 100

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CHECKPOINT_FILE = Path("ollama_ingestion_checkpoint.jsonl")
PAGE_CHECKPOINT_FILE = Path("ollama_page_checkpoint.json")

DEFAULT_PDF = (
    "data/"
    "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-"
    "staibabussalamsula.ac_.id_.pdf"
)


# --------------------------------------------------
# CHUNKING AND IDs
# --------------------------------------------------

def chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        content = text[i:i + chunk_size]

        if content.strip():
            yield content


def make_chunk_id(pdf_name, page_number, part, content):
    key = f"{pdf_name}:{page_number}:{part}:{content}"

    return hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# CHUNK CHECKPOINT
# --------------------------------------------------

def load_checkpoint():
    completed_ids = set()

    if CHECKPOINT_FILE.exists():
        with CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    item = json.loads(line)
                    completed_ids.add(item["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"Previously completed chunks: {len(completed_ids)}")

    return completed_ids


def save_chunk_checkpoint(chunk_ids):
    with CHECKPOINT_FILE.open("a", encoding="utf-8") as file:
        for chunk_id in chunk_ids:
            file.write(json.dumps({"id": chunk_id}) + "\n")

        file.flush()


# --------------------------------------------------
# PAGE CHECKPOINT
# --------------------------------------------------

def save_page_checkpoint(pdf_name, completed_page):
    data = {
        "pdf": pdf_name,
        "completed_page": completed_page
    }

    temp_file = PAGE_CHECKPOINT_FILE.with_suffix(".tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(data, file)
        file.flush()

    temp_file.replace(PAGE_CHECKPOINT_FILE)


def load_page_checkpoint(pdf_name):
    if not PAGE_CHECKPOINT_FILE.exists():
        return None

    try:
        with PAGE_CHECKPOINT_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if data.get("pdf") != pdf_name:
            print("Page checkpoint belongs to another PDF.")
            return None

        return int(data.get("completed_page", 0))

    except (json.JSONDecodeError, ValueError, OSError):
        print("Invalid page checkpoint. Reconstructing progress.")
        return None


# --------------------------------------------------
# ONE-TIME PROGRESS RECONSTRUCTION
# --------------------------------------------------

def reconstruct_completed_page(doc, pdf_name, completed_ids):
    """
    Reconstruct the last fully uploaded page from the old
    chunk-ID checkpoint. This does not generate embeddings
    or upload vectors.
    """

    print("\nNo page checkpoint found.")
    print("Checking existing chunk IDs to find your progress...")
    print("This is a one-time verification scan.")

    last_completed_page = 0

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text()

        page_complete = True

        if page_text.strip():
            for part, content in enumerate(chunks(page_text), start=1):
                chunk_id = make_chunk_id(
                    pdf_name,
                    page_number,
                    part,
                    content
                )

                if chunk_id not in completed_ids:
                    page_complete = False
                    break

        if not page_complete:
            print(f"\nFirst incomplete page: {page_number}")
            break

        last_completed_page = page_number

        if page_number % 50 == 0:
            print(
                f"Verified completed pages: "
                f"{page_number}/{len(doc)}"
            )

    save_page_checkpoint(pdf_name, last_completed_page)

    print(
        f"\nLast fully completed page: "
        f"{last_completed_page}"
    )

    return last_completed_page


# --------------------------------------------------
# EMBEDDING AND PINECONE UPLOAD
# --------------------------------------------------

def upload_batch(index, batch, completed_ids):
    if not batch:
        return 0

    start_time = time.perf_counter()

    texts = [item["text"] for item in batch]

    # Generate local Ollama embeddings.
    embeddings = embed_documents(texts)

    vectors = []

    for item, embedding in zip(batch, embeddings):
        vectors.append({
            "id": item["id"],
            "values": embedding,
            "metadata": item["metadata"]
        })

    # Upload to Pinecone.
    index.upsert(vectors=vectors)

    # Record IDs only after successful upload.
    uploaded_ids = [item["id"] for item in batch]

    save_chunk_checkpoint(uploaded_ids)
    completed_ids.update(uploaded_ids)

    elapsed = time.perf_counter() - start_time

    print(
        f"Uploaded {len(vectors)} chunks | "
        f"Dimension: {EMBEDDING_DIMENSION} | "
        f"Time: {elapsed:.2f}s"
    )

    return len(vectors)


# --------------------------------------------------
# MAIN INGESTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Resume medical PDF ingestion using Ollama."
    )

    parser.add_argument(
        "--pdf",
        type=str,
        default=DEFAULT_PDF
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pdf_name = pdf_path.name

    # Load existing uploaded chunk IDs.
    completed_ids = load_checkpoint()

    print("\nConnecting to Pinecone...")

    index = ensure_index()

    print(f"Ollama model: {EMBEDDING_MODEL}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSION}")
    print(f"Pinecone index: {OLLAMA_INDEX_NAME}")
    print(f"Batch size: {BATCH_SIZE}")

    uploaded_this_run = 0

    with pymupdf.open(pdf_path) as doc:

        total_pages = len(doc)

        # First try the saved page checkpoint.
        last_completed_page = load_page_checkpoint(pdf_name)

        # If absent, reconstruct it from existing chunk IDs.
        if last_completed_page is None:
            last_completed_page = reconstruct_completed_page(
                doc,
                pdf_name,
                completed_ids
            )

        if last_completed_page >= total_pages:
            print("\nAll PDF pages are already completed!")
            return

        start_page = last_completed_page + 1

        print("\n================================")
        print("       RESUMING INGESTION")
        print("================================")

        print(f"Total PDF pages: {total_pages}")
        print(f"Last completed page: {last_completed_page}")
        print(f"Starting from page: {start_page}")

        # Jump directly to the next incomplete page.
        for page_index in range(start_page - 1, total_pages):

            page_number = page_index + 1
            page = doc[page_index]

            page_text = page.get_text()

            batch = []

            if page_text.strip():

                for part, content in enumerate(chunks(page_text), start=1):

                    chunk_id = make_chunk_id(
                        pdf_name,
                        page_number,
                        part,
                        content
                    )

                    # Skip any chunks already uploaded.
                    if chunk_id in completed_ids:
                        continue

                    metadata = {
                        "page": page_number,
                        "text": content,
                        "file": pdf_name
                    }

                    batch.append({
                        "id": chunk_id,
                        "text": content,
                        "metadata": metadata
                    })

                    # Upload full batches.
                    if len(batch) >= BATCH_SIZE:
                        uploaded_this_run += upload_batch(
                            index,
                            batch,
                            completed_ids
                        )

                        batch = []

            # Upload remaining chunks for this page.
            if batch:
                uploaded_this_run += upload_batch(
                    index,
                    batch,
                    completed_ids
                )

            # Mark page complete only after all its chunks
            # have been successfully uploaded.
            save_page_checkpoint(pdf_name, page_number)

            if page_number % 10 == 0 or page_number == total_pages:
                print(
                    f"Progress: {page_number}/{total_pages} pages"
                )

    print("\n================================")
    print("       INGESTION COMPLETE")
    print("================================")

    print(f"PDF: {pdf_name}")
    print(f"Total pages: {total_pages}")
    print(f"Uploaded this run: {uploaded_this_run}")
    print(f"Completed chunks: {len(completed_ids)}")
    print(f"Pinecone index: {OLLAMA_INDEX_NAME}")

    print("\nAll available pages have been processed!")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nIngestion interrupted.")
        print("Successfully uploaded batches remain checkpointed.")
        print("Restart python ingest.py to resume.")

    except Exception as error:
        print(f"\nINGESTION ERROR: {error}")
        raise
