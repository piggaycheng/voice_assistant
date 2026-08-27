import os
import glob
from threading import Lock

import chromadb
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

VAULT_PATH = os.getenv("RAG_VAULT_PATH", "./wiki_data/mirle_official_wiki")
DB_PATH = os.getenv("RAG_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "obsidian_notes")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_REVISION = os.getenv(
    "RAG_RERANKER_REVISION", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
)
RERANKER_DEVICE = os.getenv("RAG_RERANKER_DEVICE", EMBEDDING_DEVICE)
RERANK_CANDIDATES = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
CHROMA_BATCH_SIZE = int(os.getenv("RAG_CHROMA_BATCH_SIZE", "1000"))

app = FastAPI(title="Voice Assistant RAG API", version="1.0.0")
embed_model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
reranker = CrossEncoder(
    RERANKER_MODEL,
    device=RERANKER_DEVICE,
    revision=RERANKER_REVISION,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": "auto"},
)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
index_lock = Lock()


class IndexRequest(BaseModel):
    rebuild: bool = True


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


def load_documents():
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    documents, metadatas, ids = [], [], []
    markdown_files = glob.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True)

    for file_path in markdown_files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        splits = markdown_splitter.split_text(content)
        chunks = text_splitter.split_documents(splits)
        relative_path = os.path.relpath(file_path, VAULT_PATH)

        for chunk in chunks:
            documents.append(chunk.page_content)
            metadatas.append({"source": relative_path, **chunk.metadata})
            ids.append(f"doc_{len(ids)}")

    return markdown_files, documents, metadatas, ids


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "running",
        "service": "rag",
        "collection": COLLECTION_NAME,
        "document_count": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": EMBEDDING_DEVICE,
        "reranker_model": RERANKER_MODEL,
        "reranker_device": RERANKER_DEVICE,
    }


@app.post("/index")
def create_index(request: IndexRequest):
    global collection

    if not index_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Indexing is already in progress")

    try:
        existing_count = collection.count()
        if existing_count > 0 and not request.rebuild:
            return {"status": "skipped", "file_count": 0, "document_count": existing_count}

        markdown_files, documents, metadatas, ids = load_documents()
        if not markdown_files:
            raise HTTPException(status_code=404, detail=f"No Markdown files found in {VAULT_PATH}")
        if not documents:
            raise HTTPException(status_code=422, detail="Markdown files contained no indexable content")

        embeddings = embed_model.encode(documents, normalize_embeddings=True).tolist()
        if request.rebuild and existing_count > 0:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            collection = chroma_client.create_collection(name=COLLECTION_NAME)

        for start in range(0, len(documents), CHROMA_BATCH_SIZE):
            end = start + CHROMA_BATCH_SIZE
            collection.add(
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        return {
            "status": "indexed",
            "file_count": len(markdown_files),
            "document_count": collection.count(),
        }
    finally:
        index_lock.release()


@app.post("/query")
def query_index(request: QueryRequest):
    if index_lock.locked():
        raise HTTPException(status_code=503, detail="Indexing is in progress")

    document_count = collection.count()
    if document_count == 0:
        raise HTTPException(status_code=409, detail="Index is empty; call POST /index first")

    query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()
    candidate_count = min(max(request.top_k, RERANK_CANDIDATES), document_count)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    candidates = list(zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ))
    rerank_scores = reranker.predict(
        [(request.query, content) for content, _, _ in candidates],
        show_progress_bar=False,
    )
    matches = sorted(
        (
            {
                "content": content,
                "metadata": metadata,
                "distance": distance,
                "rerank_score": float(score),
            }
            for (content, metadata, distance), score in zip(candidates, rerank_scores)
        ),
        key=lambda match: match["rerank_score"],
        reverse=True,
    )[:request.top_k]
    return {"query": request.query, "matches": matches}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("RAG_HOST", "0.0.0.0"),
        port=int(os.getenv("RAG_PORT", "8000")),
    )