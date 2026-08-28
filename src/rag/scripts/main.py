import os
import glob
from threading import Lock

import chromadb
import jieba
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
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
BM25_CANDIDATES = int(os.getenv("RAG_BM25_CANDIDATES", "20"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
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
bm25_index = None
bm25_documents = []
bm25_metadatas = []
bm25_ids = []


class IndexRequest(BaseModel):
    rebuild: bool = True


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


def tokenize_for_bm25(text):
    return [token.lower() for token in jieba.lcut(text) if token.strip()]


def build_bm25_index(documents, metadatas, ids):
    global bm25_index, bm25_documents, bm25_metadatas, bm25_ids

    bm25_documents = documents
    bm25_metadatas = metadatas
    bm25_ids = ids
    bm25_index = BM25Okapi([tokenize_for_bm25(document) for document in documents]) if documents else None


def load_bm25_from_collection():
    stored = collection.get(include=["documents", "metadatas"])
    build_bm25_index(stored["documents"], stored["metadatas"], stored["ids"])


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


@app.on_event("startup")
def load_lexical_index():
    load_bm25_from_collection()


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
        "bm25_document_count": len(bm25_documents),
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
        build_bm25_index(documents, metadatas, ids)
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
    if len(bm25_documents) != document_count:
        load_bm25_from_collection()

    query_embedding = embed_model.encode(request.query, normalize_embeddings=True).tolist()
    candidate_count = min(max(request.top_k, RERANK_CANDIDATES), document_count)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    vector_candidates = list(zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ))
    candidates_by_id = {
        document_id: {
            "content": content,
            "metadata": metadata,
            "distance": distance,
            "bm25_score": None,
            "rrf_score": 1.0 / (RRF_K + rank),
        }
        for rank, (document_id, content, metadata, distance) in enumerate(vector_candidates, start=1)
    }

    if bm25_index is not None:
        bm25_scores = bm25_index.get_scores(tokenize_for_bm25(request.query))
        bm25_count = min(max(request.top_k, BM25_CANDIDATES), document_count)
        bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_count]
        for rank, index in enumerate(bm25_indices, start=1):
            score = float(bm25_scores[index])
            if score <= 0:
                continue
            document_id = bm25_ids[index]
            candidate = candidates_by_id.setdefault(
                document_id,
                {
                    "content": bm25_documents[index],
                    "metadata": bm25_metadatas[index],
                    "distance": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                },
            )
            candidate["bm25_score"] = score
            candidate["rrf_score"] += 1.0 / (RRF_K + rank)

    candidates = sorted(
        candidates_by_id.values(), key=lambda candidate: candidate["rrf_score"], reverse=True
    )[:candidate_count]
    rerank_scores = reranker.predict(
        [(request.query, candidate["content"]) for candidate in candidates],
        show_progress_bar=False,
    )
    matches = sorted(
        (
            {
                **candidate,
                "rerank_score": float(score),
            }
            for candidate, score in zip(candidates, rerank_scores)
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