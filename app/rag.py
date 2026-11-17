"""Componentes compartilhados do RAG: modelo de embedding + coleção ChromaDB + busca top-k."""
import os
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(__file__), "..", "chroma_data"))
COLLECTION = "academico"
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))  # similaridade mínima (cosseno) para considerar evidência


@lru_cache
def embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return embedder().encode(texts, normalize_embeddings=True).tolist()


@lru_cache
def client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def collection():
    return client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def search(query: str, k: int = 5, topic: str | None = None) -> list[dict]:
    """Busca semântica top-k; filtra por topic (metadado) quando informado."""
    res = collection().query(
        query_embeddings=embed([query]),
        n_results=k,
        where={"topic": topic} if topic else None,
    )
    out = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = round(1 - dist, 3)
        if score >= MIN_SCORE:
            out.append({"text": text, "source": meta["source"], "section": meta.get("section", ""),
                        "topic": meta["topic"], "score": score})
    return out
