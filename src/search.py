"""
Hybrid search: dense vector retrieval fused with BM25 sparse retrieval
via Reciprocal Rank Fusion (RRF). Outperforms either method alone,
especially on queries with specific terminology or acronyms.
"""

import math
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

# RRF constant — 60 is standard from the original paper
_RRF_K = 60


class HybridSearchEngine:
    def __init__(
        self,
        openai_api_key: str,
        chroma_dir: str = "./data/chroma",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._openai = OpenAI(api_key=openai_api_key)
        self._embedding_model = embedding_model

        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_index: dict[str, dict] = {}

    def index(self, documents: list[dict]) -> int:
        """Index documents for both dense and BM25 retrieval."""
        if not documents:
            return 0

        texts = [d["text"] for d in documents]
        ids = [d["id"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]

        embeddings = self._embed(texts)
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

        # Build BM25 index
        corpus = [t.lower().split() for t in texts]
        df: dict[str, int] = {}
        for tokens in corpus:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        N = len(corpus)
        avg_dl = sum(len(t) for t in corpus) / N if N else 1

        for doc, tokens, doc_id in zip(documents, corpus, ids):
            tf: dict[str, int] = {}
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1
            self._bm25_index[doc_id] = {
                "tf": tf,
                "dl": len(tokens),
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            }

        self._bm25_df = df
        self._bm25_N = N
        self._bm25_avg_dl = avg_dl
        return len(documents)

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Run dense + BM25 retrieval independently, then fuse via RRF.
        Returns top-k results with combined score.
        """
        fetch_k = min(k * 3, 20)  # over-fetch for better fusion

        dense_hits = self._dense_search(query, fetch_k)
        sparse_hits = self._bm25_search(query, fetch_k)

        fused = self._rrf_fuse(dense_hits, sparse_hits)
        return fused[:k]

    def _dense_search(self, query: str, k: int) -> list[dict]:
        embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "metadata": meta,
                "dense_score": 1.0 - dist,
            })
        return hits

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        if not self._bm25_index:
            return []

        terms = query.lower().split()
        N = self._bm25_N
        avg_dl = self._bm25_avg_dl
        b, k1 = 0.75, 1.5

        scores: dict[str, float] = {}
        for term in terms:
            idf = math.log((N - self._bm25_df.get(term, 0) + 0.5) /
                           (self._bm25_df.get(term, 0) + 0.5) + 1)
            for doc_id, entry in self._bm25_index.items():
                tf = entry["tf"].get(term, 0)
                dl = entry["dl"]
                tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        hits = []
        for doc_id, score in ranked:
            entry = self._bm25_index[doc_id]
            hits.append({
                "text": entry["text"],
                "metadata": entry["metadata"],
                "bm25_score": score,
            })
        return hits

    @staticmethod
    def _rrf_fuse(dense: list[dict], sparse: list[dict]) -> list[dict]:
        """Reciprocal Rank Fusion — combines rankings without score normalization."""
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, hit in enumerate(dense):
            key = hit["text"][:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            payloads[key] = {**hit, "retrieval": "dense"}

        for rank, hit in enumerate(sparse):
            key = hit["text"][:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key in payloads:
                payloads[key]["retrieval"] = "hybrid"
            else:
                payloads[key] = {**hit, "retrieval": "sparse"}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, score in ranked:
            entry = payloads[key]
            results.append({
                "text": entry["text"],
                "metadata": entry.get("metadata", {}),
                "score": round(score, 6),
                "retrieval": entry.get("retrieval", "unknown"),
            })
        return results

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._openai.embeddings.create(
            model=self._embedding_model, input=texts
        )
        return [item.embedding for item in response.data]

    @property
    def document_count(self) -> int:
        return self._collection.count()
