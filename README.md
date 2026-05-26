# AI Document Search

Hybrid semantic + BM25 search API. Dense vector retrieval fused with sparse keyword retrieval via Reciprocal Rank Fusion — consistently outperforms either approach alone, especially on technical queries with specific terminology.

Drop-in REST API. Index documents, search with natural language, get ranked results with confidence scores.

## Why Hybrid Search

| Method | Strength | Weakness |
|--------|----------|----------|
| BM25 (keyword) | Exact term matching, fast | Misses paraphrase / synonyms |
| Dense (semantic) | Understands meaning | Struggles with rare terms / acronyms |
| **Hybrid (RRF)** | **Both** | Slightly more infrastructure |

Reciprocal Rank Fusion combines the ranked lists without score normalization — no tuning of score scales needed.

## Demo

**Indexing 500 enterprise documents, then searching:**

```
$ curl -s http://localhost:8000/status | python -m json.tool

{
  "status": "ok",
  "documents_indexed": 500
}
```

**Query — hybrid retrieval in action:**

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does ANF handle NFS failover during node outages?", "k": 3}' \
  | python -m json.tool
```

```json
{
  "results": [
    {
      "text": "Azure NetApp Files uses an active-passive controller pair. On node failure, the passive controller takes over within 60 seconds with no client reconfiguration required. NFS sessions reconnect automatically after the failover window.",
      "metadata": { "source": "docs/netapp-ha-guide.md", "section": "Failover Architecture" },
      "score": 0.031746,
      "retrieval": "hybrid"
    },
    {
      "text": "ANF volume failover is transparent to NFS clients using NFSv4.1 session trunking. Ensure `nconnect=8` is set in mount options for optimal throughput during controller transitions.",
      "metadata": { "source": "docs/netapp-mount-options.md", "section": "NFSv4.1 Best Practices" },
      "score": 0.022831,
      "retrieval": "hybrid"
    },
    {
      "text": "Monitor failover events via Azure Monitor metric `VolumeConsumedSizePercentage`. Set an alert threshold at 80% to catch capacity issues before they compound during a controller transition.",
      "metadata": { "source": "docs/monitoring-runbook.md", "section": "ANF Alerts" },
      "score": 0.014285,
      "retrieval": "dense"
    }
  ],
  "total": 3,
  "query": "how does ANF handle NFS failover during node outages?"
}
```

The `retrieval` field shows which method found each result. Results marked `hybrid` were found by **both** dense and BM25 — those are your highest-confidence hits.

---

## Quick Start

```bash
git clone https://github.com/mmillertechnologies-oss/ai-document-search.git
cd ai-document-search

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add OPENAI_API_KEY to .env

uvicorn src.app:app --reload
```

## Usage

**Index documents:**
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc1", "text": "Azure NetApp Files provides enterprise NFS storage.", "metadata": {"source": "docs/storage.md"}},
      {"id": "doc2", "text": "Kubernetes pods can mount persistent volumes via CSI drivers.", "metadata": {"source": "docs/k8s.md"}}
    ]
  }'
```

**Search:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how do containers access persistent storage?", "k": 3}'
```

**Response:**
```json
{
  "results": [
    {
      "text": "Kubernetes pods can mount persistent volumes via CSI drivers.",
      "metadata": {"source": "docs/k8s.md"},
      "score": 0.015873,
      "retrieval": "hybrid"
    },
    {
      "text": "Azure NetApp Files provides enterprise NFS storage.",
      "metadata": {"source": "docs/storage.md"},
      "score": 0.007752,
      "retrieval": "dense"
    }
  ],
  "total": 2,
  "query": "how do containers access persistent storage?"
}
```

The `retrieval` field shows which method found each result: `dense`, `sparse`, or `hybrid` (found by both).

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/index` | POST | Index a batch of documents |
| `/search` | POST | Search with natural language query |
| `/status` | GET | Index size and health |
| `/health` | GET | Liveness check |

## Architecture

```
Query
  │
  ├──► Dense retrieval (OpenAI text-embedding-3-small → ChromaDB)
  │                                                          │
  └──► BM25 retrieval (in-memory TF-IDF index)              │
                          │                                  │
                          └──────── RRF Fusion ─────────────┘
                                        │
                                  Ranked results
                              (score, text, source, method)
```

## Requirements

- Python 3.11+
- OpenAI API key (for embeddings only — no generation cost)
