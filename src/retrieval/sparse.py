import json
from pathlib import Path
from rank_bm25 import BM25Okapi


class SparseRetriever:
    def __init__(self, chunks_path="data/processed/chunks/chunks.jsonl"):
        self.texts = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.texts.append(json.loads(line)["text"])

        tokenized = [doc.split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int):
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(
            zip(self.texts, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        docs = []
        for text, score in ranked:
            docs.append(
                {
                    "text": text,
                    "metadata": {},
                    "score": float(score),
                }
            )
        return docs
