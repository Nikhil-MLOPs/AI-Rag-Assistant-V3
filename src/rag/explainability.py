def build_explainability(docs):
    explanation = []
    for idx, d in enumerate(docs):
        explanation.append(
            {
                "chunk_id": idx + 1,
                "score": d.get("score"),
                "rerank_score": d.get("rerank_score"),
                "preview": d["text"][:300],
            }
        )
    return explanation
