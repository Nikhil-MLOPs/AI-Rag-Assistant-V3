from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k):
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            docs[i]["rerank_score"] = float(score)

        docs = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return docs[:top_k]
