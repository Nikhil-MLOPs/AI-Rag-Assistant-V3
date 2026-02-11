class HybridRetriever:
    def __init__(self, dense, sparse, alpha=0.6):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha

    def retrieve(self, query, query_embedding, dense_k, sparse_k):
        dense_docs = self.dense.retrieve(query_embedding, dense_k)
        sparse_docs = self.sparse.retrieve(query, sparse_k)

        return self.fuse(dense_docs, sparse_docs)

    def fuse(self, dense_docs, sparse_docs):
        scores = {}

        for d in dense_docs:
            scores[d["text"]] = self.alpha * (1 - d["score"])

        for s in sparse_docs:
            if s["text"] in scores:
                scores[s["text"]] += (1 - self.alpha) * s["score"]
            else:
                scores[s["text"]] = (1 - self.alpha) * s["score"]

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [{"text": t, "score": sc} for t, sc in fused]
