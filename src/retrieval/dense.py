import chromadb


class DenseRetriever:
    def __init__(self, collection):
        self.collection = collection

    def retrieve(self, query_embedding, top_k: int):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i],
                }
            )
        return docs
