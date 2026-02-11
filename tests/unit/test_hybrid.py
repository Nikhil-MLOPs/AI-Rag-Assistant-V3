from src.retrieval.hybrid import HybridRetriever


class DummyDense:
    def retrieve(self, query_embedding, top_k):
        return [
            {"text": "doc1", "score": 0.2},
            {"text": "doc2", "score": 0.3},
        ]


class DummySparse:
    def retrieve(self, query, top_k):
        return [
            {"text": "doc1", "score": 0.8},
            {"text": "doc3", "score": 0.7},
        ]


def test_hybrid_fusion():
    hybrid = HybridRetriever(DummyDense(), DummySparse(), alpha=0.5)

    docs = hybrid.retrieve("query", [0.1], 2, 2)

    assert len(docs) >= 1
    assert isinstance(docs, list)
