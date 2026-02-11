from src.retrieval.sparse import SparseRetriever


def test_sparse_returns_docs():
    retriever = SparseRetriever()
    docs = retriever.retrieve("diabetes", top_k=3)
    assert len(docs) == 3
    assert "text" in docs[0]
