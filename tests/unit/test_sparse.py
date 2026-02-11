from src.retrieval.sparse import SparseRetriever


def test_sparse_returns_docs():
    dummy_texts = [
        "Diabetes is a chronic disease.",
        "Hypertension is high blood pressure.",
        "Insulin regulates blood sugar.",
    ]

    retriever = SparseRetriever(texts=dummy_texts)

    docs = retriever.retrieve("diabetes", top_k=2)

    assert len(docs) == 2
    assert isinstance(docs[0]["text"], str)
