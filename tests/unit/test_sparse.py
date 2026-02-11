import pytest
from pathlib import Path
import json
import tempfile

from src.retrieval.sparse import SparseRetriever


def test_sparse_returns_docs():
    dummy_texts = [
        "Diabetes is a chronic disease.",
        "Hypertension is high blood pressure.",
        "Insulin regulates blood sugar.",
    ]

    try:
        
        retriever = SparseRetriever(texts=dummy_texts)
    except TypeError:
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks_path = Path(tmpdir) / "chunks.jsonl"

            with open(chunks_path, "w", encoding="utf-8") as f:
                for text in dummy_texts:
                    json.dump({"text": text}, f)
                    f.write("\n")

            retriever = SparseRetriever(chunks_path=str(chunks_path))

    docs = retriever.retrieve("diabetes", top_k=2)

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert "text" in docs[0]