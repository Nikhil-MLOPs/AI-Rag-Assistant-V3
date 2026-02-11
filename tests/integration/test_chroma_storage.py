import pytest
import chromadb
from chromadb.config import Settings

@pytest.fixture
def test_collection():
    """Provides a fresh, in-memory Chroma collection for each test."""
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    return client.create_collection(name="test_collection")

def test_upsert_and_retrieval(test_collection):
    """Tests if we can actually retrieve what we store."""
    # Prepare small test data
    ids = ["doc1", "doc2"]
    embeddings = [[0.1, 0.2], [0.9, 0.8]]
    documents = ["The cat sat on the mat.", "The rocket launched to Mars."]
    metadatas = [{"source": "cat_book"}, {"source": "space_news"}]

    # Upsert
    test_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    # Query
    results = test_collection.query(
        query_embeddings=[[0.1, 0.2]],
        n_results=1
    )

    # Assert
    assert results["ids"][0][0] == "doc1"
    assert "cat" in results["documents"][0][0]
    assert results["metadatas"][0][0]["source"] == "cat_book"