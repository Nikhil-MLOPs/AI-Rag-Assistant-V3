import json
from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings
from src.utils.logging import setup_logging

logger = setup_logging("vector_store")

def get_vector_store(persist_directory: str = "data/chroma_db"):
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )

def store_embeddings():
    # Setup paths
    emb_path = Path("data/embeddings/embeddings.npy")
    meta_path = Path("data/embeddings/metadata.json")
    chunks_file = Path("data/processed/chunks/chunks.jsonl")
    
    if not (emb_path.exists() and meta_path.exists()):
        logger.error("Files not found. Run embed.py first.")
        return

    # Load data
    logger.info("Loading embeddings and metadata...")
    embeddings = np.load(emb_path).tolist()
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    
    texts = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    # Initialize Chroma
    client = get_vector_store()
    collection = client.get_or_create_collection(name="document_embeddings")
    ids = [f"id_{i}" for i in range(len(texts))]

    # Batch processing
    total_records = len(ids)
    batch_size = 5000  # Safely under the 5461 limit
    
    logger.info(f"Storing {total_records} vectors in batches of {batch_size}...")
    
    try:
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            
            collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=texts[i:end_idx]
            )
            logger.info(f"Stored batch {i} to {end_idx}...")
            
        logger.info("Successfully completed vector storage.")
    except Exception as e:
        logger.error(f"Error during storage: {e}")
        raise

if __name__ == "__main__":
    store_embeddings()