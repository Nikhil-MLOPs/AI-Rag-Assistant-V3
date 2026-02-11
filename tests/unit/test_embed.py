import json
import numpy as np
import pytest
from pathlib import Path
from src.embeddings.embed import embed_chunks

def test_embed_chunks_logic(mocker, tmp_path):
    # 1. Setup Mock Filesystem
    base_dir = tmp_path / "data"
    chunks_file = base_dir / "processed" / "chunks" / "chunks.jsonl"
    out_dir = base_dir / "embeddings"
    chunks_file.parent.mkdir(parents=True)
    
    # Create fake chunk data
    fake_chunks = [
        {"text": "First medical chunk", "metadata": {"topic": "A"}},
        {"text": "Second medical chunk", "metadata": {"topic": "B"}}
    ]
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in fake_chunks:
            f.write(json.dumps(c) + "\n")

    # 2. Mock Config & Paths
    mocker.patch("src.embeddings.embed.load_embedding_config", 
                 return_value={"model_name": "mock-model", "batch_size": 2})
    
    # Redirect Paths in the script to our tmp_path
    mocker.patch("src.embeddings.embed.Path", side_effect=lambda p: {
        "data/processed/chunks/chunks.jsonl": chunks_file,
        "data/embeddings": out_dir
    }.get(str(p).replace("\\", "/"), Path(p)))

    # 3. Mock SentenceTransformer
    # We don't want to load a real model!
    mock_model_instance = mocker.MagicMock()
    # Mock encode to return a numpy array of shape (num_chunks, vector_dim)
    mock_model_instance.encode.return_value = np.random.rand(2, 384)
    mocker.patch("src.embeddings.embed.SentenceTransformer", return_value=mock_model_instance)

    # 4. RUN
    embed_chunks()

    # 5. ASSERTIONS
    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.json"

    assert emb_path.exists()
    assert meta_path.exists()

    # Check if dimensions are correct
    loaded_embs = np.load(emb_path)
    assert loaded_embs.shape == (2, 384)

    # Check if metadata count matches
    with open(meta_path, "r", encoding="utf-8") as f:
        loaded_meta = json.load(f)
    assert len(loaded_meta) == 2
    assert loaded_meta[0]["topic"] == "A"