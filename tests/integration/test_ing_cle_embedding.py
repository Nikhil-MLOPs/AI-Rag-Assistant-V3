import json
import numpy as np
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.ingest import ingest
from src.cleaning.clean import clean_and_chunk
from src.embeddings.embed import embed_chunks

def test_complete_etl_pipeline(mocker, tmp_path):
    """
    Tests the full flow: 
    1. Raw PDF Ingestion -> 2. Text Cleaning/Chunking -> 3. Vector Embedding
    """

    base_dir = tmp_path / "data"
    raw_dir = base_dir / "raw"
    proc_dir = base_dir / "processed"
    emb_dir = base_dir / "embeddings"

    for d in [raw_dir, proc_dir, emb_dir]: 
        d.mkdir(parents=True, exist_ok=True)

    # Mock Ingestion Config
    mock_ingest_cfg = mocker.Mock()
    mock_ingest_cfg.raw_dir = str(raw_dir)
    mock_ingest_cfg.processed_dir = str(proc_dir)
    mock_ingest_cfg.skip_start_pages = 0
    mock_ingest_cfg.skip_end_after = 100
    mocker.patch("src.ingestion.ingest.load_ingestion_config", return_value=mock_ingest_cfg)

    # Mock Cleaning Config
    mocker.patch("src.cleaning.clean.load_cleaning_config", 
                 return_value={"chunk_size": 500, "chunk_overlap": 0})

    # Mock Embedding Config
    mocker.patch("src.embeddings.embed.load_embedding_config", 
                 return_value={"model_name": "mock-model", "batch_size": 2})

    # We force the embed script to use our temp folders instead of real project folders
    mocker.patch("src.embeddings.embed.Path", side_effect=lambda p: {
        "data/processed/chunks/chunks.jsonl": proc_dir / "chunks" / "chunks.jsonl",
        "data/embeddings": emb_dir
    }.get(str(p).replace("\\", "/"), Path(p)))

    # Setup Mock Page
    mock_page = mocker.MagicMock()
    
    mock_page.get_text.return_value = "Pneumonia\ndefinition\nInfection of the lungs."
    
    # Setup Mock Document
    mock_doc = mocker.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_doc.load_page.return_value = mock_page
    
    mock_doc.__iter__.return_value = [mock_page] 

    mocker.patch("src.ingestion.ingest.pymupdf.open", return_value=mock_doc)

    # Mock SentenceTransformer to return a fake vector
    mock_model = mocker.Mock()
    
    mock_model.encode.return_value = np.random.rand(1, 384)
    mocker.patch("src.embeddings.embed.SentenceTransformer", return_value=mock_model)

    (raw_dir / "test_encyclopedia.pdf").write_text("fake binary")
    ingest()
    
    pages_file = proc_dir / "pages.jsonl"
    assert pages_file.exists(), "Step A Failed: Ingestion did not create pages.jsonl"

    pages = []
    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            pages.append(Document(page_content=data["text"], metadata=data["metadata"]))
    
    chunks = clean_and_chunk(pages)
    
    chunks_out_file = proc_dir / "chunks" / "chunks.jsonl"
    chunks_out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk.page_content, "metadata": chunk.metadata}) + "\n")
    
    assert chunks_out_file.exists(), "Step B Failed: Cleaning did not create chunks.jsonl"

    embed_chunks()

    emb_npy = emb_dir / "embeddings.npy"
    meta_json = emb_dir / "metadata.json"

    assert emb_npy.exists(), "Final Step Failed: .npy file not found"
    assert meta_json.exists(), "Final Step Failed: metadata.json not found"

    final_vectors = np.load(emb_npy)
    with open(meta_json, "r", encoding="utf-8") as f:
        final_meta = json.load(f)

    assert final_vectors.shape == (1, 384), f"Expected (1, 384), got {final_vectors.shape}"
    assert final_meta[0]["topic"] == "Pneumonia"
    assert final_meta[0]["section"] == "definition"

    print("\n Full ETL Integration Pipeline Verified Successfully!")