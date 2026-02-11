import json
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.ingest import ingest
from src.cleaning.clean import clean_and_chunk

def test_full_ingestion_to_cleaning_pipeline(mocker, tmp_path):
    
    base_dir = tmp_path / "data"
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    chunks_dir = processed_dir / "chunks"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    mock_ingest_cfg = mocker.MagicMock()
    mock_ingest_cfg.raw_dir = str(raw_dir)
    mock_ingest_cfg.processed_dir = str(processed_dir)
    mock_ingest_cfg.skip_start_pages = 0
    mock_ingest_cfg.skip_end_after = 100
    mocker.patch("src.ingestion.ingest.load_ingestion_config", return_value=mock_ingest_cfg)

    mock_clean_cfg = {"chunk_size": 500, "chunk_overlap": 0}
    mocker.patch("src.cleaning.clean.load_cleaning_config", return_value=mock_clean_cfg)

    fake_pdf = raw_dir / "encyclopedia.pdf"
    fake_pdf.write_text("dummy") 
    
    mock_doc = mocker.MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = mocker.MagicMock()
    
    mock_page.get_text.return_value = "Pneumonia\ndefinition\nInfection of lungs.\ncauses\nBacteria."
    mock_doc.load_page.return_value = mock_page
    mocker.patch("pymupdf.open", return_value=mock_doc)

    
    ingest()
    
    
    pages_file = processed_dir / "pages.jsonl"
    assert pages_file.exists(), f"Expected pages.jsonl at {pages_file}"

    
    pages = []
    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages.append(Document(page_content=record["text"], metadata=record["metadata"]))

    chunks = clean_and_chunk(pages)

    
    assert len(chunks) >= 2
    assert chunks[0].metadata["topic"] == "Pneumonia"
    assert chunks[0].metadata["section"] == "definition"
    assert "Infection" in chunks[0].page_content