import json
from src.ingestion.ingest import clean_footer, ingest

def test_clean_footer():

    text = """This is some text.
    g a l e e n c y c l o p e d i a"""

    cleaned = clean_footer(text)

    assert cleaned == "This is some text."
    assert "g a l e e n c y c l o p e d i a" not in cleaned



def test_ingest_without_physical_files(mocker):
    
    mock_cfg = mocker.MagicMock()
    mock_cfg.raw_dir = "fake_raw_path"
    mock_cfg.processed_dir = "fake_out_path"
    mock_cfg.skip_start_pages = 0
    mock_cfg.skip_end_after = 5
    mocker.patch("src.ingestion.ingest.load_ingestion_config", return_value=mock_cfg)

    
    mocker.patch("src.ingestion.ingest.Path.mkdir")
    fake_pdf_path = mocker.MagicMock()
    fake_pdf_path.name = "virtual_document.pdf"
    
    mocker.patch("src.ingestion.ingest.Path.glob", return_value=[fake_pdf_path])

    
    mock_doc = mocker.MagicMock() 
    mock_doc.__len__.return_value = 1
    
    mock_page = mocker.MagicMock()
    mock_page.get_text.return_value = "Content from virtual PDF"
    mock_doc.load_page.return_value = mock_page
    
    mocker.patch("pymupdf.open", return_value=mock_doc)

    
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)

    
    ingest()

    handle = mock_file()
    
    written_call_args = handle.write.call_args_list[0][0][0]
    result = json.loads(written_content := written_call_args.strip())

    assert result["text"] == "Content from virtual PDF"
    assert result["metadata"]["pdf"] == "virtual_document.pdf"
    assert result["metadata"]["page"] == 1