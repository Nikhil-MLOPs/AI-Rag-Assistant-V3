import json
import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.cleaning.clean import (
    clean_line, is_noise_line, is_cross_reference, is_section_header,
    is_author_line, is_alphabet_header, is_cross_reference_block,
    merge_hyphenated_lines, detect_topic, clean_and_chunk
)

def test_clean_line():
    assert clean_line("\u0002 hello world \u0002") == "hello world"
    assert clean_line("  test  ") == "test"

def test_is_noise_line():
    assert is_noise_line("123") is True  # digit
    assert is_noise_line("ab") is True   # len <= 2
    assert is_noise_line("abc") is False

def test_is_cross_reference():
    assert is_cross_reference("Apple see Banana") is True
    assert is_cross_reference("Normal Text") is False

def test_is_section_header():
    assert is_section_header("Definition") is True
    assert is_section_header("key terms:") is True
    assert is_section_header("Unknown") is False

def test_is_author_line():
    assert is_author_line("John Doe") is True
    assert is_author_line("J. K. Rowling") is True 
    assert is_author_line("A B C D E") is False

def test_is_alphabet_header():
    assert is_alphabet_header("A") is True
    assert is_alphabet_header("ab") is False

def test_is_cross_reference_block():
    lines = ["Something see else", "definition", "Content"]
    assert is_cross_reference_block(lines, 0) is False 
    
    lines_ref = ["Something see else", "Not a definition"]
    assert is_cross_reference_block(lines_ref, 0) is True

def test_merge_hyphenated_lines():
    lines = ["Hello world-", "test", "Normal line"]
    assert merge_hyphenated_lines(lines) == ["Hello worldtest", "Normal line"]

def test_detect_topic():
    lines = ["Asthma and allergy", "definition", "Content"]
    topic, consumed = detect_topic(lines, 0)
    
    assert topic == "Asthma and allergy"
    assert consumed == 1

    lines_2 = ["Type 2", "Diabetes", "definition", "Content"]
    topic_2, consumed_2 = detect_topic(lines_2, 0)
    assert topic_2 == "Type 2 Diabetes"
    assert consumed_2 == 2

def test_clean_and_chunk(mocker):
    mocker.patch("src.cleaning.clean.load_cleaning_config", 
                 return_value={"chunk_size": 100, "chunk_overlap": 0})

    pages = [
        Document(
            page_content="Asthma\ndefinition\nDifficulty breathing.\ncauses\nDust.",
            metadata={"pdf": "test.pdf", "page": 1},
        )
    ]

    chunks = clean_and_chunk(pages)

    assert len(chunks) >= 2
    assert any(c.metadata["topic"] == "Asthma" for c in chunks)
    assert any(c.metadata["section"] == "definition" for c in chunks)
    assert any(c.metadata["section"] == "causes" for c in chunks)

def test_clean_without_physical_files(mocker, tmp_path):
    data_dir = tmp_path / "data" / "processed"
    pages_file = data_dir / "pages" / "pages.jsonl"
    out_dir = data_dir / "chunks"
    pages_file.parent.mkdir(parents=True)
    
    fake_record = {
        "text": "Flu\ndefinition\nChills.", 
        "metadata": {"pdf": "virtual.pdf", "page": 1}
    }
    pages_file.write_text(json.dumps(fake_record) + "\n")

    mocker.patch("src.cleaning.clean.load_cleaning_config", 
                 return_value={"chunk_size": 100, "chunk_overlap": 0})

    # Simulate Main Logic
    pages = []
    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages.append(Document(page_content=record["text"], metadata=record["metadata"]))

    chunks = clean_and_chunk(pages)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk.page_content, "metadata": chunk.metadata}) + "\n")

    assert out_file.exists()
    final_output = json.loads(out_file.read_text().splitlines()[0])
    assert final_output["metadata"]["topic"] == "Flu"