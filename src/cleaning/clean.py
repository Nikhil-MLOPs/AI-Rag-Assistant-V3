import json
import yaml
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logging import setup_logging

logger = setup_logging("Cleaning")

# Constants for section header detection and line cleaning

SECTION_HEADERS = {
    "definition",
    "description",
    "purpose",
    "preparation",
    "causes and symptoms",
    "causes",
    "symptoms",
    "diagnosis",
    "treatment",
    "alternative treatment",
    "alternative treatments",
    "prevention",
    "prognosis",
    "risks",
    "aftercare",
    "normal results",
    "abnormal results",
    "precautions",
    "cost",
    "results",
    "key terms",
}

CONTROL_CHARS = ["\u0002"]

# Configuration loading

def load_cleaning_config() -> dict:
    logger.info("Loading cleaning configuration from configs/cleaning.yaml")
    with open("configs/cleaning.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Line cleaning and filtering functions

def clean_line(line: str) -> str:
    for ch in CONTROL_CHARS:
        line = line.replace(ch, "")
    return line.strip()


def is_noise_line(line: str) -> bool:
    return not line or line.isdigit() or len(line) <= 2


def is_cross_reference(line: str) -> bool:
    return " see " in line.lower()


def is_section_header(line: str) -> bool:
    
    # Normalize the line to handle "Header:" or " Header  "
    normalized = line.lower().strip().rstrip(':').strip()
    return normalized in SECTION_HEADERS


def is_author_line(line: str) -> bool:
    words = line.split()
    if 2 <= len(words) <= 4:
        return all(w[0].isupper() for w in words)
    return False


def is_alphabet_header(line: str) -> bool:
    return len(line) == 1 and line.isalpha() and line.isupper()


def is_cross_reference_block(lines: List[str], idx: int) -> bool:
    if idx + 1 < len(lines) and lines[idx + 1].lower() == "definition":
        return False
    if is_cross_reference(lines[idx]):
        return True
    if idx > 0 and is_cross_reference(lines[idx - 1]):
        return True
    return False


# Handling hyphenated lines that are split across two lines in the PDF text extraction

def merge_hyphenated_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.endswith("-")
            and i + 1 < len(lines)
            and lines[i + 1]
            and lines[i + 1][0].islower()
        ):
            merged.append(line[:-1] + lines[i + 1])
            i += 2
        else:
            merged.append(line)
            i += 1
    return merged

# Detecting topics based on the presence of "Definition" in the next line

def detect_topic(lines: List[str], idx: int) -> tuple[str | None, int]:
    def is_valid_topic_line(line: str) -> bool:
        return not (
            is_cross_reference(line)
            or is_author_line(line)
            or is_section_header(line)
            or ";" in line
        )

    if is_cross_reference_block(lines, idx):
        return None, 0

    line = lines[idx]

    if (
        idx + 1 < len(lines)
        and lines[idx + 1].lower() == "definition"
        and is_valid_topic_line(line)
    ):
        return line.strip(), 1

    if (
        idx + 2 < len(lines)
        and lines[idx + 2].lower() == "definition"
        and is_valid_topic_line(line)
        and is_valid_topic_line(lines[idx + 1])
    ):
        return f"{line} {lines[idx + 1]}".strip(), 2

    return None, 0


# Cleaning and chunking

def _emit_chunks(chunks: List[Document], buffers: Dict[str, List[str]], topic: str, page_meta: dict, splitter: RecursiveCharacterTextSplitter):
    for section, lines in buffers.items():
        if not lines:
            continue
        
        text = " ".join(lines).replace("  ", " ").strip()
        
        split_texts = splitter.split_text(text)
        
        for chunk in split_texts:
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "topic": topic,
                        "section": section,
                        "pdf": page_meta["pdf"],
                        "page": page_meta["page"],
                    },
                )
            )

def clean_and_chunk(pages: List[Document]) -> List[Document]:
    cfg = load_cleaning_config()
    logger.info(f"Initialized splitter with chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )

    chunks: List[Document] = []
    current_topic = None
    current_section = None
    section_buffers: Dict[str, List[str]] = {}

    for page_idx, page in enumerate(pages):
        pdf_name = page.metadata.get('pdf', 'unknown')
        page_num = page.metadata.get('page', 'unknown')
        
        raw_lines = [
            clean_line(l)
            for l in page.page_content.splitlines()
            if clean_line(l)
        ]
        raw_lines = merge_hyphenated_lines(raw_lines)

        i = 0
        inside_resources = False

        while i < len(raw_lines):
            line = raw_lines[i]

            if is_noise_line(line):
                i += 1
                continue

            if is_alphabet_header(line):
                i += 1
                continue

            # Resources section handling
            if line.lower() == "resources":
                logger.debug(f"[{pdf_name} p.{page_num}] Found 'Resources' section. Flushing buffers.")
                _emit_chunks(chunks, section_buffers, current_topic, page.metadata, splitter)
                section_buffers = {}
                inside_resources = True
                i += 1
                continue

            if inside_resources:
                topic, consumed = detect_topic(raw_lines, i)
                if topic:
                    logger.info(f"[{pdf_name} p.{page_num}] Detected new topic after Resources: {topic}")
                    inside_resources = False
                    current_topic = topic
                    current_section = "definition"
                    section_buffers = {"definition": []}
                    i += consumed + 1
                else:
                    i += 1
                continue

            # Topic detection
            topic, consumed = detect_topic(raw_lines, i)
            if topic:
                if current_topic:
                    _emit_chunks(chunks, section_buffers, current_topic, page.metadata, splitter)
                
                logger.info(f"[{pdf_name} p.{page_num}] New Topic: {topic}")
                current_topic = topic
                current_section = "definition"
                section_buffers = {"definition": []}
                i += consumed + 1
                continue

            # Section header detection
            if is_section_header(line):
                # If we hit a new header, flush the current content immediately
                if current_section and section_buffers.get(current_section):
                    _emit_chunks(chunks, section_buffers, current_topic, page.metadata, splitter)
                    section_buffers[current_section] = []

                # Clean the section name for consistent metadata
                current_section = line.lower().strip().rstrip(':').strip()
                logger.debug(f"[{pdf_name} p.{page_num}] Entering section: {current_section}")
                section_buffers.setdefault(current_section, [])
                i += 1
                continue

            # Buffering logic
            if current_section and current_topic:
                if current_section not in section_buffers:
                    section_buffers[current_section] = []
                section_buffers[current_section].append(line)
            
            i += 1

        # Emit page content
        _emit_chunks(chunks, section_buffers, current_topic, page.metadata, splitter)
        
        # Reset buffers but keep current topic
        section_buffers = {k: [] for k in section_buffers}

    logger.info(f"Cleaning complete. Generated {len(chunks)} hierarchical chunks total.")
    return chunks


# Main execution

if __name__ == "__main__":
    pages_file = Path("data/processed/pages/pages.jsonl")
    out_dir = Path("data/processed/chunks")
    
    logger.info(f"Starting cleaning process. Output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Document] = []
    if pages_file.exists():
        logger.info(f"Reading ingested pages from {pages_file}")
        with open(pages_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                record = json.loads(line)
                pages.append(
                    Document(
                        page_content=record["text"],
                        metadata=record["metadata"],
                    )
                )
    else:
        logger.error(f"Pages file not found at {pages_file}!")

    if pages:
        logger.info(f"Loaded {len(pages)} pages for processing.")
        chunks = clean_and_chunk(pages)

        out_file = out_dir / "chunks.jsonl"
        logger.info(f"Writing chunks to {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        {
                            "text": chunk.page_content,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info(f"Successfully wrote {len(chunks)} chunks to {out_file}")
    else:
        logger.warning("No pages loaded. Cleaning process aborted.")