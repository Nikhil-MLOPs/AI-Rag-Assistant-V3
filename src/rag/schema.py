from pydantic import BaseModel
from typing import List, Optional


class RetrievedChunk(BaseModel):
    text: str
    score: float
    rerank_score: Optional[float] = None


class RAGResponse(BaseModel):
    answer: str
    citations: List[int]
    confidence: float
    refusal: bool
    explanation: Optional[str]
    retrieved_chunks: Optional[List[RetrievedChunk]]
