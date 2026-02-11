import ollama
import time
from src.rag.guardrails import Guardrails
from src.rag.explainability import build_explainability
from src.rag.schema import RAGResponse


class RagChain:
    def __init__(self, model, temperature, guardrail_cfg):
        self.model = model
        self.temperature = temperature
        self.guardrails = Guardrails(guardrail_cfg)

    def generate(self, query, docs, prompt):
        start_llm = time.time()

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )

        llm_time = time.time() - start_llm

        answer = response["message"]["content"]

        citations = self.guardrails.validate_citations(answer)
        confidence = self.guardrails.compute_confidence(docs)

        refusal = False
        explanation = None

        # Guardrail logic
        if self.guardrails.check_emergency(query):
            refusal = True
            answer = (
                "This appears to be a medical emergency. "
                "Please seek immediate medical attention."
            )

        elif self.guardrails.check_no_context(docs):
            refusal = True
            answer = (
                "I cannot find sufficient medical evidence in the retrieved documents."
            )

        elif self.guardrails.check_low_confidence(confidence):
            refusal = True
            answer = (
                "The available evidence is insufficient to provide a confident answer."
            )

        if not citations:
            refusal = True

        explanation = build_explainability(docs)

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            refusal=refusal,
            explanation="Guardrails applied",
            retrieved_chunks=explanation,
        ), llm_time
