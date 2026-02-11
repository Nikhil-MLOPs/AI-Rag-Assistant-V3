import re


class Guardrails:

    def __init__(self, cfg):
        self.cfg = cfg["medical_guardrails"]

    def check_emergency(self, query):
        for keyword in self.cfg["emergency_keywords"]:
            if keyword.lower() in query.lower():
                return True
        return False

    def check_no_context(self, retrieved_docs):
        return len(retrieved_docs) == 0

    def compute_confidence(self, docs):
        if not docs:
            return 0.0

        if "rerank_score" in docs[0]:
            scores = [d["rerank_score"] for d in docs]
        else:
            scores = [1 - d["score"] for d in docs]

        return float(sum(scores) / len(scores))

    def check_low_confidence(self, confidence):
        return confidence < self.cfg["confidence_threshold"]

    def validate_citations(self, answer):
        citations = re.findall(r"\[(\d+)\]", answer)
        return list(set(int(c) for c in citations))
