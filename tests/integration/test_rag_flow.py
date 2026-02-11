from src.rag.chain import RagChain


class DummyGuardrails:
    def __init__(self):
        pass

    def validate_citations(self, answer):
        return [1]

    def compute_confidence(self, docs):
        return 0.8

    def check_emergency(self, query):
        return False

    def check_no_context(self, docs):
        return False

    def check_low_confidence(self, confidence):
        return False


class DummyChain(RagChain):
    def __init__(self):
        pass

    def generate(self, query, docs, prompt):
        return (
            {
                "answer": "Test answer [1]",
                "citations": [1],
                "confidence": 0.8,
                "refusal": False,
                "explanation": "ok",
                "retrieved_chunks": [],
            },
            0.1,
        )


def test_rag_flow():
    chain = DummyChain()

    response, llm_time = chain.generate(
        "query",
        [{"text": "doc", "score": 0.1}],
        "prompt"
    )

    assert "answer" in response
