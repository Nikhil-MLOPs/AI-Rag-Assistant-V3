from src.rag.guardrails import Guardrails


def test_confidence_computation():
    cfg = {
        "medical_guardrails": {
            "confidence_threshold": 0.3,
            "emergency_keywords": [],
        }
    }

    guard = Guardrails(cfg)

    docs = [
        {"rerank_score": 0.8},
        {"rerank_score": 0.6},
    ]

    confidence = guard.compute_confidence(docs)

    assert confidence > 0.5


def test_emergency_detection():
    cfg = {
        "medical_guardrails": {
            "confidence_threshold": 0.3,
            "emergency_keywords": ["heart attack"],
        }
    }

    guard = Guardrails(cfg)

    assert guard.check_emergency("I think I have a heart attack")
