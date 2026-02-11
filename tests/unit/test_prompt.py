from src.rag.prompt import build_medical_prompt


def test_prompt_contains_context():
    prompt = build_medical_prompt(
        "What is diabetes?",
        "Context text",
        "History"
    )

    assert "Context text" in prompt
    assert "What is diabetes?" in prompt
