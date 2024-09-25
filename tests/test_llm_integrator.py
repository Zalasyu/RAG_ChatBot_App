import pytest

from ..llm_model import LLMIntegrator


@pytest.fixture
def llm():
    return LLMIntegrator()


def test_answer_question(llm):
    question = "What is Python?"
    context = "Python is a high-level programming language."
    result = llm.answer_question(question, context)
    assert "answer" in result
    assert "confidence_score" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["confidence_score"], float)
