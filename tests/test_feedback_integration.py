"""Integration tests -- require ANTHROPIC_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()
from app.feedback import get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


@pytest.mark.asyncio
async def test_spanish_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_correct_german():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_french_gender_errors():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_japanese_particle():
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


# ── Candidate tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chinese_word_order():
    result = await get_feedback(
        FeedbackRequest(
            sentence="生 人人 而 自由 平等, 在 尊嚴 和 權利 上 一律。",
            target_language="Chinese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 2
    assert any(e.error_type == "word_order" for e in result.errors)
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.explanation) > 0


@pytest.mark.asyncio
async def test_portuguese_mesoclisis():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ei-te-dar o endereço novo em breve.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert any(e.error_type == "word_order" for e in result.errors)
    assert result.difficulty in {"C1", "C2"}
