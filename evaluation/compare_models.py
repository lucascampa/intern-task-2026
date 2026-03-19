"""
Compare Haiku, Sonnet, and Opus on a set of test sentences.
Records output, latency, and cost per request.

Run with:
    python scripts/compare_models.py

Results are saved to scripts/comparison_results.json
"""

import asyncio
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ── Model config ──────────────────────────────────────────────────────────────

MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
]

# Prices in USD per million tokens
PRICING = {
    "claude-haiku-4-5":   {"input": 1.00,  "output": 5.00},
    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00},
}

# ── Test sentences ────────────────────────────────────────────────────────────
# Each has a known error and an expected correction you can verify.

TEST_CASES = [
    {
        "description": "Portuguese – verb tense error (verifiable by user)",
        "request": {
            "sentence": "Eu fico em casa ontem porque estava doente.",
            "target_language": "Portuguese",
            "native_language": "English",
        },
        "known_error_type": "conjugation",
        "expected_correction": "Eu fiquei em casa ontem porque estava doente.",
        "notes": (
            "'Fico' is present tense (I stay). "
            "With 'ontem' (yesterday) it must be 'fiquei' (I stayed)."
        ),
    },
    {
        "description": "Korean – particle error (non-Latin script)",
        "request": {
            "sentence": "나는 학교을 갔어요.",
            "target_language": "Korean",
            "native_language": "English",
        },
        "known_error_type": "grammar",
        "expected_correction": "나는 학교에 갔어요.",
        "notes": (
            "'을' is the object particle. "
            "With 갔어요 (went), the location particle '에' is required."
        ),
    },
    {
        "description": "Russian – conjugation error (non-Latin script)",
        "request": {
            "sentence": "Я идёт в магазин.",
            "target_language": "Russian",
            "native_language": "English",
        },
        "known_error_type": "conjugation",
        "expected_correction": "Я иду в магазин.",
        "notes": (
            "'Идёт' is 3rd person singular (he/she goes). "
            "With 'Я' (I) it must be 'иду'."
        ),
    },
    {
        "description": "Chinese – extra word error (non-Latin script)",
        "request": {
            "sentence": "我是很高兴认识你。",
            "target_language": "Chinese",
            "native_language": "English",
        },
        "known_error_type": "extra_word",
        "expected_correction": "我很高兴认识你。",
        "notes": (
            "'是' (to be) is not used before adjectives in Chinese. "
            "The correct form omits 是."
        ),
    },
    {
        "description": "Spanish – correct sentence, no errors",
        "request": {
            "sentence": "Ayer fui al supermercado y compré leche y pan.",
            "target_language": "Spanish",
            "native_language": "English",
        },
        "known_error_type": None,
        "expected_correction": "Ayer fui al supermercado y compré leche y pan.",
        "notes": "Grammatically correct. Model should return is_correct=true.",
    },
]

# ── Prompt (same as app/feedback.py) ─────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a language-learning assistant. A student is practicing writing in their \
target language. Your job is to analyze their sentence, find errors, and provide \
helpful feedback.

RULES:
1. If the sentence is already correct, return is_correct=true, an empty errors \
array, and set corrected_sentence to the original sentence exactly.
2. For each error, identify the original text, provide the correction, classify \
the error type, and explain the error in the learner's NATIVE language so they \
can understand.
3. Error types must be one of: grammar, spelling, word_choice, punctuation, \
word_order, missing_word, extra_word, conjugation, gender_agreement, \
number_agreement, tone_register, other.
4. Assign a CEFR difficulty level (A1–C2) based on the complexity of the \
sentence (vocabulary, grammar structures used), NOT based on whether it has errors.
5. The corrected_sentence should be the minimal correction -- preserve the \
learner's original meaning and style as much as possible.
6. Explanations should be concise (1–2 sentences), friendly, and educational.

Respond with ONLY valid JSON — no explanation, no markdown, no code fences. \
Match this exact schema:
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "string",
      "explanation": "string (in native language)"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}
"""

# ── Core logic ────────────────────────────────────────────────────────────────

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


async def run_one(client: anthropic.AsyncAnthropic, model: str, test: dict) -> dict:
    req = test["request"]
    user_message = (
        f"Target language: {req['target_language']}\n"
        f"Native language: {req['native_language']}\n"
        f"Sentence: {req['sentence']}"
    )

    start = time.perf_counter()
    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    latency = time.perf_counter() - start

    content = response.content[0].text.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        output = json.loads(content)
        parse_error = None
    except json.JSONDecodeError as e:
        output = None
        parse_error = str(e)

    cost = calc_cost(model, response.usage.input_tokens, response.usage.output_tokens)

    return {
        "model": model,
        "latency_seconds": round(latency, 2),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": round(cost, 6),
        "output": output,
        "parse_error": parse_error,
        "caught_error": (
            None if test["known_error_type"] is None
            else (output is not None and not output.get("is_correct", True))
        ),
        "correct_sentence_match": (
            output is not None
            and output.get("corrected_sentence") == test["expected_correction"]
        ),
        "error_type_correct": (
            None if test["known_error_type"] is None or output is None
            else any(
                e.get("error_type") == test["known_error_type"]
                for e in output.get("errors", [])
            )
        ),
    }


async def main():
    client = anthropic.AsyncAnthropic()
    results = []

    for test in TEST_CASES:
        print(f"\n── {test['description']}")
        test_results = []

        for model in MODELS:
            print(f"   {model}...", end=" ", flush=True)
            result = await run_one(client, model, test)
            test_results.append(result)
            print(
                f"{result['latency_seconds']}s  "
                f"${result['cost_usd']}  "
                f"caught={result['caught_error']}  "
                f"match={result['correct_sentence_match']}"
            )

        results.append({
            "test": test,
            "results": test_results,
        })

    # Save full results
    output_path = Path(__file__).parent / "comparison_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nFull results saved to {output_path}")

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"{'Model':<22} {'Avg Latency':>12} {'Total Cost':>12} {'Errors Caught':>14} {'Sentence Match':>15}")
    print("-" * 80)

    for model in MODELS:
        model_results = [r for entry in results for r in entry["results"] if r["model"] == model]
        avg_latency = sum(r["latency_seconds"] for r in model_results) / len(model_results)
        total_cost = sum(r["cost_usd"] for r in model_results)
        caught = [r for r in model_results if r["caught_error"] is True]
        expected_catches = [r for r in model_results if r["caught_error"] is not None]
        matches = sum(1 for r in model_results if r["correct_sentence_match"])
        print(
            f"{model:<22} {avg_latency:>11.2f}s "
            f"${total_cost:>10.6f} "
            f"{len(caught)}/{len(expected_catches):>12} "
            f"{matches}/{len(model_results):>14}"
        )


if __name__ == "__main__":
    asyncio.run(main())
