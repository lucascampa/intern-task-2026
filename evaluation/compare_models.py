"""
Compare Haiku, Sonnet, and Opus on the test sentences in answer_key.json.
Records output, latency, and cost per request.

Run with:
    python evaluation/compare_models.py

Test cases are loaded from examples/candidate_inputs.json.
Results are saved to evaluation/comparison_results.json
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
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},
}

# ── Prompt (must match app/feedback.py) ──────────────────────────────────────

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
    user_message = (
        f"Target language: {test['target_language']}\n"
        f"Native language: {test['native_language']}\n"
        f"Sentence: {test['sentence']}"
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
    expected = test["expected"]

    # Derive expected error type from first error in answer key (if any)
    expected_error_types = {e["error_type"] for e in expected.get("errors", [])}

    return {
        "model": model,
        "latency_seconds": round(latency, 2),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": round(cost, 6),
        "output": output,
        "parse_error": parse_error,
        # Did is_correct match?
        "is_correct_match": (
            output is not None
            and output.get("is_correct") == expected["is_correct"]
        ),
        # Does corrected_sentence match expected?
        "corrected_sentence_match": (
            output is not None
            and output.get("corrected_sentence") == expected["corrected_sentence"]
        ),
        # Did at least one error type match the answer key?
        "error_type_match": (
            None if not expected_error_types
            else (
                output is not None
                and bool(
                    {e.get("error_type") for e in output.get("errors", [])}
                    & expected_error_types
                )
            )
        ),
    }


async def main():
    answer_key_path = Path(__file__).parent.parent / "examples" / "candidate_inputs.json"
    test_cases = json.loads(answer_key_path.read_text(encoding="utf-8"))

    client = anthropic.AsyncAnthropic()
    results = []

    for test in test_cases:
        print(f"\n── {test['target_language']}: {test['sentence'][:60]}")
        test_results = []

        for model in MODELS:
            print(f"   {model}...", end=" ", flush=True)
            result = await run_one(client, model, test)
            test_results.append(result)
            print(
                f"{result['latency_seconds']}s  "
                f"${result['cost_usd']}  "
                f"is_correct={result['is_correct_match']}  "
                f"sentence={result['corrected_sentence_match']}  "
                f"error_type={result['error_type_match']}"
            )

        results.append({"test": test, "results": test_results})

    # Save full results
    output_path = Path(__file__).parent / "comparison_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nFull results saved to {output_path}")

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────────────────────")
    print(f"{'Model':<22} {'Avg Latency':>12} {'Total Cost':>12} {'is_correct':>11} {'Sentence':>9} {'Error type':>11}")
    print("-" * 82)

    for model in MODELS:
        model_results = [r for entry in results for r in entry["results"] if r["model"] == model]
        avg_latency = sum(r["latency_seconds"] for r in model_results) / len(model_results)
        total_cost = sum(r["cost_usd"] for r in model_results)
        n = len(model_results)
        is_correct_matches = sum(1 for r in model_results if r["is_correct_match"])
        sentence_matches = sum(1 for r in model_results if r["corrected_sentence_match"])
        type_checks = [r for r in model_results if r["error_type_match"] is not None]
        type_matches = sum(1 for r in type_checks if r["error_type_match"])
        print(
            f"{model:<22} {avg_latency:>11.2f}s "
            f"${total_cost:>10.6f} "
            f"{is_correct_matches}/{n:>9} "
            f"{sentence_matches}/{n:>7} "
            f"{type_matches}/{len(type_checks):>9}"
        )


if __name__ == "__main__":
    asyncio.run(main())
