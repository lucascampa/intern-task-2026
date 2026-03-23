# Language Feedback API

## Overview

This is a language learning API, more specifically built for text correction. It's built using FastAPI and Anthropic's models.
The user inputs a sentence in the language they're learning, the name of the language, and their native language. The API returns the corrected version of the sentence and a feedback.
This repository contains files pertaining the agent itself, a few test cases, as well as a model comparison.

## Design Decisions

### LLM Provider

Switched from OpenAI to Anthropic.
From my personal use, OAI's models are good enough to consumers, but not necessarily the best for business. Moreover, a popular opinion is that Anthropic's models are better suited for B2B products. So, following that simple heuristic, I decided to switch to Anthropic.

### Model Choice

I ran four iterations using three models:
- Claude Haiku 4.5
- Claude Sonnet 4.6
- Claude Opus 4.6

The choice took into consideration accuracy of the responses, latency, and cost. **Haiku** consistently ranked the highest among the three, so it's my choice.

### Prompt Design

The original prompt gives context about what the LLM should do, as well as six rules to follow:

1. Handle the correct-sentence case (is_correct=true, empty errors, original sentence)
2. For each error: identify original text, provide correction, classify type, explain in the native language
3. Constrain error types to the 12 allowed values
4. Assign CEFR difficulty based on sentence complexity, not error presence
5. Minimal correction — preserve the learner's voice
6. Explanations: concise (1–2 sentences), friendly, educational

Before the comparison runs, one early fix was also needed: the model was wrapping its JSON output in markdown fences (` ```json ... ``` `), which broke parsing. An explicit instruction to return plain JSON only was added to the prompt.

I observed a few issues:

- Verbosity in the answers
- Lack of precision in classifying errors (e.g. often the sentence had an extra word, and instead of labeling the error as `extra_word` it would label it as `grammar`)
- It would correct punctuation at the end of the sentence. Punctuation is a core part of grammar, and it should be flagged in the middle of a sentence. But at the end it's irrelevant for learning
- Inconsistency in presenting the corrected sentence when there was an extra word error. At times it would return "", others the full sentence without the extra word, or just a section. The first two work, but the third can lead to confusion

I added five other rules:

7. Ignore punctuation at the end, with two examples
8. A reminder to keep in mind some sentences might be more or less complex than they might seem at first, with an example
9. Another instruction highlighting the importance of brevity
10. Mentioning that there might be more than one error type, and that they should be labeled as precisely as possible
11. A rule for `extra_word` specifically, with an example

Prompt engineering can only go so far. Most of the limitations are at the model layer. What I have found is that the models optimize for simplicity, not pure accuracy. This means that, for example, if they see an extra word, they tend to try to fit it into the sentence in the correction, rather than just removing it altogether. It is assumed that removing a word would remove meaning.

### Known Limitations

- *Portuguese mesoclisis*: except for Haiku in the last iteration, all the models would consistently fail at it. It's not that they can't recognize it - Sonnet even recognizes it at the last iteration, but its simplicity bias makes it dismiss it: "you may be thinking of the mesoclitic future construction (like 'dar-te-ei'), which is very formal/archaic and structured differently."
- The Chinese sentence contains several mistakes, which made the feedback verbose. I'm not a speaker, so it was a challenge to even verify accuracy and consistency of the explanations
- *German*: inconsistent across runs, and another glaring example of the models being "lazy" about removing words. The wrong sentence has "Deutsche" as an extra word, completely breaking the sentence's meaning. Be it Haiku, Sonnet, or Opus, the models would very often change the corrected sentence to fit the extra information instead of seeing it as noise

In a nutshell: the models optimize for a "most natural native sentence" rather than "closest fix to what the learner wrote."

## Evaluation

- Built a [comparison script](evaluation/compare_models.py), running the three models against the six sentences and outputs latency, token usage, cost ($), and three accuracy metrics (`is_correct_match`, `corrected_sentence_match`, and `error_type_match`)
- Ran four iterations: a baseline, a second after fixing the Russian test sentence, and two post-prompt engineering rounds
- Both `is_correct_match` and `corrected_sentence_match` are strict, while `error_type_match` is quite lenient (every overlap counts), so none tell the full story in isolation. Results had to be reviewed manually

### Methodology

[`compare_models.py`](evaluation/compare_models.py) runs all three models against the same 6 test sentences from `examples/candidate_inputs.json` and records the following per request:

- **Latency** (seconds)
- **Token usage** (input and output)
- **Cost** (USD, calculated from fixed per-model pricing)
- **`is_correct_match`** — whether the model's `is_correct` flag matched the answer key (strict)
- **`corrected_sentence_match`** — whether the model's corrected sentence matched the answer key exactly (strict)
- **`error_type_match`** — whether any of the model's error types overlapped with the answer key's (lenient)

Results are saved to timestamped JSON files under `evaluation/`. Four runs were performed: a baseline, a second after fixing a malformed test sentence, and two post-prompt engineering rounds.

### Results Summary

The table below showcases the best performing model for each language in each iteration.

| Language   | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
|------------|-------------|-------------|-------------|-------------|
| Chinese    | Opus        | Opus        | Opus        | Sonnet      |
| Portuguese | Haiku       | Haiku       | Haiku       | Haiku       |
| Korean     | Haiku       | Haiku       | Haiku       | Sonnet      |
| Russian    | —           | Haiku       | Haiku       | Haiku       |
| German     | Haiku       | Sonnet      | Haiku       | Sonnet      |
| Japanese   | Haiku       | Haiku       | Haiku       | Haiku       |

Performance here = the best tradeoff between cost, latency, and accuracy.
Across the board, the best is Haiku. That's because, except for Chinese, Sonnet's and Opus' feedback were essentially the same as Haiku's. Given how much slower and expensive they are, Haiku stands as the best option. It is not the best 100% of the time, but often enough to be preferable over the other alternatives.
I believe that this agent already works well if the goal is learning at the beginner and intermediate levels. Any improvements would mainly benefit advanced learners.

## Test Suite

Three test modules, all under `tests/`:

- **`test_schema.py`**: validates request and response schemas against both the provided sample inputs and the candidate inputs. No API key required.
- **`test_feedback_unit.py`**: unit tests that mock the Anthropic client. Tests that the feedback logic correctly parses model responses and returns the expected structure. No API key required.
- **`test_feedback_integration.py`**: 10 tests that make real API calls. 4 cover the original sample languages (Spanish, German, French, Japanese). 6 are based on `examples/candidate_inputs.json` and cover Chinese, Portuguese, Korean, Russian, German, and Japanese, including non-Latin scripts and one correct sentence to test the `is_correct=true` path. The answer key for these sentences was manually verified where possible; thorough verification of non-Latin scripts would require a human linguistic expert, ideally a language educator.

## How to Run

### Local

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 4. Start the server
uvicorn app.main:app --reload
```

### Docker

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
docker compose up --build
```

### Tests

```bash
# Unit and schema tests (no API key needed)
python -m pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires ANTHROPIC_API_KEY in .env)
python -m pytest tests/test_feedback_integration.py -v

# All tests
python -m pytest -v
```

## Scaling Considerations

LLM-based feedback has a hard ceiling on niche grammar rules and low-resource languages (see: Portuguese mesoclisis). Two ideas are worth mentioning:

- **Crowdsourced linguistic curation**: a Wikipedia-style corpus contributed by vetted native speakers and language educators. Decouples coverage from headcount (vs. frontier labs hiring specialists). Trust layer matters: native speaker != linguist != language educator; contributions should be weighted accordingly.

- **Calibration at the implementation layer**: without retraining, the tool can be improved by injecting curated examples into the prompt (few-shot) or adding a retrieval layer (RAG) that pulls verified patterns before the LLM sees the sentence. Hard cases get handled by lookup; the LLM handles the long tail. The crowdsourced corpus and the retrieval layer reinforce each other.