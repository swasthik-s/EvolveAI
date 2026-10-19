# EvolveAI

A mobile-first, instruction-driven chat agent built for Streamlit Community Cloud's free tier.

## Design principle

EvolveAI does **not** call OpenAI or any other hosted language model. Its first version uses explicit Python instructions and deterministic rules. It can store facts the user teaches it, retrieve relevant past examples, handle basic arithmetic, and use 👍 / 👎 feedback to influence which examples are surfaced in future replies. For factual questions outside its current rules and saved knowledge, it can try free, keyless public web search and save source excerpts as Ghost Knowledge.

This is an honest prototype, not AGI. Feedback does not train model weights, and a rule-based system cannot yet answer arbitrary open-ended questions like a general-purpose language model. The deterministic reasoning planner classifies the message, resolves short follow-ups against recent chat context, chooses a route, checks saved knowledge, and exposes its execution stages in the UI. Web research separates retrieval from response generation: it extracts paragraph text, filters boilerplate, ranks relevant sentences, removes duplicates, builds a concise evidence-based answer, and shows source links. This is a deterministic evidence-processing engine, not human-like semantic reasoning; its scores are heuristics and can still miss context or make mistakes.

## Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/
2. Create an app from repository `swasthik-s/EvolveAI`.
3. Select branch `main` and main file `app.py`.
4. Deploy. No API key or external model credentials are required.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How the learning loop works

1. The agent follows the policies in `evolveai/instructions.py`.
2. It checks programmed rules, personal memories, and Ghost Knowledge before researching.
3. For a factual question that appears outside its known rules, it tries keyless public web search.
4. It extracts page paragraphs, ranks relevant sentences, filters common navigation noise, and synthesizes a short answer with source links.
5. The full relevant excerpts and source URLs are saved as Ghost Knowledge candidates; later questions reuse and process that evidence rather than dumping the raw page text.
6. Time-sensitive questions are researched again.
7. User feedback creates positive or negative examples that can influence future retrieval.
8. The sidebar lets the user review personal memories and saved Ghost Knowledge.

No generated code is executed. There is no external model API dependency. The `ddgs` search library and public websites may be rate-limited or block automated requests; research can fail without an API key.

## Free-cloud storage limitation

The app stores data in `data/evolveai.db`. Streamlit Community Cloud's local filesystem is not guaranteed to persist across restarts, rebuilds, or redeployments. This means the chat's learning data may reset. Durable cross-deployment memory will require a separate persistent database or storage service; this version intentionally avoids adding a paid dependency.

## Tests

Run `pytest -q` locally. The project includes tests for memory, Ghost Knowledge retrieval, the uncertainty-triggered research decision, and deterministic evidence synthesis; the skills module is retained for future reviewed skill drafts, but the current chat UI stays focused on conversation.

## Custom understanding engine (no generative model)

The first cognition layer lives in `evolveai/cognition.py`. It parses common question forms into a structured frame, canonicalizes a small reviewed set of entity aliases (for example, “ironman” → “Iron Man”), records why a rule matched, and produces a more specific web-search query. Unknown subjects are preserved rather than confidently guessed. The agent uses this frame before web retrieval, and tests cover canonicalization and safe fallback.

This borrows *ideas*, not code, from established NLP approaches: spaCy demonstrates practical pipelines for linguistic features and entity processing ([spaCy 101](https://spacy.io/usage/spacy-101/)); ConceptNet provides a commonsense relation graph that can inform a later knowledge layer ([ConceptNet API](https://github.com/commonsense/conceptnet5/wiki/API)). EvolveAI currently does not depend on either package or service. We are implementing a small, transparent engine first so each decision can be tested and explained. Its entity lexicon is intentionally small; it is not yet general-purpose language understanding.
