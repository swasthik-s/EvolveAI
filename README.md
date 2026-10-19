# EvolveAI

A mobile-first, instruction-driven chat agent built for Streamlit Community Cloud's free tier.

## Design principle

EvolveAI does **not** call OpenAI or any other hosted language model. Its first version uses explicit Python instructions and deterministic rules. It can store facts the user teaches it, retrieve relevant past examples, handle basic arithmetic, and use 👍 / 👎 feedback to influence which examples are surfaced in future replies.

This is an honest prototype, not AGI. Feedback does not train model weights, and a rule-based system cannot yet answer arbitrary open-ended questions like a general-purpose language model.

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
2. Explicit user-taught facts can be stored as editable memories.
3. Each user prompt and agent response is saved in SQLite.
4. Feedback creates positive or negative examples.
5. Related examples may be retrieved for later prompts.
6. The user can review and delete saved memories from the sidebar.

No generated code is executed. There is no external model dependency.

## Free-cloud storage limitation

The app stores data in `data/evolveai.db`. Streamlit Community Cloud's local filesystem is not guaranteed to persist across restarts, rebuilds, or redeployments. This means the chat's learning data may reset. Durable cross-deployment memory will require a separate persistent database or storage service; this version intentionally avoids adding a paid dependency.

## Tests

Run `pytest -q` locally. The project includes tests for memory and skill storage; the skills module is retained for future reviewed skill drafts, but the current chat UI stays focused on conversation.
