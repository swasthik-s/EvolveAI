# EvolveAI

EvolveAI is a custom, mobile-friendly assistant UI built with **Next.js, Tailwind CSS, and shadcn-style components**, connected to the existing Python instruction-driven engine. The intended deployment target is one Vercel project.

## Product principles

- **No hosted generative model:** responses come from explicit Python rules, saved context, and deterministic processing of public web research.
- **Inspectable behavior:** the engine classifies requests, selects a response path, and records its strategy.
- **User feedback:** 👍 / 👎 feedback is recorded and can influence future retrieval and communication-strategy selection.
- **Honest limitations:** this is a bounded prototype, not a general-purpose LLM. Research ranking is heuristic and may miss context.

## Deploy on Vercel

1. Push this repository to GitHub.
2. In [Vercel](https://vercel.com/new), import `swasthik-s/EvolveAI`.
3. Keep the project root set to the repository root.
4. Use the Next.js framework preset. Vercel should detect `package.json` and build the frontend.
5. Deploy, then verify the home page and the `/api/chat` endpoint.

The frontend is a Next.js App Router app at `app/`. The Python function is in `api/chat.py`; chat and feedback actions share the same function so they can reuse the same warm-instance memory. Python dependencies are in the root `requirements.txt`.

## Local development

Frontend:

```bash
npm install
npm run dev
```

Python API (requires Python 3.11+ recommended):

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt uvicorn
uvicorn api.chat:app --reload
```

The Python function is configured for Vercel's function runtime. For local API testing, the command above serves the chat function on port 8000; use a local rewrite/proxy or adjust the frontend request URL during local testing if needed.

## Storage warning

The Vercel Python function currently defaults to SQLite at `/tmp/evolveai.db`. **Vercel's function filesystem is ephemeral**: data may disappear between cold starts, and separate function instances may not share the same file. Browser chat threads are stored in the user's local browser, but that is not the same as durable server-side memory.

Before relying on long-term personal memories or learning in production, configure a durable database and point the engine at it. Do not store sensitive personal data in ephemeral function storage and assume it will persist.

## Existing Python engine and tests

The current engine remains under `evolveai/`. To run the existing tests locally:

```bash
pytest -q
```

No paid model API is required. The free public web-search dependency may be rate-limited or blocked by upstream websites.
