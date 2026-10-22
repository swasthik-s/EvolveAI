# EvolveAI

EvolveAI has two interfaces over the same Python agent:
- **User app:** Next.js + Tailwind + shadcn-style components, intended for Vercel.
- **Admin control room:** a private Streamlit app in `admin/streamlit_app.py`.
- **Engine:** Python rules, conversation state, memory, strategy selection, and feedback in `evolveai/`.
- **API:** `api/chat.py` serves user chat and authenticated admin actions.

No hosted generative model API is required. EvolveAI is a bounded, instruction-driven prototype; web research and relevance ranking are heuristic.

## Symbolic autonomy core (no LLM)

The first model-free autonomy primitives live in `evolveai/autonomy.py`. This module provides:

- Explicit goal definitions, success criteria, constraints, and execution budgets.
- A registry of implemented actions with declared preconditions, effects, cost, risk, reversibility, and approval requirements.
- A bounded symbolic planner that selects eligible actions based on goal progress, cost, and risk.
- A fail-closed policy gate for risky or irreversible actions.
- An observe-plan-act loop with retries, structured events, and goal verification.
- JSON-serializable execution snapshots for persistence and admin inspection.

This is a foundational baseline, not general intelligence or a complete STRIPS/HTN planner. It does not call an LLM, execute arbitrary generated code, or automatically grant new permissions. Application-specific actions must be registered explicitly. Run the autonomy tests with `pytest -q`.


## Deploy the user app on Vercel

1. Import `swasthik-s/EvolveAI` at [Vercel](https://vercel.com/new).
2. Set the root directory to the repository root and use the Next.js preset.
3. Add environment variable `EVOLVEAI_ADMIN_TOKEN` with a long, random secret. This protects administrative API actions; do not expose it to the browser or use a `NEXT_PUBLIC_` variable.
4. Deploy and verify the homepage and `/api/chat`.

## Deploy the private Streamlit control room

Deploy the same repository as a **separate Streamlit Community Cloud app**:

1. Select this repository and the `main` branch.
2. Set **Main file path** to `admin/streamlit_app.py`.
3. Set the app's Python dependencies file to `admin/requirements.txt` if the deployment UI offers that option. If it doesn't, use the root requirements file plus the three admin dependencies listed in `admin/requirements.txt`.
4. In Streamlit app **Settings → Secrets**, set:

```toml
ADMIN_PASSWORD = "a-long-unique-admin-password"
EVOLVEAI_ADMIN_TOKEN = "the-same-random-secret-configured-in-vercel"
EVOLVEAI_API_URL = "https://YOUR-VERCEL-DOMAIN/api/chat"
```

Use a different value for `ADMIN_PASSWORD` and `EVOLVEAI_ADMIN_TOKEN`. The password gates the dashboard; the token authenticates admin actions at the API itself. Never commit real secrets to GitHub. If the Vercel token is rotated, update both deployments.

The dashboard includes a circuit-board overview, engine test console, memory add/delete, lesson and strategy review, Ghost Knowledge search, recent task inspection, and diagnostics. It sends admin actions to the same API/agent used by the user frontend; it does not create a second agent.

## Local development

Frontend:

```bash
npm install
npm run dev
```

Python API:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt uvicorn
uvicorn api.chat:app --reload
```

Admin dashboard (in a second terminal):

```bash
pip install -r admin/requirements.txt
streamlit run admin/streamlit_app.py
```

For local admin testing, set `ADMIN_PASSWORD`, `EVOLVEAI_ADMIN_TOKEN`, and `EVOLVEAI_API_URL=http://localhost:8000/` in your environment or Streamlit secrets. Set the same `EVOLVEAI_ADMIN_TOKEN` in the API environment.

## Storage warning — do not skip

The Vercel API currently defaults to SQLite at `/tmp/evolveai.db`. Vercel function filesystems are ephemeral, and separate instances may not share a database file. The admin dashboard reports this as `sqlite_ephemeral`. **This is not durable production memory.** Before relying on memory, lessons, feedback, or learned strategies across deployments, migrate `ExperienceMemory` to a shared persistent database such as managed PostgreSQL and configure it for the API. That requires a database instance and connection string; neither is created by this repository change.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q
```

The GitHub Actions workflow runs Python tests and the Next.js typecheck/build. Check the workflow status before treating a deployment as production-ready.
