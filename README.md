# EvolveAI

An experimental Python + Streamlit agent that improves through recorded experience, explicit feedback, reusable skill drafts, and evaluation.

This is an evolving-agent prototype, not AGI. It does not automatically train its underlying model or execute generated code.

## Run locally

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

Without OPENAI_API_KEY, the app runs in demo mode. Set OPENAI_API_KEY and optionally EVOLVEAI_MODEL in your environment or Streamlit Secrets. Never hardcode secrets.

## Tests

    pytest -q

## Learning loop
1. Retrieve relevant lessons from SQLite.
2. Generate a response using the configured model or demo mode.
3. Store the task and result.
4. Collect user feedback.
5. Create candidate lessons from positive feedback.
6. Review skill drafts and approve or reject them.

Feedback and stored lessons are not model-weight training. The dashboard reports feedback metrics rather than claiming that the underlying model has become more intelligent.

Streamlit Community Cloud is suitable for a prototype, but local filesystem data may not persist across restarts. Configure persistent storage for durable use. Generated skills are not executed automatically.