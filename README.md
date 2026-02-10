# BFHL API

Simple FastAPI implementation for the Qualifier assignment.

Endpoints

- POST /bfhl
  - Accepts exactly one key: fibonacci (int), prime ([int]), lcm ([int]), hcf ([int]), AI (string)
  - Success response (200): {"is_success": true, "official_email": "YOUR CHITKARA EMAIL", "data": ...}
  - Errors return appropriate HTTP status codes and {"is_success": false, "official_email": "...", "error": "..."}

- GET /health
  - Returns {"is_success": true, "official_email": "YOUR CHITKARA EMAIL"}

Environment

Create a `.env` with:

OFFICIAL_EMAIL=your_chitkara_email_here
OPENAI_API_KEY=your_openai_api_key_here
PORT=8000

Install & Run (local):

1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt
4. copy .env to a real .env and fill values
5. python main.py

Deployment

- Railway / Render / Vercel: push this repo to GitHub and follow provider steps. Ensure environment variables `OFFICIAL_EMAIL` and `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) are configured.

Provider config files included:
- `vercel.json` — Vercel build settings
- `render.yaml` — Render service template
- `Procfile` — For Railway/Heroku style deploy

CI

- GitHub Actions workflow is included at `.github/workflows/ci.yml` to run tests on push/PR.

AI Provider

Set `AI_PROVIDER` to `openai` (default) or `anthropic`. For Anthropic, set `ANTHROPIC_API_KEY` in environment. Gemini support is not implemented — set `AI_PROVIDER=gemini` only after adding provider credentials and implementation.

Notes

- The AI integration uses OpenAI; replace with other provider by editing `main.py`.
- Request payloads larger than 10KB are rejected (413).
- Fibonacci has a safety cap (n <= 1000).
