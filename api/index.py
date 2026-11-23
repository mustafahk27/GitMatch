"""
Vercel entrypoint — imports the FastAPI app so Vercel's Python runtime can serve it.
"""
from src.app.main import app  # noqa: F401 — Vercel picks up the ASGI app named `app`
