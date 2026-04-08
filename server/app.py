# server/app.py — Re-export for OpenEnv auto-discovery
# The validator expects `server/app.py` to contain the FastAPI `app` instance
# and a callable `main()` for CLI entry.

from server.main import app  # noqa: F401
import uvicorn


def main():
    """Entry point for `openenv serve` and [project.scripts]."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
