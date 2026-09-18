"""Backward-compatible ASGI entry point for the live demo."""

from server.main import app


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=os.getenv("DEMO_HOST", "127.0.0.1"),
        port=int(os.getenv("DEMO_PORT", "8008")),
    )
