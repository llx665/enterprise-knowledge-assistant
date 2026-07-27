"""Render deployment entry point for EnterpriseKB.
Reads PORT from environment (set by Render or render.yaml) and starts uvicorn."""
import os
import sys

# Ensure project root is on sys.path so "backend" package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    # Render sets PORT env var (default 10000). render.yaml overrides to 8081.
    port = int(os.environ.get("PORT", 8081))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        log_level="info",
    )
