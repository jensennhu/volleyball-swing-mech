"""Entry point: run the platform server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "spike_platform.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["spike_platform"],
    )
