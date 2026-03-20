"""
FastAPI application factory for the volleyball spike detector platform.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from spike_platform.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup."""
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Volleyball Spike Detector",
        description="Human-in-the-loop training platform for volleyball spike detection",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow mobile app and dev origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    from spike_platform.routers import videos, segments, training, phases, analysis, mobile
    app.include_router(videos.router, prefix="/api", tags=["videos"])
    app.include_router(segments.router, prefix="/api", tags=["segments"])
    app.include_router(training.router, prefix="/api", tags=["training"])
    app.include_router(phases.router, prefix="/api", tags=["phases"])
    app.include_router(analysis.router, prefix="/api", tags=["analysis"])
    app.include_router(mobile.router, prefix="/api", tags=["mobile"])

    # Serve frontend static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
