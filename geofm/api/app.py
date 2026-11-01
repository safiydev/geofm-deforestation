"""FastAPI application factory and entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from geofm.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="GeoFM Deforestation Detection & EUDR Compliance API",
        description=(
            "Automated Geospatial AI early warning system for deforestation detection, "
            "bitemporal change analysis, and EUDR (Regulation EU 2023/1115) regulatory audits."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
