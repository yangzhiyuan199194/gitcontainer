"""
Main application module for Gitcontainer.

This module sets up the FastAPI application, configures middleware,
and defines the main routes.
"""

import logging
from pathlib import Path

from api_analytics.fastapi import Analytics
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from autobuild.api.routes import router
from autobuild.core.config import Settings

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = Settings()
    
    # Initialize FastAPI app
    app = FastAPI(
        title="Gitcontainer - AI-powered Dockerfile Generator",
        description="Generate production-ready Dockerfiles from GitHub repositories using AI",
        version="1.0.0"
    )
    
    # Add API Analytics middleware if key is provided
    if settings.fastapi_analytics_key:
        app.add_middleware(Analytics, api_key=settings.fastapi_analytics_key)
    
    # Mount static files
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Include API routes
    app.include_router(router)
    
    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info("Gitcontainer application starting up...")
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Gitcontainer application shutting down...")
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = Settings()
    uvicorn.run(
        "autobuild.core.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )