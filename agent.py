"""Agent-based system for GitHub to Dockerfile generation with reflection capabilities."""

import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api_analytics.fastapi import Analytics
from dotenv import load_dotenv

from agents.routes import (
    home, 
    dynamic_github_route, 
    generate_dockerfile_endpoint, 
    websocket_endpoint,
    health_check
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="GitHub to Dockerfile Generator - Agent Version")

# Add API Analytics middleware
app.add_middleware(Analytics, api_key=os.getenv("FASTAPI_ANALYTICS_KEY"))

# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Add routes
app.get("/")(home)
app.get("/{path:path}")(dynamic_github_route)
app.post("/")(generate_dockerfile_endpoint)
app.websocket("/ws/{session_id}")(websocket_endpoint)
app.get("/health")(health_check)


if __name__ == "__main__":
    import uvicorn
    # Load environment variables
    load_dotenv()
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)