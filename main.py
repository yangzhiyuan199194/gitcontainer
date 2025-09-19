"""
Gitcontainer - AI-powered Dockerfile Generator

This is the main entry point for the Gitcontainer application.
"""

import os
import logging

from dotenv import load_dotenv

from autobuild.core.app import app

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import uvicorn
    
    # Load environment variables
    load_dotenv()
    
    # Get server configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"Starting Gitcontainer on {host}:{port}")
    
    uvicorn.run(
        "autobuild.core.app:app",
        host=host,
        port=port,
        reload=debug
    )