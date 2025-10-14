"""
API routes for Gitcontainer application.

This module defines all the HTTP routes and WebSocket endpoints for the application.
"""

import asyncio
import json
import logging
import os
import hashlib
from typing import Dict, Any

from fastapi import APIRouter, Request, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from autobuild.core.config import Settings
from autobuild.services.workflow import create_workflow
from autobuild.services.langgraph_workflow import create_langgraph_workflow, LangGraphWorkflowState
from autobuild.utils.session_manager import SessionManager
from autobuild.utils.build_history import build_history_manager

# Setup logging
logger = logging.getLogger(__name__)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Setup router
router = APIRouter()

# Session manager
session_manager = SessionManager()

# Application settings
settings = Settings()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page with the input form and list of successful builds.
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        TemplateResponse: Rendered home page
    """
    available_models = settings.get_available_models()

    # Get current model - first from available models, or fallback to environment or default
    current_model = (
        available_models[0]["name"] if available_models
        else settings.model
    )

    # Check if there's a model parameter in the request
    request_model = request.query_params.get("model")
    if request_model and request_model in [model["name"] for model in available_models]:
        current_model = request_model

    # Get successful build records
    successful_builds = build_history_manager.get_successful_builds()

    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": "",
        "loading": False,
        "streaming": False,
        "result": None,
        "error": None,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model,
        "successful_builds": successful_builds
    })


@router.get("/builds/{repo_name}", response_class=HTMLResponse)
async def build_detail(request: Request, repo_name: str):
    """
    Build detail page showing build logs and results.
    
    Args:
        request (Request): FastAPI request object
        repo_name (str): Repository name (derived from GitHub URL)
        
    Returns:
        TemplateResponse: Rendered build detail page
    """
    # Find the build record by repo name
    build_records = build_history_manager.get_all_build_records()
    build_record = None

    for record in build_records:
        record_identifier = record["repo_url"].replace('https://github.com/', '').replace('http://github.com/',
                                                                                          '').replace('/', '_')
        if record_identifier == repo_name:
            build_record = record
            break

    if not build_record:
        raise HTTPException(status_code=404, detail="Build record not found")

    # Load build logs if they exist
    log_content = build_history_manager.get_build_log(repo_url=build_record["repo_url"])

    available_models = settings.get_available_models()
    current_model = (
        available_models[0]["name"] if available_models
        else settings.model
    )

    return templates.TemplateResponse("build_detail.jinja", {
        "request": request,
        "build_record": build_record,
        "log_content": log_content,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


@router.get("/{path:path}", response_class=HTMLResponse)
async def dynamic_github_route(request: Request, path: str):
    """
    Handle GitHub-style URLs by replacing autobuild.com with github.com.
    
    Args:
        request (Request): FastAPI request object
        path (str): URL path
        
    Returns:
        TemplateResponse: Rendered home page with pre-filled repo URL
    """
    # Skip certain paths that shouldn't be treated as GitHub routes
    skip_paths = {
        "health", "favicon.ico", "favicon-16x16.png",
        "favicon-32x32.png", "apple-touch-icon.png", "static", "ws", "builds"
    }

    # Split path into segments
    segments = [segment for segment in path.split('/') if segment]

    # If it's a skip path, let it fall through
    if segments and segments[0] in skip_paths:
        raise HTTPException(status_code=404, detail="Not found")

    # Check if we have at least 2 segments (username/repo)
    if len(segments) < 2:
        available_models = settings.get_available_models()
        current_model = (
            available_models[0]["name"] if available_models
            else settings.model
        )

        return templates.TemplateResponse("index.jinja", {
            "request": request,
            "repo_url": "",
            "loading": False,
            "streaming": False,
            "result": None,
            "error": "Invalid GitHub URL format. Expected format: autobuild.com/username/repository",
            "pre_filled": False,
            "available_models": [model["name"] for model in available_models],
            "current_model": current_model
        })

    # Use only the first two segments (username/repo)
    username, repo = segments[0], segments[1]
    github_url = f"https://github.com/{username}/{repo}"

    # Check if this repo has already been built successfully
    if build_history_manager.repo_already_built_successfully(github_url):
        # Redirect to build detail page
        repo_hash = github_url.replace('https://github.com/', '').replace('http://github.com/', '').replace('/', '_')
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/builds/{repo_name}")

    # Get available models from environment variable
    available_models = settings.get_available_models()
    current_model = (
        available_models[0]["name"] if available_models
        else settings.model
    )

    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": github_url,
        "loading": False,
        "streaming": False,
        "result": None,
        "error": None,
        "pre_filled": True,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


@router.post("/", response_class=HTMLResponse)
async def generate_dockerfile_endpoint(
        request: Request,
        repo_url: str = Form(...),
        additional_instructions_hidden: str = Form(""),
        model: str = Form(None),
        generate_wiki: str = Form(None)
):
    """
    Redirect to streaming page for Dockerfile generation.
    
    Args:
        request (Request): FastAPI request object
        repo_url (str): GitHub repository URL
        additional_instructions_hidden (str): Additional instructions for Dockerfile generation
        model (str): Model to use for generation
        
    Returns:
        TemplateResponse: Rendered page with streaming enabled
    """
    # Check if this repo has already been built successfully
    if build_history_manager.repo_already_built_successfully(repo_url):
        # Redirect to build detail page
        repo_name = repo_url.replace('https://github.com/', '').replace('http://github.com/', '').replace('/', '_')
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/builds/{repo_name}", status_code=303)

    # Store the repo URL, additional instructions, model, and generate_wiki flag in a session
    session_id = session_manager.create_session({
        "repo_url": repo_url,
        "additional_instructions": (
            additional_instructions_hidden.strip()
            if additional_instructions_hidden
            else ""
        ),
        "model": model,
        "generate_wiki": generate_wiki == "on"  # Convert checkbox value to boolean
    })

    # Get available models
    available_models = settings.get_available_models()
    current_model = model or (
        available_models[0]["name"] if available_models
        else settings.model
    )

    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": repo_url,
        "loading": False,
        "streaming": True,
        "session_id": session_id,
        "result": None,
        "error": None,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming Dockerfile generation with multi-agent reflection.
    
    Args:
        websocket (WebSocket): WebSocket connection
        session_id (str): Session identifier
    """
    await websocket.accept()

    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            from autobuild.utils import get_websocket_manager
            ws_manager = get_websocket_manager(websocket)
            await ws_manager.send_error("Invalid session ID")
            return

        repo_url = session_data["repo_url"]
        additional_instructions = session_data.get("additional_instructions", "")
        model = session_data.get("model", None)

        # Create log file path using GitHub URL transformation
        repo_name = repo_url.replace('https://github.com/', '').replace('http://github.com/', '').replace('/', '_')
        ws_log_file_path = f"build_history/logs/{repo_name}.log"

        # Initialize WebSocket manager with log file
        from autobuild.utils import get_websocket_manager
        ws_manager = get_websocket_manager(websocket, ws_log_file_path)

        # Get generate_wiki flag from session data, default to True if not present
        generate_wiki = session_data.get("generate_wiki", True)
        
        # Use enhanced LangGraph workflow with automatic message forwarding
        workflow = create_langgraph_workflow()
        app_workflow = workflow.compile()

        app_workflow.get_graph().print_ascii()
        
        # Send workflow metadata to frontend immediately after connection
        try:
            await ws_manager.send_workflow_metadata(workflow.metadata)
            await ws_manager.send_status("🚀 工作流启动中...")
        except Exception as metadata_error:
            logger.error(f"Error sending initial workflow metadata: {str(metadata_error)}")

        # Initialize enhanced state with auto message handling
        initial_state = LangGraphWorkflowState(
            repo_url=repo_url,
            additional_instructions=additional_instructions,
            model=model,
            generate_wiki=generate_wiki,  # Pass generate_wiki flag to workflow state
            clone_result={},
            analysis_result={},
            dockerfile_result={},
            build_result={},
            reflection_result={},
            iteration=0,
            max_iterations=settings.max_iterations,
            final_result={},
            websocket=websocket,
            ws_log_file_path=ws_log_file_path,
            messages=[],
            current_node=None,
            node_status={},
            wiki_result={}
        )

        # Run workflow
        logger.info(f"Starting workflow execution for repository: {repo_url}")
        final_state = await app_workflow.ainvoke(initial_state)
        logger.info(f"Workflow execution completed for repository: {repo_url}")

        # Construct enhanced final result with workflow state information
        final_result = {
            "project_name": final_state["dockerfile_result"].get("project_name", ""),
            "technology_stack": final_state["dockerfile_result"].get("technology_stack", ""),
            "dockerfile": final_state["dockerfile_result"].get("dockerfile", ""),
            "base_image_reasoning": final_state["dockerfile_result"].get("base_image_reasoning", ""),
            "additional_notes": final_state["dockerfile_result"].get("additional_notes", ""),
            # 添加验证代码到最终结果中
            "verification_code": final_state["dockerfile_result"].get("verification_code", ""),
            # 添加运行资源规格到最终结果中
            "resource_requirements": final_state["dockerfile_result"].get("resource_requirements", {}),
            "image_build": final_state["build_result"],
            "repo_info": {
                "name": final_state["clone_result"].get("repo_name", ""),
                "size_mb": final_state["clone_result"].get("repo_size_mb", 0),
                "file_count": final_state["clone_result"].get("file_count", 0)
            },
            # Add workflow execution information for frontend
            "workflow_info": {
                "current_node": final_state.get("current_node"),
                "node_status": final_state.get("node_status", {}),
                "iteration": final_state.get("iteration", 0),
                "max_iterations": final_state.get("max_iterations", 0),
                "success": final_state["build_result"].get("success", False)
            }
        }

        # Send final result - regardless of build success
        logger.info(f"Sending final result to frontend for repository: {repo_url}")
        await ws_manager.send_complete("Generation complete!", final_result)

        if "wiki_result" in final_state and final_state["wiki_result"]:
            final_result["wiki_result"] = final_state["wiki_result"]

        # Save build record
        build_data = {
            "success": True,
            "result": final_result,
            "repo_url": repo_url
        }
        build_history_manager.save_build_record(repo_url, build_data)

        # Store result in session for potential refresh
        session_manager.update_session(session_id, {
            "result": final_result,
            "status": "complete"
        })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Check if websocket is still open before trying to send error message
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": error_msg
                }))
        except Exception as send_error:
            logger.error(f"Could not send error message, WebSocket likely closed: {send_error}", exc_info=True)

        # Save failed build record if repo_url is available
        try:
            session_data = session_manager.get_session(session_id)
            if session_data and "repo_url" in session_data:
                repo_url = session_data["repo_url"]
                build_data = {
                    "success": False,
                    "error": error_msg,
                    "repo_url": repo_url
                }
                build_history_manager.save_build_record(repo_url, build_data)
        except Exception as save_error:
            logger.error(f"Could not save failed build record: {save_error}", exc_info=True)

    finally:
        # Clean up session data
        try:
            session_manager.update_session(session_id, {"status": "disconnected"})
        except Exception as se:
            logger.error(f"Could not update_session disconnected: {se}", exc_info=True)
            pass


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {"status": "healthy"}
