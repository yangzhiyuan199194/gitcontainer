import asyncio
import json
import os
import logging
from typing import Dict, Any, Optional
from fastapi import Request, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from agents.utils import parse_available_models, get_model_stream_support
from agents.workflow import create_workflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Store for session data
sessions = {}


async def home(request: Request):
    """Home page with the input form."""
    # Get available models from environment variable
    available_models = parse_available_models()
    
    # Get current model - first from available models, or fallback to environment or default
    if available_models:
        current_model = available_models[0]["name"]  # Default to first available model
    else:
        current_model = os.getenv("MODEL", "gpt-4o-mini")
    
    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": "",
        "loading": False,
        "streaming": False,
        "result": None,
        "error": None,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


async def dynamic_github_route(request: Request, path: str):
    """Handle GitHub-style URLs by replacing gitcontainer.com with github.com."""
    # Skip certain paths that shouldn't be treated as GitHub routes
    skip_paths = {"health", "favicon.ico", "favicon-16x16.png", "favicon-32x32.png", "apple-touch-icon.png", "static", "ws"}
    
    # Split path into segments
    segments = [segment for segment in path.split('/') if segment]
    
    # If it's a skip path, let it fall through
    if segments and segments[0] in skip_paths:
        raise HTTPException(status_code=404, detail="Not found")
    
    # Check if we have at least 2 segments (username/repo)
    if len(segments) < 2:
        # Get available models from environment variable
        available_models = parse_available_models()
        
        # Get current model - first from available models, or fallback to environment or default
        if available_models:
            current_model = available_models[0]["name"]  # Default to first available model
        else:
            current_model = os.getenv("MODEL", "gpt-4o-mini")
        
        return templates.TemplateResponse("index.jinja", {
            "request": request,
            "repo_url": "",
            "loading": False,
            "streaming": False,
            "result": None,
            "error": f"Invalid GitHub URL format. Expected format: gitcontainer.com/username/repository",
            "pre_filled": False,
            "available_models": [model["name"] for model in available_models],
            "current_model": current_model
        })
    
    # Use only the first two segments (username/repo)
    username, repo = segments[0], segments[1]
    github_url = f"https://github.com/{username}/{repo}"
    
    # Get available models from environment variable
    available_models = parse_available_models()
    
    # Get current model - first from available models, or fallback to environment or default
    if available_models:
        current_model = available_models[0]["name"]  # Default to first available model
    else:
        current_model = os.getenv("MODEL", "gpt-4o-mini")
    
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


async def generate_dockerfile_endpoint(
    request: Request, 
    repo_url: str = Form(...),
    additional_instructions_hidden: str = Form(""),
    model: str = Form(None)
):
    """Redirect to streaming page for Dockerfile generation."""
    # Store the repo URL, additional instructions, and model in a session
    session_id = str(hash(repo_url + str(asyncio.get_event_loop().time())))
    sessions[session_id] = {
        "repo_url": repo_url,
        "additional_instructions": additional_instructions_hidden.strip() if additional_instructions_hidden else "",
        "model": model,
        "status": "pending"
    }
    
    # Redirect to streaming page
    # Get available models from environment variable
    available_models = parse_available_models()
    
    # Get current model from form, or first available model, or environment variable
    if model:
        current_model = model
    elif available_models:
        current_model = available_models[0]["name"]  # Default to first available model
    else:
        current_model = os.getenv("MODEL", "gpt-4o-mini")
    
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


async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming Dockerfile generation with multi-agent reflection."""
    await websocket.accept()

    try:
        if session_id not in sessions:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Invalid session ID"
            }))
            return
        
        repo_url = sessions[session_id]["repo_url"]
        additional_instructions = sessions[session_id].get("additional_instructions", "")
        model = sessions[session_id].get("model", None)
        
        # 编译工作流
        workflow = create_workflow()
        app_workflow = workflow.compile()
        
        # 初始化状态
        from agents.workflow import WorkflowState
        initial_state = WorkflowState(
            repo_url=repo_url,
            additional_instructions=additional_instructions,
            model=model,
            clone_result={},
            analysis_result={},
            dockerfile_result={},
            build_result={},
            reflection_result={},
            iteration=0,
            max_iterations=2,
            final_result={},
            websocket=websocket,
            messages=[]
        )
        
        # 运行工作流
        final_state = await app_workflow.ainvoke(initial_state)
        
        # 构造最终结果 - 无论构建成功与否都发送结果
        final_result = {
            "project_name": final_state["dockerfile_result"].get("project_name", ""),
            "technology_stack": final_state["dockerfile_result"].get("technology_stack", ""),
            "dockerfile": final_state["dockerfile_result"].get("dockerfile", ""),
            "base_image_reasoning": final_state["dockerfile_result"].get("base_image_reasoning", ""),
            "additional_notes": final_state["dockerfile_result"].get("additional_notes", ""),
            "image_build": final_state["build_result"],
            "repo_info": {
                "name": final_state["clone_result"].get("repo_name", ""),
                "size_mb": final_state["clone_result"].get("repo_size_mb", 0),
                "file_count": final_state["clone_result"].get("file_count", 0)
            }
        }
        
        # 发送最终结果 - 无论构建成功与否
        await websocket.send_text(json.dumps({
            "type": "complete",
            "content": "Generation complete!",
            "result": final_result
        }))
        
        # Store result in session for potential refresh
        sessions[session_id]["result"] = final_result
        sessions[session_id]["status"] = "complete"
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        # Check if websocket is still open before trying to send error message
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": error_msg
                }))
        except Exception as send_error:
            print(f"Could not send error message, WebSocket likely closed: {send_error}")
    finally:
        # Clean up session data
        try:
            if session_id in sessions:
                sessions[session_id]["status"] = "disconnected"
        except:
            pass


async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}