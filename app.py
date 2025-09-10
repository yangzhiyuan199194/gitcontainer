"""Minimal FastAPI app for GitHub URL to Dockerfile generator."""

import asyncio
import json
import os
from pathlib import Path

from api_analytics.fastapi import Analytics
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tools import gitingest_tool, clone_repo_tool, create_container_tool, build_docker_image

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="GitHub to Dockerfile Generator")

# Add API Analytics middleware
app.add_middleware(Analytics, api_key=os.getenv("FASTAPI_ANALYTICS_KEY"))

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files (we'll create this directory)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Store for session data
sessions = {}


def parse_available_models():
    """
    Parse the AVAILABLE_MODELS environment variable.
    Format: model_name|stream_support,model_name|stream_support,...
    Example: gpt-4o-mini|true,gpt-4o|true,o1-mini|false,o1|false
    """
    available_models_str = os.getenv("AVAILABLE_MODELS", "")
    if not available_models_str:
        return []
    
    models = []
    for model_entry in available_models_str.split(","):
        model_entry = model_entry.strip()
        if "|" in model_entry:
            model_name, stream_support = model_entry.split("|")
            models.append({
                "name": model_name.strip(),
                "stream": stream_support.strip().lower() == "true"
            })
        else:
            # For backward compatibility, if no | is present, assume stream is supported
            models.append({
                "name": model_entry,
                "stream": True
            })
    return models


def get_model_stream_support(model_name):
    """
    Check if a specific model supports streaming based on AVAILABLE_MODELS environment variable.
    """
    models = parse_available_models()
    for model in models:
        if model["name"] == model_name:
            return model["stream"]
    # Default to True if model not found in the list
    return True


@app.get("/favicon.ico")
async def favicon():
    """Serve the main favicon."""
    return FileResponse("static/icons8-docker-doodle-32.png")


@app.get("/favicon-16x16.png")
async def favicon_16():
    """Serve 16x16 favicon."""
    return FileResponse("static/icons8-docker-doodle-16.png")


@app.get("/favicon-32x32.png") 
async def favicon_32():
    """Serve 32x32 favicon."""
    return FileResponse("static/icons8-docker-doodle-32.png")


@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    """Serve Apple touch icon (120x120 is close to the 180x180 standard)."""
    return FileResponse("static/icons8-docker-doodle-120.png")


@app.get("/", response_class=HTMLResponse)
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


@app.get("/{path:path}", response_class=HTMLResponse)
async def dynamic_github_route(request: Request, path: str):
    """Handle GitHub-style URLs by replacing gitcontainer.com with github.com."""
    # Skip certain paths that shouldn't be treated as GitHub routes
    skip_paths = {"health", "favicon.ico", "favicon-16x16.png", "favicon-32x32.png", "apple-touch-icon.png", "static", "ws"}
    
    # Split path into segments
    segments = [segment for segment in path.split('/') if segment]
    
    # If it's a skip path, let it fall through
    if segments and segments[0] in skip_paths:
        from fastapi import HTTPException
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


@app.post("/{path:path}", response_class=HTMLResponse)
async def dynamic_github_route_post(
    request: Request,
    path: str,
    repo_url: str = Form(...),
    additional_instructions_hidden: str = Form(""),
    model: str = Form(None)
):
    """Handle POST requests for GitHub-style URLs, reusing the generate_dockerfile logic."""
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


@app.post("/", response_class=HTMLResponse) 
async def generate_dockerfile(
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


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming Dockerfile generation."""
    await websocket.accept()
    print(f"New WebSocket connection: {session_id}")
    
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
        
        # Determine if the selected model supports streaming
        stream_support = get_model_stream_support(model) if model else True
        
        # Step 1: Clone repository
        await websocket.send_text(json.dumps({
            "type": "status", 
            "content": f"🔄 Cloning repository: {repo_url}"
        }))
        
        clone_result = await clone_repo_tool(repo_url, websocket=websocket)
        
        if not clone_result["success"]:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Failed to clone repository: {clone_result['error']}"
            }))
            return
        
        # Step 2: Analyze with gitingest
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "📊 Analyzing repository structure..."
        }))
        
        ingest_result = await gitingest_tool(clone_result['local_path'], websocket=websocket)
        
        if not ingest_result["success"]:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Failed to analyze repository: {ingest_result['error']}"
            }))
            return
        
        # Step 3: Generate Dockerfile with streaming
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🐳 Generating Dockerfile with AI..."
        }))
        
        # Prepare arguments for create_container_tool
        container_args = {
            "gitingest_summary": ingest_result['summary'],
            "gitingest_tree": ingest_result['tree'], 
            "gitingest_content": ingest_result['content'],
            "project_name": clone_result['repo_name'],
            "websocket": websocket,  # Pass WebSocket for streaming
            "additional_instructions": additional_instructions,
            "stream": stream_support  # Set stream based on model support
        }
        
        # Add model parameter if specified
        if model:
            container_args["model"] = model
        
        container_result = await create_container_tool(**container_args)
        
        if not container_result["success"]:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Failed to generate Dockerfile: {container_result['error']}"
            }))
            return
        
        # Step 4: Build Docker image
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🔨 Building Docker image..."
        }))
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": "🚀 开始构建 Docker 镜像...\n"
            }))
        
        build_result = await build_docker_image(
            dockerfile_content=container_result['dockerfile'],
            project_name=clone_result['repo_name'],
            local_path=clone_result['local_path'],
            websocket=websocket
        )
        
        # Send build result information
        if websocket:
            if build_result["success"]:
                await websocket.send_text(json.dumps({
                    "type": "build_log",
                    "content": f"✅ Docker 镜像构建完成: {build_result['image_tag']}\n"
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "build_log",
                    "content": f"❌ Docker 镜像构建失败: {build_result.get('error', 'Unknown error')}\n"
                }))
                # 发送错误消息，确保前端能正确更新步骤状态
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": f"Docker 镜像构建失败: {build_result.get('error', 'Unknown error')}"
                }))
        
        # Send final result
        final_result = {
            "project_name": container_result['project_name'],
            "technology_stack": container_result['technology_stack'],
            "dockerfile": container_result['dockerfile'],
            # "docker_compose": container_result.get('docker_compose_suggestion', ''),
            "reasoning": container_result.get('base_image_reasoning', ''),
            "additional_notes": container_result.get('additional_notes', ''),
            "image_build": build_result,
            "repo_info": {
                "name": clone_result['repo_name'],
                "size_mb": clone_result['repo_size_mb'],
                "file_count": clone_result['file_count']
            }
        }
        
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    # Load environment variables
    load_dotenv()
    PORT = int(os.getenv("PORT",8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)