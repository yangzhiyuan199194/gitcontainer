"""Minimal FastAPI app for GitHub URL to Dockerfile generator."""

import asyncio
import json
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from tools import gitingest_tool, clone_repo_tool, create_container_tool

# Initialize FastAPI app
app = FastAPI(title="GitHub to Dockerfile Generator")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files (we'll create this directory)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Store for session data
sessions = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with the input form."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "repo_url": "",
        "loading": False,
        "streaming": False,
        "result": None,
        "error": None
    })


@app.post("/", response_class=HTMLResponse) 
async def generate_dockerfile(request: Request, repo_url: str = Form(...)):
    """Redirect to streaming page for Dockerfile generation."""
    # Store the repo URL in a session (simple in-memory for demo)
    session_id = str(hash(repo_url + str(asyncio.get_event_loop().time())))
    sessions[session_id] = {"repo_url": repo_url, "status": "pending"}
    
    # Redirect to streaming page
    return templates.TemplateResponse("index.html", {
        "request": request,
        "repo_url": repo_url,
        "loading": False,
        "streaming": True,
        "session_id": session_id,
        "result": None,
        "error": None
    })


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming Dockerfile generation."""
    await websocket.accept()
    
    try:
        if session_id not in sessions:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Invalid session ID"
            }))
            return
        
        repo_url = sessions[session_id]["repo_url"]
        
        # Step 1: Clone repository
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": f"🔄 Cloning repository: {repo_url}"
        }))
        
        clone_result = await clone_repo_tool(repo_url)
        
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
        
        ingest_result = await gitingest_tool(clone_result['local_path'])
        
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
        
        container_result = await create_container_tool(
            gitingest_summary=ingest_result['summary'],
            gitingest_tree=ingest_result['tree'], 
            gitingest_content=ingest_result['content'],
            project_name=clone_result['repo_name'],
            websocket=websocket  # Pass WebSocket for streaming
        )
        
        if not container_result["success"]:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Failed to generate Dockerfile: {container_result['error']}"
            }))
            return
        
        # Send final result
        final_result = {
            "project_name": container_result['project_name'],
            "technology_stack": container_result['technology_stack'],
            "dockerfile": container_result['dockerfile'],
            "docker_compose": container_result.get('docker_compose_suggestion', ''),
            "reasoning": container_result.get('base_image_reasoning', ''),
            "additional_notes": container_result.get('additional_notes', ''),
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
        await websocket.send_text(json.dumps({
            "type": "error",
            "content": f"Unexpected error: {str(e)}"
        }))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 