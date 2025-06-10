"""Minimal FastAPI app for GitHub URL to Dockerfile generator."""

import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with the input form."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "repo_url": "",
        "loading": False,
        "result": None,
        "error": None
    })


@app.post("/", response_class=HTMLResponse) 
async def generate_dockerfile(request: Request, repo_url: str = Form(...)):
    """Process GitHub URL and generate Dockerfile."""
    
    # Initial response with loading state
    context = {
        "request": request,
        "repo_url": repo_url,
        "loading": True,
        "result": None,
        "error": None
    }
    
    try:
        # Step 1: Clone repository
        print(f"Cloning repository: {repo_url}")
        clone_result = await clone_repo_tool(repo_url)
        
        if not clone_result["success"]:
            context["loading"] = False
            context["error"] = f"Failed to clone repository: {clone_result['error']}"
            return templates.TemplateResponse("index.html", context)
        
        # Step 2: Analyze with gitingest
        print(f"Analyzing repository...")
        ingest_result = await gitingest_tool(clone_result['local_path'])
        
        if not ingest_result["success"]:
            context["loading"] = False
            context["error"] = f"Failed to analyze repository: {ingest_result['error']}"
            return templates.TemplateResponse("index.html", context)
        
        # Step 3: Generate Dockerfile
        print(f"Generating Dockerfile...")
        container_result = await create_container_tool(
            gitingest_summary=ingest_result['summary'],
            gitingest_tree=ingest_result['tree'], 
            gitingest_content=ingest_result['content'],
            project_name=clone_result['repo_name']
        )
        
        if not container_result["success"]:
            context["loading"] = False
            context["error"] = f"Failed to generate Dockerfile: {container_result['error']}"
            return templates.TemplateResponse("index.html", context)
        
        # Success! Prepare result
        context["loading"] = False
        context["result"] = {
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
        
    except Exception as e:
        context["loading"] = False
        context["error"] = f"Unexpected error: {str(e)}"
    
    return templates.TemplateResponse("index.html", context)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 