import asyncio
import os
import json
from gitingest import ingest_async
from typing import Dict, Any, Optional


async def gitingest_tool(local_repo_path: str, websocket: Optional[Any] = None) -> Dict[str, Any]:
    """
    Analyze a local GitHub repository using gitingest and return structured results.
    
    Args:
        local_repo_path (str): The local path to the cloned repository to analyze
        websocket (Optional[Any]): WebSocket connection for streaming output
        
    Returns:
        Dict[str, Any]: Dictionary containing summary, tree, and content
    """
    try:
        # Check if the local path exists
        if not os.path.exists(local_repo_path):
            raise FileNotFoundError(f"Local repository path does not exist: {local_repo_path}")
        
        if not os.path.isdir(local_repo_path):
            raise ValueError(f"Path is not a directory: {local_repo_path}")
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": "🔍 Starting repository analysis...\n"
            }))
        
        # Use gitingest to analyze the local repository with timeout
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": "📂 Scanning directory structure...\n"
            }))
        
        try:
            # Add timeout to prevent hanging
            summary, tree, content = await asyncio.wait_for(
                ingest_async(
                    source=local_repo_path,
                    max_file_size=5 * 1024 * 1024,  # 5MB limit per file
                ),
                timeout=120.0  # 2 minute timeout
            )
        except asyncio.TimeoutError:
            error_msg = "Repository analysis timed out (took more than 2 minutes). The repository might be too large."
            if websocket:
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": f"❌ {error_msg}\n"
                }))
            return {
                "success": False,
                "error": error_msg,
                "local_path": local_repo_path
            }
        
        if websocket:
            # Check the type of content and handle accordingly
            file_count = 0
            if isinstance(content, dict):
                file_count = len(content.get('files', []))
            elif isinstance(content, str):
                # If content is a string, we can't get file count from it
                file_count = "unknown"
            
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": f"📊 Analysis complete. Found {file_count} files.\n"
            }))
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": "🧠 Analyzing technology stack...\n"
            }))
        
        return {
            "success": True,
            "summary": summary,
            "tree": tree,
            "content": content,
            "local_path": local_repo_path
        }
        
    except Exception as e:
        error_msg = f"Error during repository analysis: {str(e)}"
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": f"❌ {error_msg}\n"
            }))
        return {
            "success": False,
            "error": error_msg,
            "local_path": local_repo_path
        }


def run_gitingest(local_repo_path: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for the gitingest tool.
    
    Args:
        local_repo_path (str): The local path to the cloned repository to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing analysis results
    """
    return asyncio.run(gitingest_tool(local_repo_path))


# Tool definition for OpenAI Agents SDK
gitingest_function = {
    "type": "function",
    "function": {
        "name": "analyze_local_repo",
        "description": "Analyze a locally cloned repository and extract its structure and content for analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": "The local file system path to the cloned repository (e.g., ./repos/my-repo)"
                }
            },
            "required": ["local_repo_path"]
        }
    }
}