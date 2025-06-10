import asyncio
import os
from gitingest import ingest_async
from typing import Dict, Any


async def gitingest_tool(local_repo_path: str) -> Dict[str, Any]:
    """
    Analyze a local GitHub repository using gitingest and return structured results.
    
    Args:
        local_repo_path (str): The local path to the cloned repository to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing summary, tree, and content
    """
    try:
        # Check if the local path exists
        if not os.path.exists(local_repo_path):
            raise FileNotFoundError(f"Local repository path does not exist: {local_repo_path}")
        
        if not os.path.isdir(local_repo_path):
            raise ValueError(f"Path is not a directory: {local_repo_path}")
        
        # Use gitingest to analyze the local repository
        summary, tree, content = await ingest_async(
            source=local_repo_path,
            max_file_size=5 * 1024 * 1024,  # 5MB limit per file
        )
        
        return {
            "success": True,
            "summary": summary,
            "tree": tree,
            "content": content,
            "local_path": local_repo_path
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
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