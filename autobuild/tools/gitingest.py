"""
Prompts for the gitingest module.

This module contains prompt templates used by the gitingest analysis tool.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional

from gitingest import ingest_async

from autobuild.prompts.gitingest import create_dockerfile_selection_prompt
from autobuild.services.llm_client import LLMClient
from autobuild.utils import get_websocket_manager

logger = logging.getLogger(__name__)


async def search_dockerfile(local_repo_path: str, summary: str, tree: str, content: str,
                            websocket: Optional[Any] = None) -> Optional[str]:
    """
    Search for existing Dockerfile in the repository.
    
    Args:
        local_repo_path (str): The local path to the cloned repository
        summary (str): Project summary from gitingest
        tree (str): Directory tree from gitingest
        content (str): Source code content from gitingest
        websocket (Optional[Any]): WebSocket connection for streaming output
        
    Returns:
        Optional[str]: Content of the shortest path Dockerfile if found, None otherwise
    """
    try:
        # Find all Dockerfile paths in the repository
        dockerfile_paths = []

        # Walk through the directory structure
        for root, dirs, files in os.walk(local_repo_path):
            for file in files:
                if file.lower().startswith("dockerfile"):
                    dockerfile_paths.append(os.path.join(root, file))

        # If no Dockerfile found, return None
        # Initialize WebSocket manager
        ws_manager = get_websocket_manager(websocket)
        
        if not dockerfile_paths:
            return None
        else:
            # Convert dockerfile_paths to a newline-separated string
            paths = "\n".join(dockerfile_paths)
            await ws_manager.send_chunk(f"The paths of the current project Dockerfiles:{paths}\n")

        # If there's only one Dockerfile, return its content directly
        if len(dockerfile_paths) == 1:
            with open(dockerfile_paths[0], 'r', encoding='utf-8') as f:
                return f.read()

        # If there are multiple Dockerfiles, use LLM to analyze and select the most appropriate one
        dockerfile_contents = {}
        for path in dockerfile_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    dockerfile_contents[path] = f.read()
            except Exception as e:
                logger.warning(f"Could not read Dockerfile at {path}: {str(e)}")
                # Even if reading fails, keep the path with empty content
                dockerfile_contents[path] = ""

        # Use LLM to analyze and select the most appropriate Dockerfile
        selected_dockerfile = await _select_best_dockerfile(
            dockerfile_contents, summary, tree, content, websocket
        )
        return selected_dockerfile

    except Exception as e:
        logger.error(f"Error searching for Dockerfile: {str(e)}")
        return None


async def _select_best_dockerfile(dockerfile_contents: Dict[str, str], summary: str, tree: str, content: str,
                                 websocket: Optional[Any] = None) -> Optional[str]:
    """
    Use LLM to analyze and select the most appropriate Dockerfile.
    
    Args:
        dockerfile_contents: Dictionary mapping file paths to their contents
        summary: Project summary from gitingest
        tree: Directory tree from gitingest
        content: Source code content from gitingest
        websocket: WebSocket connection for streaming output
        
    Returns:
        Content of the selected Dockerfile or None if selection failed
    """
    # Initialize WebSocket manager
    ws_manager = get_websocket_manager(websocket)
    
    try:
        # Initialize LLM client
        llm_client = LLMClient()
        
        # Get Dockerfile paths
        dockerfile_paths = list(dockerfile_contents.keys())
        
        # Create the prompt using the dedicated prompt module
        base_prompt = create_dockerfile_selection_prompt(
            dockerfile_paths=dockerfile_paths,
            summary=summary,
            tree=tree,
            content=content
        )

        # Build messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert DevOps engineer specializing in Docker containerization. "
                    "Select the most appropriate Dockerfile for a project based on repository analysis. "
                    "Respond ONLY with the file path of the selected Dockerfile."
                )
            },
            {
                "role": "user",
                "content": base_prompt
            }
        ]

        # Send status message
        await ws_manager.send_chunk("🧠 Analyzing multiple Dockerfiles to select the most appropriate one...\n")

        # Call LLM for analysis
        result = await llm_client.call_llm(
            messages=messages,
            temperature=0.3,
            websocket=websocket
        )
        
        if result["success"]:
            selected_path = result["content"].strip()
            # Check if returned path is in our list
            if selected_path in dockerfile_contents:
                await ws_manager.send_chunk(f"✅ Selected Dockerfile: {selected_path}\n")
                return dockerfile_contents[selected_path]
            else:
                # If returned path is not in list, try fuzzy matching
                for path in dockerfile_contents.keys():
                    if path in selected_path or selected_path in path:
                        await ws_manager.send_chunk(f"✅ Selected Dockerfile: {path}\n")
                        return dockerfile_contents[path]
                
                # If unable to match, use first Dockerfile
                first_path = list(dockerfile_contents.keys())[0]
                await ws_manager.send_chunk(f"⚠️ Could not match selected path, using: {first_path}\n")
                return dockerfile_contents[first_path]
        else:
            # If LLM call fails, use first Dockerfile
            first_path = list(dockerfile_contents.keys())[0]
            await ws_manager.send_chunk(f"⚠️ Failed to analyze Dockerfiles, using: {first_path}\n")
            return dockerfile_contents[first_path]

    except Exception as e:
        logger.error(f"Error selecting best Dockerfile: {str(e)}")
        # On error, return first Dockerfile
        if dockerfile_contents:
            first_path = list(dockerfile_contents.keys())[0]
            await ws_manager.send_chunk(f"⚠️ Error during analysis, using: {first_path}\n")
            return dockerfile_contents[first_path]
        return None


async def gitingest_tool(local_repo_path: str, websocket: Optional[Any] = None) -> Dict[str, Any]:
    """
    Analyze a local GitHub repository using gitingest and return structured results.
    
    Args:
        local_repo_path (str): The local path to the cloned repository to analyze
        websocket (Optional[Any]): WebSocket connection for streaming output
        
    Returns:
        Dict[str, Any]: Dictionary containing summary, tree, content, and git_dockerfile
    """
    # Initialize WebSocket manager
    ws_manager = get_websocket_manager(websocket)
    
    try:
        # Check if the local path exists
        if not os.path.exists(local_repo_path):
            raise FileNotFoundError(f"Local repository path does not exist: {local_repo_path}")

        if not os.path.isdir(local_repo_path):
            raise ValueError(f"Path is not a directory: {local_repo_path}")

        await ws_manager.send_chunk("🔍 Starting repository analysis...\n")

        # Use gitingest to analyze the local repository with timeout
        await ws_manager.send_chunk("📂 Scanning directory structure...\n")

        try:
            # Add timeout to prevent hanging
            summary, tree, content = await asyncio.wait_for(
                ingest_async(
                    source=local_repo_path,
                    max_file_size=5 * 1024 * 1024,  # 5MB limit per file
                ),
                timeout=120.0  # 2 minute timeout
            )

            logger.info("summary: %s", summary)
            logger.info("tree: %s", tree)

        except asyncio.TimeoutError:
            error_msg = (
                "Repository analysis timed out (took more than 2 minutes). "
                "The repository might be too large."
            )
            await ws_manager.send_chunk(f"❌ {error_msg}\n")
            return {
                "success": False,
                "error": error_msg,
                "local_path": local_repo_path
            }

        # Check the type of content and handle accordingly
        await ws_manager.send_chunk(f"📊 Analysis complete. summary: {summary}\n")
        await ws_manager.send_chunk(f"📊 Analysis complete. tree: {tree}\n")
        await ws_manager.send_chunk("🧠 Analyzing technology stack...\n")

        # Search for existing Dockerfile
        git_dockerfile = await search_dockerfile(
            local_repo_path, summary, tree, content, websocket
        )

        logger.info("git_dockerfile: %s", git_dockerfile)
        return {
            "success": True,
            "summary": summary,
            "tree": tree,
            "content": content,
            "local_path": local_repo_path,
            "git_dockerfile": git_dockerfile
        }
    except Exception as e:
        error_msg = f"Error during repository analysis: {str(e)}"
        await ws_manager.send_chunk(f"❌ {error_msg}\n")
    return {
        "success": False,
        "error": error_msg,
        "local_path": local_repo_path,
        "git_dockerfile": None
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
        "description": (
            "Analyze a locally cloned repository and extract its structure and content for analysis"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "local_repo_path": {
                    "type": "string",
                    "description": (
                        "The local file system path to the cloned repository "
                        "(e.g., ./repos/my-repo)"
                    )
                }
            },
            "required": ["local_repo_path"]
        }
    }
}