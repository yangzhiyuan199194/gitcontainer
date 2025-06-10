import asyncio
from gitingest import ingest_async
from typing import Dict, Any


async def gitingest_tool(github_url: str) -> Dict[str, Any]:
    """
    Analyze a GitHub repository using gitingest and return structured results.
    
    Args:
        github_url (str): The GitHub repository URL to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing summary, tree, and content
    """
    try:
        # Use gitingest to analyze the repository
        summary, tree, content = await ingest_async(
            source=github_url,
            max_file_size=5 * 1024 * 1024,  # 5MB limit per file
        )
        
        return {
            "success": True,
            "summary": summary,
            "tree": tree,
            "content": content,
            "url": github_url
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": github_url
        }


def run_gitingest(github_url: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for the gitingest tool.
    
    Args:
        github_url (str): The GitHub repository URL to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing analysis results
    """
    return asyncio.run(gitingest_tool(github_url))


# Tool definition for OpenAI Agents SDK
gitingest_function = {
    "type": "function",
    "function": {
        "name": "analyze_github_repo",
        "description": "Analyze a GitHub repository and extract its structure and content for analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "github_url": {
                    "type": "string",
                    "description": "The GitHub repository URL to analyze (e.g., https://github.com/user/repo)"
                }
            },
            "required": ["github_url"]
        }
    }
} 