import asyncio
import logging
import os
import json
from gitingest import ingest_async
from typing import Dict, Any, Optional

# 添加 LLMClient 导入
from tools.llm_client import LLMClient

logger = logging.getLogger(__name__)


async def search_dockerfile(local_repo_path: str, summary: str, tree: str, content: str,
                            websocket: Optional[Any] = None) -> Optional[str]:
    """
    Search for existing Dockerfile in the repository.
    
    Args:
        local_repo_path (str): The local path to the cloned repository
        
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
        if not dockerfile_paths:
            return None
        else:
            # Convert dockerfile_paths to a newline-separated string
            paths = "\n".join(dockerfile_paths)
            if websocket:
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": f"The paths of the current project Dockerfiles:{paths}\n"
                }))

        # 如果只有一个 Dockerfile，直接返回其内容
        if len(dockerfile_paths) == 1:
            with open(dockerfile_paths[0], 'r', encoding='utf-8') as f:
                return f.read()

        # 如果有多个 Dockerfile，使用大模型分析选择最适合的
        dockerfile_contents = {}
        for path in dockerfile_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    dockerfile_contents[path] = f.read()
            except Exception as e:
                logger.warning(f"Could not read Dockerfile at {path}: {str(e)}")
                # 即使读取失败，也保留路径，内容为空字符串
                dockerfile_contents[path] = ""

        # 使用大模型分析选择最适合的 Dockerfile
        selected_dockerfile = await select_best_dockerfile(dockerfile_contents, summary, tree, content, websocket)
        return selected_dockerfile

    except Exception as e:
        logger.error(f"Error searching for Dockerfile: {str(e)}")
        return None


async def select_best_dockerfile(dockerfile_contents: Dict[str, str], summary: str, tree: str, content: str,
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
    try:
        # 初始化LLM客户端
        llm_client = LLMClient()
        
        # 构建 Dockerfile 信息列表（仅路径）
        dockerfile_paths = list(dockerfile_contents.keys())
        dockerfile_list = []
        for path in dockerfile_paths:
            dockerfile_list.append(f"PATH: {path}")
        
        dockerfiles_section = "\n".join(dockerfile_list)
        
        # 构建基础 prompt（不包含具体内容）
        base_prompt = f"""Based on the repository analysis, select the most appropriate Dockerfile for containerizing this project.

PROJECT SUMMARY:
{summary}

DIRECTORY STRUCTURE:
{tree}

AVAILABLE DOCKERFILES:
{dockerfiles_section}

Please analyze the Dockerfiles and the project context, then select the most appropriate Dockerfile for this project. Consider:
1. Which Dockerfile best matches the project's technology stack
2. Which Dockerfile follows the best practices
3. Which Dockerfile is most likely to successfully build the project
4. Which Dockerfile is the most complete and production-ready

Respond ONLY with the file path of the selected Dockerfile. Do not include any explanations or additional text."""

        # 估算 prompt 长度（英文字符大约 4:1 与 token 的比例）
        prompt_tokens = len(base_prompt) // 4
        max_tokens = 1000  # LLM 返回的最大 token 数
        max_context_tokens = 128000  # 假设使用大模型，上下文窗口为 128K tokens
        available_tokens = max_context_tokens - prompt_tokens - max_tokens
        
        # 计算所有 Dockerfile 内容的 token 数
        total_dockerfile_tokens = 0
        for content in dockerfile_contents.values():
            total_dockerfile_tokens += len(content) // 4  # 简单估算 token 数
        
        # 计算源代码内容的 token 数
        source_content_tokens = len(content) // 4
        
        # 总需要的 token 数
        total_needed_tokens = prompt_tokens + total_dockerfile_tokens + source_content_tokens + max_tokens
        
        # 如果 token 数在限制范围内，则添加具体 Dockerfile 内容和源代码内容
        detailed_dockerfiles_section = ""
        source_code_section = ""
        if total_needed_tokens <= max_context_tokens:
            dockerfile_details = []
            for path, content in dockerfile_contents.items():
                dockerfile_details.append(f"PATH: {path}\nCONTENT:\n{content}\n" + "="*50)
            detailed_dockerfiles_section = "\n\n".join(dockerfile_details)
            
            # 添加源代码内容
            source_code_section = f"SOURCE CODE CONTEXT:\n{content[:10000]}\n"  # 使用更长的上下文
            
            # 更新 prompt，包含详细内容
            base_prompt = f"""Based on the repository analysis, select the most appropriate Dockerfile for containerizing this project.

PROJECT SUMMARY:
{summary}

DIRECTORY STRUCTURE:
{tree}

{source_code_section}
AVAILABLE DOCKERFILES WITH CONTENT:
{detailed_dockerfiles_section}

Please analyze the Dockerfiles and the project context, then select the most appropriate Dockerfile for this project. Consider:
1. Which Dockerfile best matches the project's technology stack
2. Which Dockerfile follows the best practices
3. Which Dockerfile is most likely to successfully build the project
4. Which Dockerfile is the most complete and production-ready

Respond ONLY with the file path of the selected Dockerfile. Do not include any explanations or additional text."""
        else:
            # 如果超出 token 限制，添加提示信息
            source_code_section = f"SOURCE CODE CONTEXT (first 1000 chars):\n{content[:1000]}\n"  # 使用较短的上下文
            base_prompt += "\n\nNote: Due to context length limitations, the specific content of Dockerfiles is not provided. Please make your selection based on the file paths and project context."

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "You are an expert DevOps engineer specializing in Docker containerization. Select the most appropriate Dockerfile for a project based on repository analysis. Respond ONLY with the file path of the selected Dockerfile."
            },
            {
                "role": "user",
                "content": base_prompt
            }
        ]

        # 发送状态消息
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": "🧠 Analyzing multiple Dockerfiles to select the most appropriate one...\n"
            }))

        # 调用 LLM 进行分析
        result = await llm_client.call_llm(
            messages=messages,
            temperature=0.3,
            websocket=websocket
        )
        
        if result["success"]:
            selected_path = result["content"].strip()
            # 检查返回的路径是否在我们的列表中
            if selected_path in dockerfile_contents:
                if websocket:
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "content": f"✅ Selected Dockerfile: {selected_path}\n"
                    }))
                return dockerfile_contents[selected_path]
            else:
                # 如果返回的路径不在列表中，尝试模糊匹配
                for path in dockerfile_contents.keys():
                    if path in selected_path or selected_path in path:
                        if websocket:
                            await websocket.send_text(json.dumps({
                                "type": "chunk",
                                "content": f"✅ Selected Dockerfile: {path}\n"
                            }))
                        return dockerfile_contents[path]
                
                # 如果无法匹配，使用第一个 Dockerfile
                first_path = list(dockerfile_contents.keys())[0]
                if websocket:
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "content": f"⚠️ Could not match selected path, using: {first_path}\n"
                    }))
                return dockerfile_contents[first_path]
        else:
            # 如果 LLM 调用失败，使用第一个 Dockerfile
            first_path = list(dockerfile_contents.keys())[0]
            if websocket:
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": f"⚠️ Failed to analyze Dockerfiles, using: {first_path}\n"
                }))
            return dockerfile_contents[first_path]

    except Exception as e:
        logger.error(f"Error selecting best Dockerfile: {str(e)}")
        # 出错时返回第一个 Dockerfile
        if dockerfile_contents:
            first_path = list(dockerfile_contents.keys())[0]
            if websocket:
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": f"⚠️ Error during analysis, using: {first_path}\n"
                }))
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

            logger.info("summary: %s", summary)
            logger.info("tree: %s", tree)

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
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": f"📊 Analysis complete. summary: {summary}\n"
            }))
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": f"📊 Analysis complete. tree: {tree}\n"
            }))
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": "🧠 Analyzing technology stack...\n"
            }))

        # Search for existing Dockerfile
        git_dockerfile = await search_dockerfile(local_repo_path, summary, tree, content, websocket)

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
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "chunk",
                "content": f"❌ {error_msg}\n"
            }))
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
