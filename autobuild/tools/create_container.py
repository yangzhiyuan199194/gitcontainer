"""
Prompts module for Gitcontainer application.
Contains prompt templates for various AI interactions.
"""
"""
Prompts for Dockerfile generation in Gitcontainer application.
"""


import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from autobuild.prompts.dockerfile import create_dockerfile_prompt
from autobuild.services.llm_client import LLMClient

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _handle_dockerfile_response(response_content: str) -> Dict[str, Any]:
    """
    处理Dockerfile生成的响应内容
    
    Args:
        response_content: LLM返回的响应内容
        
    Returns:
        解析后的Dockerfile数据
    """
    # 尝试解析为JSON，如果需要则回退到纯文本
    try:
        # 首先尝试直接JSON解析
        dockerfile_data = json.loads(response_content)
    except json.JSONDecodeError:
        # 尝试从代码块中提取JSON
        try:
            # 在代码块中查找JSON
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                dockerfile_data = json.loads(json_match.group(1))
            else:
                # 尝试查找JSON-like内容
                json_match = re.search(r'(\{[^{}]*"dockerfile"[^{}]*\})', response_content, re.DOTALL)
                if json_match:
                    dockerfile_data = json.loads(json_match.group(1))
                else:
                    raise json.JSONDecodeError("No JSON found", "", 0)
        except (json.JSONDecodeError, AttributeError):
            # 如果仍然没有有效的JSON，则视为纯Dockerfile内容
            # 尝试提取看起来像Dockerfile的内容
            dockerfile_content = response_content
            if 'FROM ' in response_content:
                # 提取看起来像Dockerfile命令的行
                lines = response_content.split('\n')
                dockerfile_lines = []
                in_dockerfile = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('FROM ') or stripped.startswith('RUN ') or stripped.startswith(
                            'COPY ') or stripped.startswith('WORKDIR ') or stripped.startswith(
                        'EXPOSE ') or stripped.startswith('CMD ') or stripped.startswith('ENTRYPOINT '):
                        in_dockerfile = True
                        dockerfile_lines.append(line)
                    elif in_dockerfile and (stripped.startswith('#') or stripped == '' or stripped.startswith(
                            'ENV ') or stripped.startswith('ARG ') or stripped.startswith(
                        'USER ') or stripped.startswith('VOLUME ') or stripped.startswith('LABEL ')):
                        dockerfile_lines.append(line)
                    elif in_dockerfile and not stripped:
                        dockerfile_lines.append(line)
                    elif in_dockerfile and stripped and not any(stripped.startswith(cmd) for cmd in
                                                                ['FROM', 'RUN', 'COPY', 'WORKDIR', 'EXPOSE', 'CMD',
                                                                 'ENTRYPOINT', 'ENV', 'ARG', 'USER', 'VOLUME',
                                                                 'LABEL', '#']):
                        break

                if dockerfile_lines:
                    dockerfile_content = '\n'.join(dockerfile_lines).strip()

            dockerfile_data = {
                "dockerfile": dockerfile_content,
                "base_image_reasoning": "Generated as plain text response",
                "technology_stack": "Could not parse detailed analysis",
                "port_recommendations": [],
                "additional_notes": "Response was not in expected JSON format",
                "docker_compose_suggestion": None,
                "verification_code": {
                    "language": "bash",
                    "code": "#!/bin/bash\n# Build and run the Docker image\ndocker build -t my-app .\ndocker run -p 8000:8000 my-app\n",
                    "description": "Basic verification script",
                    "dependencies": ["docker"]
                }
            }

    return {
        "success": True,
        "dockerfile": dockerfile_data.get("dockerfile", ""),
        "base_image_reasoning": dockerfile_data.get("base_image_reasoning", ""),
        "technology_stack": dockerfile_data.get("technology_stack", ""),
        "port_recommendations": dockerfile_data.get("port_recommendations", []),
        "additional_notes": dockerfile_data.get("additional_notes", ""),
        "docker_compose_suggestion": dockerfile_data.get("docker_compose_suggestion"),
        "verification_code": dockerfile_data.get("verification_code", {})
    }


async def create_container_tool(
        gitingest_summary: str,
        gitingest_tree: str,
        gitingest_content: str,
        git_dockerfile: str = None,
        project_name: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        max_context_chars: int = 50000,  # Limit to stay within context window
        ws_manager: Optional[Any] = None,  # WebSocket connection for streaming
        model: Optional[str] = None,  # Model to use for generation
        stream: bool = True  # Whether to stream the response
) -> Dict[str, Any]:
    """
    Generate a Dockerfile using OpenAI API based on gitingest context.
    
    Args:
        gitingest_summary (str): Summary from gitingest analysis
        gitingest_tree (str): Directory tree from gitingest
        gitingest_content (str): Full content from gitingest
        project_name (str, optional): Name of the project for the container
        additional_instructions (str, optional): Additional instructions for the Dockerfile generation
        max_context_chars (int): Maximum characters to send in context
        websocket (Any, optional): WebSocket connection for streaming
        model (str, optional): Model to use for generation
        stream (bool): Whether to stream the response
        
    Returns:
        Dict[str, Any]: Dictionary containing the generated Dockerfile and metadata
    """

    try:
        # 初始化LLM客户端
        llm_client = LLMClient()

        # Truncate content if it exceeds max context to avoid hitting limits
        truncated_content = gitingest_content
        if len(gitingest_content) > max_context_chars:
            truncated_content = gitingest_content[:max_context_chars] + "\n\n... [Content truncated due to length] ..."

        # Create the prompt for Dockerfile generation
        additional_instructions_section = ""
        if additional_instructions and additional_instructions.strip():
            additional_instructions_section = f"\n\nADDITIONAL INSTRUCTIONS:\n{additional_instructions.strip()}"

        # 构建prompt
        prompt = create_dockerfile_prompt(
            gitingest_summary, 
            gitingest_tree, 
            truncated_content, 
            git_dockerfile, 
            additional_instructions_section
        )

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": "You are an expert DevOps engineer specializing in containerization. Generate production-ready Dockerfiles based on repository analysis. ALWAYS respond with valid JSON only - no explanations, no code blocks. Just pure JSON that can be parsed directly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # 调用LLM
        result = await llm_client.call_llm(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=20000,
            stream=stream,
            ws_manager=ws_manager,
            response_handler=_handle_dockerfile_response
        )
        
        # 添加额外的元数据
        if result["success"]:
            result["project_name"] = project_name or "generated-project"
            result["context_truncated"] = len(gitingest_content) > max_context_chars
            result["original_content_length"] = len(gitingest_content)
            result["used_content_length"] = len(truncated_content)
        
        return result

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "project_name": project_name or "unknown-project"
        }
        # Send error message
        if ws_manager:
            await ws_manager.send_error(str(e))
        return error_result


def run_create_container(
        gitingest_summary: str,
        gitingest_tree: str,
        gitingest_content: str,
        project_name: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        ws_manager: Optional[Any] = None,
        model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the create_container tool.
    
    Args:
        gitingest_summary (str): Summary from gitingest analysis
        gitingest_tree (str): Directory tree from gitingest
        gitingest_content (str): Full content from gitingest
        project_name (str, optional): Name of the project
        additional_instructions (str, optional): Additional instructions for Dockerfile generation
        ws_manager (Any, optional): WebSocket connection for streaming
        model (str, optional): Model to use for generation
        
    Returns:
        Dict[str, Any]: Dictionary containing generated Dockerfile and metadata
    """
    return asyncio.run(create_container_tool(
        gitingest_summary, gitingest_tree, gitingest_content, project_name, additional_instructions,
        ws_manager=ws_manager, model=model
    ))


# Tool definition for OpenAI Agents SDK
create_container_function = {
    "type": "function",
    "function": {
        "name": "generate_dockerfile",
        "description": "Generate a production-ready Dockerfile based on repository analysis from gitingest",
        "parameters": {
            "type": "object",
            "properties": {
                "gitingest_summary": {
                    "type": "string",
                    "description": "Summary of the repository from gitingest analysis"
                },
                "gitingest_tree": {
                    "type": "string",
                    "description": "Directory tree structure from gitingest"
                },
                "gitingest_content": {
                    "type": "string",
                    "description": "Full source code content from gitingest"
                },
                "project_name": {
                    "type": "string",
                    "description": "Optional name for the project/container"
                },
                "additional_instructions": {
                    "type": "string",
                    "description": "Optional additional instructions for customizing the Dockerfile generation"
                },
                "model": {
                    "type": "string",
                    "description": "Optional model to use for generation"
                }
            },
            "required": ["gitingest_summary", "gitingest_tree", "gitingest_content"]
        }
    }
}