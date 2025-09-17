import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from tools.llm_client import LLMClient
from tools.utils import emit_ws_message

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dockerfile_prompt(
    gitingest_summary: str,
    gitingest_tree: str,
    truncated_content: str,
    git_dockerfile: Optional[str] = None,
    additional_instructions_section: str = ""
) -> str:
    """
    创建Dockerfile生成的prompt
    
    Args:
        gitingest_summary: 项目摘要
        gitingest_tree: 目录结构
        truncated_content: 截断的内容
        git_dockerfile: 现有的Dockerfile内容
        additional_instructions_section: 附加指令
        
    Returns:
        构建好的prompt字符串
    """
    # Prepare Dockerfile information section
    dockerfile_info_section = ""
    if git_dockerfile:
        dockerfile_info_section = f"\nTHE PROJECT ALREADY HAS DOCKERFILE INFORMATION:\n{git_dockerfile}\n"
    else:
        dockerfile_info_section = "\nTHE PROJECT DOES NOT CONTAIN A DOCKERFILE CURRENTLY.\n"

    return f"""Based on the following repository analysis, generate a comprehensive and production-ready Dockerfile that will successfully build without errors.

PROJECT SUMMARY:
{gitingest_summary}{dockerfile_info_section}
DIRECTORY STRUCTURE:
{gitingest_tree}

SOURCE CODE CONTEXT:
{truncated_content}{additional_instructions_section}

Please generate a Dockerfile that:
1. If there is already Dockerfile information in the project, give priority to referring to this information
2. Uses appropriate base images for the detected technology stack
3. Includes proper dependency management with attention to version compatibility
4. Sets up the correct working directory structure
5. Exposes necessary ports based on the application type
6. Includes health checks where appropriate for the application type
7. Follows Docker best practices:
   - Use multi-stage builds when beneficial to reduce image size
   - Combine related RUN commands to minimize layers
   - Properly handle package manager caches (e.g., apt-get clean, rm -rf /var/lib/apt/lists/*)
   - Use COPY instead of ADD unless specifically needed
   - Set proper user permissions for security (do not run as root if possible)
8. Handles environment variables and configuration appropriately
9. Ensures all commands are compatible with the chosen base image
10. Avoids common build errors:
    - Always use specific versions for base images (avoid 'latest')
    - Use a definite existing base image and do not fabricate non-existent images at will
    - Properly escape special characters in commands
    - Ensure all required files are copied or created before being used
    - Handle platform-specific dependencies correctly
    - Install build dependencies before runtime dependencies where applicable
    - Properly configure the entrypoint and command for the application type

If you detect multiple services or a complex architecture, provide a main Dockerfile for the primary service and suggest a docker-compose.yml structure.

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any explanations, or code blocks. The response must be parseable JSON.

Required JSON format:
{{
  "dockerfile": "FROM python:3.9-slim\\nWORKDIR /app\\nCOPY . .\\nRUN pip install -r requirements.txt\\nEXPOSE 8000\\nCMD [\\"python\\", \\"app.py\\"]",
  "base_image_reasoning": "Explanation of why you chose the base image, including why it will successfully build",
  "technology_stack": "Detected technologies and frameworks",
  "port_recommendations": ["8000", "80"],
  "additional_notes": "Any important setup or deployment notes, including potential build issues and how to avoid them",
  "docker_compose_suggestion": "Optional docker-compose.yml content if multiple services detected"
}}"""


def create_reflection_prompt(
    dockerfile_content: str,
    build_log: str,
    error_message: str,
    gitingest_summary: str,
    gitingest_tree: str,
    truncated_content: str
) -> str:
    """
    创建用于分析 Docker 构建失败原因并提出改进建议的 prompt
    
    Args:
        dockerfile_content: 当前 Dockerfile 内容
        build_log: 构建日志
        error_message: 错误信息
        gitingest_summary: 项目摘要
        gitingest_tree: 目录结构
        truncated_content: 截断的内容
        
    Returns:
        构建好的 prompt 字符串
    """
    return f"""Based on the following Docker build failure information, analyze the root cause of the failure and provide specific improvement suggestions for the Dockerfile.

CURRENT DOCKERFILE:
{dockerfile_content}

BUILD ERROR MESSAGE:
{error_message}

BUILD LOG (last 2000 characters):
{build_log[-2000:] if build_log else "No build log available"}

PROJECT SUMMARY:
{gitingest_summary}

DIRECTORY STRUCTURE:
{gitingest_tree}

SOURCE CODE CONTEXT:
{truncated_content}

Please analyze the Docker build failure and provide:
1. Root cause analysis of why the build failed
2. Specific issues in the Dockerfile that contributed to the failure
3. Detailed suggestions for fixing the Dockerfile
4. A revised Dockerfile that addresses the identified issues

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any explanations or code blocks. The response must be parseable JSON.

Required JSON format:
{{
  "root_cause": "Detailed explanation of the root cause of the build failure",
  "issues": [
    "Specific issue 1 in the Dockerfile",
    "Specific issue 2 in the Dockerfile"
  ],
  "suggestions": [
    "Detailed suggestion 1 for fixing the Dockerfile",
    "Detailed suggestion 2 for fixing the Dockerfile"
  ],
  "revised_dockerfile": "FROM python:3.9-slim\\nWORKDIR /app\\nCOPY . .\\nRUN pip install -r requirements.txt\\nEXPOSE 8000\\nCMD [\\"python\\", \\"app.py\\"]"
}}"""


async def handle_dockerfile_response(response_content: str) -> Dict[str, Any]:
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
                "docker_compose_suggestion": None
            }

    return {
        "success": True,
        "dockerfile": dockerfile_data.get("dockerfile", ""),
        "base_image_reasoning": dockerfile_data.get("base_image_reasoning", ""),
        "technology_stack": dockerfile_data.get("technology_stack", ""),
        "port_recommendations": dockerfile_data.get("port_recommendations", []),
        "additional_notes": dockerfile_data.get("additional_notes", ""),
        "docker_compose_suggestion": dockerfile_data.get("docker_compose_suggestion"),
    }


async def create_container_tool(
        gitingest_summary: str,
        gitingest_tree: str,
        gitingest_content: str,
        git_dockerfile: str = None,
        project_name: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        max_context_chars: int = 50000,  # Limit to stay within context window
        websocket: Optional[Any] = None,  # WebSocket connection for streaming
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
            websocket=websocket,
            response_handler=handle_dockerfile_response
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
        # Only send error message if WebSocket might still be active
        try:
            await emit_ws_message(websocket, "error", str(e))
        except:
            # WebSocket is definitely closed, just log the error
            print(f"Could not send error to WebSocket: {e}")
        return error_result


def run_create_container(
        gitingest_summary: str,
        gitingest_tree: str,
        gitingest_content: str,
        project_name: Optional[str] = None,
        additional_instructions: Optional[str] = None,
        websocket: Optional[Any] = None,
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
        websocket (Any, optional): WebSocket connection for streaming
        model (str, optional): Model to use for generation
        
    Returns:
        Dict[str, Any]: Dictionary containing generated Dockerfile and metadata
    """
    return asyncio.run(create_container_tool(
        gitingest_summary, gitingest_tree, gitingest_content, project_name, additional_instructions,
        websocket=websocket, model=model
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