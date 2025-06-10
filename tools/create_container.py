import asyncio
import os
import json
from typing import Dict, Any, Optional, Union
from openai import AsyncOpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()


async def create_container_tool(
    gitingest_summary: str,
    gitingest_tree: str,
    gitingest_content: str,
    project_name: Optional[str] = None,
    max_context_chars: int = 50000,  # Limit to stay within context window
    websocket: Optional[Any] = None  # WebSocket connection for streaming
) -> Dict[str, Any]:
    """
    Generate a Dockerfile using OpenAI API based on gitingest context.
    
    Args:
        gitingest_summary (str): Summary from gitingest analysis
        gitingest_tree (str): Directory tree from gitingest
        gitingest_content (str): Full content from gitingest
        project_name (str, optional): Name of the project for the container
        max_context_chars (int): Maximum characters to send in context
        websocket (Any, optional): WebSocket connection for streaming
        
    Returns:
        Dict[str, Any]: Dictionary containing the generated Dockerfile and metadata
    """
    try:
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Truncate content if it exceeds max context to avoid hitting limits
        truncated_content = gitingest_content
        if len(gitingest_content) > max_context_chars:
            truncated_content = gitingest_content[:max_context_chars] + "\n\n... [Content truncated due to length] ..."
        
        # Create the prompt for Dockerfile generation
        prompt = f"""Based on the following repository analysis, generate a comprehensive and production-ready Dockerfile.

PROJECT SUMMARY:
{gitingest_summary}

DIRECTORY STRUCTURE:
{gitingest_tree}

SOURCE CODE CONTEXT:
{truncated_content}

Please generate a Dockerfile that:
1. Uses appropriate base images for the detected technology stack
2. Includes proper dependency management
3. Sets up the correct working directory structure
4. Exposes necessary ports
5. Includes health checks where appropriate
6. Follows Docker best practices (multi-stage builds if beneficial, minimal layers, etc.)
7. Handles environment variables and configuration
8. Sets up proper user permissions for security

If you detect multiple services or a complex architecture, provide a main Dockerfile and suggest docker-compose.yml structure.

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any markdown formatting, explanations, or code blocks. The response must be parseable JSON.

Required JSON format:
{{
  "dockerfile": "FROM python:3.9-slim\\nWORKDIR /app\\nCOPY . .\\nRUN pip install -r requirements.txt\\nEXPOSE 8000\\nCMD [\\"python\\", \\"app.py\\"]",
  "base_image_reasoning": "Explanation of why you chose the base image",
  "technology_stack": "Detected technologies and frameworks",
  "port_recommendations": ["8000", "80"],
  "additional_notes": "Any important setup or deployment notes",
  "docker_compose_suggestion": "Optional docker-compose.yml content if multiple services detected"
}}"""

        # Make API call to generate Dockerfile with streaming
        await _emit_ws_message(websocket, "status", "🐳 Generating Dockerfile...")
        print("🐳 Generating Dockerfile... (streaming response)\n")
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4 for better code generation
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert DevOps engineer specializing in containerization. Generate production-ready Dockerfiles based on repository analysis. ALWAYS respond with valid JSON only - no markdown, no explanations, no code blocks. Just pure JSON that can be parsed directly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            max_tokens=2000,   # Sufficient for Dockerfile generation
            stream=True       # Enable streaming
        )
        
        # Collect the streaming response and print in real-time
        dockerfile_response = ""
        await _emit_ws_message(websocket, "stream_start", "Starting generation...")
        print("📝 Response:")
        print("-" * 50)
        
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                dockerfile_response += content
                # Emit each chunk to WebSocket clients
                await _emit_ws_message(websocket, "chunk", content)
        
        print("\n" + "-" * 50)
        print("✅ Generation complete!\n")
        await _emit_ws_message(websocket, "status", "✅ Generation complete!")
        
        # Try to parse as JSON, fallback to plain text if needed
        try:
            # First try direct JSON parsing
            dockerfile_data = json.loads(dockerfile_response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            try:
                # Look for JSON in code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', dockerfile_response, re.DOTALL)
                if json_match:
                    dockerfile_data = json.loads(json_match.group(1))
                else:
                    # Try to find JSON-like content
                    json_match = re.search(r'(\{[^{}]*"dockerfile"[^{}]*\})', dockerfile_response, re.DOTALL)
                    if json_match:
                        dockerfile_data = json.loads(json_match.group(1))
                    else:
                        raise json.JSONDecodeError("No JSON found", "", 0)
            except (json.JSONDecodeError, AttributeError):
                # If still no valid JSON, treat as plain Dockerfile content
                # Try to extract just the Dockerfile if it looks like one
                dockerfile_content = dockerfile_response
                if 'FROM ' in dockerfile_response:
                    # Extract lines that look like Dockerfile commands
                    lines = dockerfile_response.split('\n')
                    dockerfile_lines = []
                    in_dockerfile = False
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('FROM ') or stripped.startswith('RUN ') or stripped.startswith('COPY ') or stripped.startswith('WORKDIR ') or stripped.startswith('EXPOSE ') or stripped.startswith('CMD ') or stripped.startswith('ENTRYPOINT '):
                            in_dockerfile = True
                            dockerfile_lines.append(line)
                        elif in_dockerfile and (stripped.startswith('#') or stripped == '' or stripped.startswith('ENV ') or stripped.startswith('ARG ') or stripped.startswith('USER ') or stripped.startswith('VOLUME ') or stripped.startswith('LABEL ')):
                            dockerfile_lines.append(line)
                        elif in_dockerfile and not stripped:
                            dockerfile_lines.append(line)
                        elif in_dockerfile and stripped and not any(stripped.startswith(cmd) for cmd in ['FROM', 'RUN', 'COPY', 'WORKDIR', 'EXPOSE', 'CMD', 'ENTRYPOINT', 'ENV', 'ARG', 'USER', 'VOLUME', 'LABEL', '#']):
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
            "project_name": project_name or "generated-project",
            "context_truncated": len(gitingest_content) > max_context_chars,
            "original_content_length": len(gitingest_content),
            "used_content_length": len(truncated_content)
        }
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "project_name": project_name or "unknown-project"
        }
        await _emit_ws_message(websocket, "error", str(e))
        return error_result


async def _emit_ws_message(websocket: Optional[Any], message_type: str, content: str) -> None:
    """Helper function to emit WebSocket messages safely."""
    if websocket is not None:
        try:
            message = {
                "type": message_type,
                "content": content,
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"WebSocket error: {e}")


def run_create_container(
    gitingest_summary: str,
    gitingest_tree: str,
    gitingest_content: str,
    project_name: Optional[str] = None,
    websocket: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the create_container tool.
    
    Args:
        gitingest_summary (str): Summary from gitingest analysis
        gitingest_tree (str): Directory tree from gitingest
        gitingest_content (str): Full content from gitingest
        project_name (str, optional): Name of the project
        websocket (Any, optional): WebSocket connection for streaming
        
    Returns:
        Dict[str, Any]: Dictionary containing generated Dockerfile and metadata
    """
    return asyncio.run(create_container_tool(
        gitingest_summary, gitingest_tree, gitingest_content, project_name, websocket=websocket
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
                }
            },
            "required": ["gitingest_summary", "gitingest_tree", "gitingest_content"]
        }
    }
} 