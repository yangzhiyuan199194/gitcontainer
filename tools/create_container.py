import asyncio
import json
import logging
import os
import re
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from tools.utils import emit_ws_message

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        inf_api_key = os.getenv("INF_API_KEY")
        default_model = os.getenv("MODEL", "gpt-4o-mini")

        # Use provided model or fallback to default
        model_to_use = model or default_model

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        client = AsyncOpenAI(api_key=inf_api_key, base_url=base_url)

        # Truncate content if it exceeds max context to avoid hitting limits
        truncated_content = gitingest_content
        if len(gitingest_content) > max_context_chars:
            truncated_content = gitingest_content[:max_context_chars] + "\n\n... [Content truncated due to length] ..."

        # Create the prompt for Dockerfile generation
        additional_instructions_section = ""
        if additional_instructions and additional_instructions.strip():
            additional_instructions_section = f"\n\nADDITIONAL INSTRUCTIONS:\n{additional_instructions.strip()}"

        # Prepare Dockerfile information section
        dockerfile_info_section = ""
        if git_dockerfile:
            dockerfile_info_section = f"\nTHE PROJECT ALREADY HAS DOCKERFILE INFORMATION:\n{git_dockerfile}\n"
        else:
            dockerfile_info_section = "\nTHE PROJECT DOES NOT CONTAIN A DOCKERFILE CURRENTLY.\n"

        prompt = f"""Based on the following repository analysis, generate a comprehensive and production-ready Dockerfile that will successfully build without errors.

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
        # logger.info("create container prompt:%s", prompt)
        # Make API call to generate Dockerfile with streaming
        websocket_active = await emit_ws_message(websocket, "status", "🐳 Generating Dockerfile...")
        if websocket_active:
            print("🐳 Generating Dockerfile... (streaming response)\n")

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

        print("Debug - About to make API call")
        print(f"Debug - Model: " + model)
        print(f"Debug - Messages count: {len(messages)}")
        print(f"Debug - Temperature: 0.3")
        print(f"Debug - Max tokens: 2000")
        print(f"Debug - Stream: {stream}")

        try:
            response = await client.chat.completions.create(
                model=model_to_use,  # Use the selected model for generation
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000,  # Sufficient for Dockerfile generation
                stream=stream,  # Use the stream parameter
                extra_headers={'apikey': api_key} if api_key else None,
            )

            print("Debug - API call initiated successfully")
        except Exception as e:
            print(f"Debug - API call failed with error: {str(e)}")
            raise e

        # Collect response based on streaming setting
        dockerfile_response = ""
        if stream:
            # Handle streaming response
            if websocket_active:
                websocket_active = await emit_ws_message(websocket, "stream_start", "Starting generation...")
            print("📝 Response:")
            print("-" * 50)

            async for chunk in response:
                if len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        dockerfile_response += content
                        # Only emit chunks if WebSocket is still active
                        if websocket_active:
                            websocket_active = await emit_ws_message(websocket, "chunk", content)

            print("\n" + "-" * 50)
            print("✅ Generation complete!\n")
            if websocket_active:
                await emit_ws_message(websocket, "status", "✅ Generation complete!")
        else:
            # Handle non-streaming response
            dockerfile_response = response.choices[0].message.content
            print("📝 Response:")
            print("-" * 50)
            print(dockerfile_response)
            print("-" * 50)
            print("✅ Generation complete!\n")

            # Send the entire response at once if WebSocket is active
            if websocket_active:
                await emit_ws_message(websocket, "stream_start", "Starting generation...")
                await emit_ws_message(websocket, "chunk", dockerfile_response)
                await emit_ws_message(websocket, "status", "✅ Generation complete!")

        # Try to parse as JSON, fallback to plain text if needed
        try:
            # First try direct JSON parsing
            dockerfile_data = json.loads(dockerfile_response)
        except json.JSONDecodeError:
            # Try to extract JSON from code blocks
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
