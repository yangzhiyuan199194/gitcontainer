"""
Dockerfile generation prompts for Gitcontainer application.
"""

def create_dockerfile_prompt(
    gitingest_summary: str,
    gitingest_tree: str,
    truncated_content: str,
    git_dockerfile: str = None,
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
7. If the user does not explicitly specify the Dockerfile and the project relies on the python environment, conda should be used first to build the user-dependent environment，avoid using the default Anaconda channel that requires a ToS and instead use the conda suggests explicitly specifying -c conda-forge


8. Follows Docker best practices:
   - Use multi-stage builds when beneficial to reduce image size
   - Combine related RUN commands to minimize layers
   - Properly handle package manager caches (e.g., apt-get clean, rm -rf /var/lib/apt/lists/*)
   - Use COPY instead of ADD unless specifically needed
   - Make sure the file you want to copy exists in the corresponding directory
9. Handles environment variables and configuration appropriately
10. Ensures all commands are compatible with the chosen base image
11. Avoids common build errors:
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