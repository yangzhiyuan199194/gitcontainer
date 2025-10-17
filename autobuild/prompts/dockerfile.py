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

    return '''Based on the following repository analysis, generate a comprehensive and production-ready Dockerfile that will successfully build without errors.

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
7. If the user does not explicitly specify the Dockerfile and the project relies on the python environment, conda should be used first to build the user-dependent environment，avoid using the default Anaconda channel that requires a ToS and instead use the conda suggests explicitly specifying --override-channels -c conda-forge
8. Use the base image pre-installed devel version with CUDA Toolkit first, it is recommended to use cuda 12.4 or above. To ensure compatibility, please run pip install... Previously, explicitly install numpy 1.x version by running pip install 'numpy<2'
9. If you need to install DeepSpeed, use the environment variable: DS_BUILD_CCL=0 to explicitly disable oneCCL support
10. Follows Docker best practices:
   - Use multi-stage builds when beneficial to reduce image size
   - Combine related RUN commands to minimize layers
   - Properly handle package manager caches (e.g., apt-get clean, rm -rf /var/lib/apt/lists/*)
   - Use COPY instead of ADD unless specifically needed
   - Make sure the file you want to copy exists in the corresponding directory
11. Handles environment variables and configuration appropriately
12. Ensures all commands are compatible with the chosen base image
13. Avoids common build errors:
    - Always use specific versions for base images (avoid 'latest')
    - Use a definite existing base image and do not fabricate non-existent images at will
    - Properly escape special characters in commands
    - Ensure all required files are copied or created before being used
    - Handle platform-specific dependencies correctly
    - Install build dependencies before runtime dependencies where applicable
    - Properly configure the entrypoint and command for the application type

Additionally, please generate a verification code snippet that demonstrates how to:

Also, please specify the resource requirements for running this application in the container:
1. Minimal resource specifications required for basic functionality
2. Recommended resource specifications for optimal performance

Each resource specification should include:
- CPU cores required
- Memory required (in GB)
- GPU count required (0 if not needed)


1. Test the core functionality of the application within the Docker container
2. Verify that the application's key features are working correctly
3. Perform relevant API calls, CLI commands, or functional tests specific to this project
4. Include proper error handling and meaningful output verification

Additionally, please generate both verification code and execution commands:

1. VERIFICATION CODE:
- The verification code should be generated as a separate file
- It should be highly specific to the cloned project's actual code and functionality
- Focused on validating the application works as expected
- Tailored to the project's technology stack and architecture
- Include comments explaining each test step and expected outcomes
- The code should be designed to be copied to a well-known location in the image (e.g., /app/verification.py for Python)
- Provide a descriptive code name for the verification file (e.g., verification.py, verify.sh)

2. EXECUTION COMMAND:
- Provide the exact command to execute the verification code in a Kubernetes pod
- Ensure the command correctly references the verification code path
- The command should be formatted as a list suitable for Kubernetes execution
- The command should return a structured JSON output with the following fields:
  * success: Boolean indicating if the test passed
  * output: String containing the test output
  * duration: Number indicating how long the test took (in seconds)
  * details: Object containing any additional relevant test details

Important: Ensure the verification code is properly installed in the image and the execution command correctly references it.

If you detect multiple services or a complex architecture, provide a main Dockerfile for the primary service and suggest a docker-compose.yml structure.

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any explanations, or code blocks. The response must be parseable JSON.

Required JSON format:
{{
  "dockerfile": "FROM python:3.9-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nEXPOSE 8000\nCMD [\"python\", \"app.py\"]",
  "base_image_reasoning": "Explanation of why you chose the base image, including why it will successfully build",
  "technology_stack": "Detected technologies and frameworks",
  "port_recommendations": ["8000", "80"],
  "additional_notes": "Any important setup or deployment notes, including potential build issues and how to avoid them",
  "docker_compose_suggestion": "Optional docker-compose.yml content if multiple services detected",
  "verification_code": {{
    "language": "python",
    "code": "#!/usr/bin/env python3\n\"\"\"\nVerification script for the application\nThis script will be built into the Docker image and executed in Kubernetes\n\"\"\"\nimport json\nimport time\nimport sys\n\n# Record start time\nstart_time = time.time()\n\nresult = {{{{\n    'success': False,\n    'output': '',\n    'duration': 0,\n    'details': {{}}\n}}}}\n\ntry:\n    # Add application-specific verification code here\n    # Example for a Python application:\n    result['output'] = 'Testing application core functionality...\n'\n    \n    # Check if application files exist\n    import os\n    if os.path.exists('main.py'):\n        result['output'] += '✅ Application entry point found\n'\n        result['details']['main_file_exists'] = True\n    else:\n        result['output'] += '❌ Application entry point not found\n'\n        result['details']['main_file_exists'] = False\n    \n    # Add more application-specific tests here\n    \n    # For demonstration purposes - mark as successful\n    result['success'] = True\n    result['output'] += '✅ All tests completed successfully'\n    \nexcept Exception as e:\n    result['output'] += '❌ Error during verification: ' + str(e) + '\n'\n    result['details']['error'] = str(e)\n    result['success'] = False\nfinally:\n    # Calculate duration\n    result['duration'] = time.time() - start_time\n    \n    # Print JSON result to stdout\n    print(json.dumps(result, indent=2))\n    \n    # Exit with appropriate status code\n    sys.exit(0 if result['success'] else 1)\n",
    "description": "Python verification script that returns structured JSON output",
    "dependencies": ["python3"],
    "install_path": "/app/verification.py",
    "code_name": "verification.py"
  }},
  "execution_command": [
    "/bin/sh", 
    "-c", 
    "python3 /app/verification.py"
  ],
  "resource_requirements": {{
    "minimal": {{
      "cpu_cores": 1,
      "memory_gb": 2,
      "gpu_count": 0
    }},
    "recommended": {{
      "cpu_cores": 4,
      "memory_gb": 8,
      "gpu_count": 1
    }}
  }}
}}'''.format(gitingest_summary=gitingest_summary, gitingest_tree=gitingest_tree, truncated_content=truncated_content, additional_instructions_section=additional_instructions_section, dockerfile_info_section=dockerfile_info_section)

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

BUILD LOG (last 500 characters):
{build_log[-500:] if build_log else "No build log available"}

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
    "Detailed suggestion 1",
    "Detailed suggestion 2"
  ],
  "revised_dockerfile": "FROM python:3.9-slim\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nEXPOSE 8000\nCMD [\"python\", \"app.py\"]"
}}"""