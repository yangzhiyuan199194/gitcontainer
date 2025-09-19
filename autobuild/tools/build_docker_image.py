"""
Docker image building tool for Gitcontainer application.

This module provides functionality for building Docker images from generated Dockerfiles.
"""

import asyncio
import os
from typing import Dict, Any, Optional

from autobuild.utils import get_websocket_manager


async def build_docker_image(
        dockerfile_content: str,
        project_name: str,
        local_path: str,
        websocket: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Build a Docker image from the provided Dockerfile content.

    Args:
        dockerfile_content (str): The content of the Dockerfile
        project_name (str): Name of the project
        local_path (str): Local path to the repository
        websocket (Any, optional): WebSocket connection for streaming build logs

    Returns:
        Dict[str, Any]: Dictionary containing the build result and image information
    """
    # Initialize WebSocket manager
    ws_manager = get_websocket_manager(websocket)
    
    try:
        import tempfile
        import shutil
        import subprocess
        from pathlib import Path

        # Create a temporary directory for building
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write the Dockerfile
            dockerfile_path = temp_path / "Dockerfile"
            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(dockerfile_content)

            # Copy project files to temp directory (excluding .git)
            if os.path.exists(local_path):
                for item in os.listdir(local_path):
                    if item != ".git":  # Skip .git directory
                        src = os.path.join(local_path, item)
                        dst = os.path.join(temp_dir, item)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)

            # Generate image tag (use project name and timestamp)
            import time
            timestamp = int(time.time())
            image_tag = f"{project_name.lower().replace('/', '-')}:{timestamp}"

            # Build the Docker image
            build_args = ["docker", "build","--no-cache", "-t", image_tag, "."]

            # Send status message about the build
            await ws_manager.send_status(f"🔨 正在构建 Docker 镜像...")
            await ws_manager.send_build_log(f"🚀 开始构建 Docker 镜像: {image_tag}\n")
            await ws_manager.send_build_log(f"📂 构建目录: {temp_dir}\n")
            await ws_manager.send_build_log(f"🏗️ 构建命令: {' '.join(build_args)}\n")
            await ws_manager.send_build_log("=" * 50 + "\n")

            build_process = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=temp_dir,
                env=os.environ,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT  # Combine stderr with stdout
            )

            print(f"Debug - Docker build process started with PID: {build_process.pid}")

            # Stream the build output in real-time with limit
            build_log = ""
            line_count = 0
            max_lines = 10000  # Increase limit to 10000 lines
            error_lines = []  # Store error-related lines
            collecting_error_context = 0  # Number of lines to collect after an error line

            while True:
                line = await build_process.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8')
                build_log += decoded_line
                line_count += 1

                # Check if this line contains error indicators
                if any(keyword in decoded_line.lower() for keyword in
                       ['error', 'failed', 'exception', 'invalid', 'cannot', 'could not']):
                    error_lines.append(decoded_line)
                    collecting_error_context = 20  # Collect 20 more lines after error
                elif collecting_error_context > 0:
                    error_lines.append(decoded_line)
                    collecting_error_context -= 1

                # Limit the lines we collect to prevent memory issues
                if line_count > max_lines:
                    if line_count == max_lines + 1:  # Only send this message once
                        await ws_manager.send_build_log(f"\n... [Output truncated to {max_lines} lines] ...\n")

                        # Send important error lines if we have any
                        if error_lines:
                            await ws_manager.send_build_log(f"\n[关键错误信息摘要]:\n")
                            for error_line in error_lines[-50:]:  # Send last 50 error lines
                                await ws_manager.send_build_log(error_line)
                    continue

                # Send each line to the WebSocket
                # Check if WebSocket is still active before sending
                if not await ws_manager.send_build_log(decoded_line):
                    print("WebSocket closed, stopping log streaming")
                    break

            print(f"Debug - Docker build process finished. Read {line_count} lines of output")

            await build_process.wait()
            print(f"Debug - Docker build process return code: {build_process.returncode}")

            if build_process.returncode == 0:
                await ws_manager.send_build_log("=" * 50 + "\n")
                await ws_manager.send_build_log(f"✅ 镜像构建成功: {image_tag}\n")
                return {
                    "success": True,
                    "image_tag": image_tag,
                    "message": f"Successfully built Docker image: {image_tag}",
                    "build_log": build_log
                }
            else:
                error_output = build_log if build_log else "Docker build failed"
                # Include error lines if we have them
                if error_lines:
                    error_output = "[关键错误信息]:\n" + "".join(
                        error_lines[-100:]) + "\n\n[完整日志摘要(最后100行)]:\n" + "\n".join(
                        build_log.split('\n')[-100:]) if build_log else error_output

                await ws_manager.send_build_log("=" * 50 + "\n")
                await ws_manager.send_build_log(f"❌ 镜像构建失败\n")
                return {
                    "success": False,
                    "error": error_output,
                    "image_tag": image_tag,
                    "build_log": error_output
                }

    except Exception as e:
        # 记录详细的异常信息用于调试，但不暴露给用户
        import logging
        logging.exception("Failed to build Docker image")

        return {
            "success": False,
            "error": "Failed to build Docker image due to an internal error",
            "image_tag": None
        }