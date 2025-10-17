"""
Docker image building tool for Gitcontainer application.

This module provides functionality for building Docker images from generated Dockerfiles.
"""

import asyncio
import os
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from autobuild.utils import get_websocket_manager

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


async def build_docker_image(
        dockerfile_content: str,
        project_name: str,
        local_path: str,
        ws_manager: Optional[Any] = None,
        verification_code: Optional[str] = None,
        verification_code_name: Optional[str] = None
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
                
            # Write verification code if provided
            if verification_code and verification_code_name:
                verification_path = temp_path / verification_code_name
                with open(verification_path, "w", encoding="utf-8") as f:
                    f.write(verification_code)
                # Make the verification script executable if it's a shell script
                if verification_code_name.endswith('.sh'):
                    os.chmod(verification_path, 0o755)
                if ws_manager:
                    await ws_manager.send_build_log(f"📝 验证代码已写入: {verification_code_name}\n")

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
            # 生成年月日时分格式的时间戳（如202510161606）
            timestamp = time.strftime("%Y%m%d%H%M")
            image_tag = f"{project_name.lower().replace('/', '-')}:{timestamp}"

            # Build the Docker image
            build_args = ["docker", "build","--no-cache", "-t", image_tag, "."]

            # Send status message about the build
            if ws_manager:
                await ws_manager.send_status(f"🔨 正在构建 Docker 镜像...")
                await ws_manager.send_build_log(f"🚀 开始构建 Docker 镜像: {image_tag}\n")
                await ws_manager.send_build_log(f"📂 构建目录: {temp_dir}\n")
                await ws_manager.send_build_log(f"🏗️ 构建命令: {' '.join(build_args)}\n")
                await ws_manager.send_build_log("=" * 50 + "\n")

            build_process = await asyncio.create_subprocess_exec(
                *build_args,
                cwd=temp_dir,
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
                if ws_manager:
                    await ws_manager.send_build_log("=" * 50 + "\n")
                    await ws_manager.send_build_log(f"✅ 镜像构建成功: {image_tag}\n")
                
                # 调用push_docker_image函数推送镜像到远程仓库
                if ws_manager:
                    await ws_manager.send_build_log("\n" + "=" * 50 + "\n")
                    await ws_manager.send_build_log("🔄 准备推送镜像到远程仓库...\n")
                
                push_result = await push_docker_image(image_tag, ws_manager)
                
                # 如果推送成功，使用远程镜像地址作为返回的image_tag
                if push_result["success"]:
                    remote_image_tag = push_result["remote_image_url"]
                    return {
                        "success": True,
                        "image_tag": remote_image_tag,  # 返回远程镜像地址
                        "local_image_tag": image_tag,    # 保留本地镜像标签
                        "message": f"Successfully built and pushed Docker image: {remote_image_tag}",
                        "build_log": build_log,
                        "push_log": push_result["push_log"]
                    }
                else:
                    # 推送失败但构建成功，返回构建结果和推送失败信息
                    if ws_manager:
                        await ws_manager.send_status(f"⚠️ 镜像构建成功，但推送失败")
                    return {
                        "success": True,
                        "image_tag": image_tag,  # 推送失败时返回本地镜像标签
                        "push_error": push_result["error"],
                        "message": f"Successfully built Docker image, but failed to push: {push_result['error']}",
                        "build_log": build_log,
                        "push_log": push_result.get("push_log", "")
                    }
            else:
                error_output = build_log if build_log else "Docker build failed"
                # Include error lines if we have them
                if error_lines:
                    error_output = "[关键错误信息]:\n" + "".join(
                        error_lines[-100:]) + "\n\n[完整日志摘要(最后100行)]:\n" + "\n".join(
                        build_log.split('\n')[-100:]) if build_log else error_output
                if ws_manager:
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
        logger.exception("Failed to build Docker image")

        return {
            "success": False,
            "error": "Failed to build Docker image due to an internal error",
            "image_tag": None
        }


async def push_docker_image(
        local_image_tag: str,
        ws_manager: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Push a local Docker image to a remote registry after tagging it.
    
    Args:
        local_image_tag (str): The local Docker image tag to push (e.g., sais-fuxi-evalalign:1760594921)
        ws_manager (Any, optional): WebSocket connection for streaming logs
    
    Returns:
        Dict[str, Any]: Dictionary containing the push result and remote image URL
    """
    try:
        import subprocess
        import time
        from pathlib import Path
        
        # 从环境变量读取Docker仓库配置
        registry_url = os.getenv('DOCKER_REGISTRY_URL')
        registry_username = os.getenv('DOCKER_REGISTRY_USERNAME')
        registry_password = os.getenv('DOCKER_REGISTRY_PASSWORD')
        registry_namespace = os.getenv('DOCKER_REGISTRY_NAMESPACE')
        
        # 验证必要的配置
        if not registry_url:
            raise ValueError("DOCKER_REGISTRY_URL is not set in environment variables")

        
        # 生成远程镜像标签
        remote_image_tag = f"{registry_url}/{registry_namespace}/{local_image_tag}"
        
        # 发送开始推送消息
        if ws_manager:
            await ws_manager.send_status(f"📤 正在推送 Docker 镜像到远程仓库...")
            await ws_manager.send_build_log(f"🚀 准备推送镜像到远程仓库: {registry_url}\n")
            await ws_manager.send_build_log(f"🏷️ 本地镜像: {local_image_tag}\n")
            await ws_manager.send_build_log(f"🔗 远程标签: {remote_image_tag}\n")
            await ws_manager.send_build_log("=" * 50 + "\n")
        
        # Step 1: 为本地镜像添加远程标签
        tag_args = ["docker", "tag", local_image_tag, remote_image_tag]
        if ws_manager:
            await ws_manager.send_build_log(f"🏷️ 执行标签操作: {' '.join(tag_args)}\n")
        
        tag_process = await asyncio.create_subprocess_exec(
            *tag_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        tag_output = await tag_process.communicate()
        tag_output_str = tag_output[0].decode('utf-8') if tag_output[0] else ""
        
        if tag_process.returncode != 0:
            if ws_manager:
                await ws_manager.send_build_log(f"❌ 标签操作失败: {tag_output_str}\n")
            return {
                "success": False,
                "error": f"Failed to tag image: {tag_output_str}",
                "remote_image_url": None
            }
        
        if ws_manager and tag_output_str:
            await ws_manager.send_build_log(tag_output_str + "\n")
        
        # Step 2: 如果提供了用户名和密码，则进行登录
        if registry_username and registry_password:
            if ws_manager:
                await ws_manager.send_build_log(f"🔐 正在登录到仓库 {registry_url}...\n")
            
            # 使用echo和管道进行登录，避免密码在命令行中可见
            login_cmd = f"echo {registry_password} | docker login {registry_url} -u {registry_username} --password-stdin"
            
            login_process = await asyncio.create_subprocess_shell(
                login_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            login_output = await login_process.communicate()
            login_output_str = login_output[0].decode('utf-8') if login_output[0] else ""
            
            if login_process.returncode != 0:
                if ws_manager:
                    await ws_manager.send_build_log(f"❌ 登录失败: {login_output_str}\n")
                return {
                    "success": False,
                    "error": f"Failed to login to registry: {login_output_str}",
                    "remote_image_url": None
                }
            
            if ws_manager:
                await ws_manager.send_build_log(f"✅ 登录成功\n")
        else:
            if ws_manager:
                await ws_manager.send_build_log("ℹ️ 未提供登录凭据，跳过登录步骤\n")
        
        # Step 3: 推送镜像到远程仓库
        push_args = ["docker", "push", remote_image_tag]
        if ws_manager:
            await ws_manager.send_build_log(f"📤 执行推送操作: {' '.join(push_args)}\n")
        
        push_process = await asyncio.create_subprocess_exec(
            *push_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        # 流式输出推送日志
        push_log = ""
        while True:
            line = await push_process.stdout.readline()
            if not line:
                break
            decoded_line = line.decode('utf-8')
            push_log += decoded_line
            
            # 发送每行日志到WebSocket
            if ws_manager:
                await ws_manager.send_build_log(decoded_line)
        
        await push_process.wait()
        
        if push_process.returncode == 0:
            if ws_manager:
                await ws_manager.send_build_log("=" * 50 + "\n")
                await ws_manager.send_build_log(f"✅ 镜像推送成功: {remote_image_tag}\n")
                await ws_manager.send_status(f"✅ 镜像已成功推送到远程仓库")
            
            return {
                "success": True,
                "remote_image_url": remote_image_tag,
                "message": f"Successfully pushed Docker image to {remote_image_tag}",
                "push_log": push_log
            }
        else:
            error_output = push_log if push_log else "Docker push failed"
            if ws_manager:
                await ws_manager.send_build_log("=" * 50 + "\n")
                await ws_manager.send_build_log(f"❌ 镜像推送失败\n")
                await ws_manager.send_status(f"❌ 镜像推送失败")
            
            return {
                "success": False,
                "error": error_output,
                "remote_image_url": None,
                "push_log": error_output
            }
    
    except Exception as e:
        logger.exception("Failed to push Docker image")
        error_msg = str(e)
        
        if ws_manager:
            await ws_manager.send_build_log(f"❌ 推送过程发生错误: {error_msg}\n")
            await ws_manager.send_status(f"❌ 镜像推送失败")
        
        return {
            "success": False,
            "error": f"Failed to push Docker image: {error_msg}",
            "remote_image_url": None
        }