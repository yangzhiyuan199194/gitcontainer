"""
Git operations tool for Gitcontainer application.

This module provides functionality for cloning Git repositories.
"""

import asyncio
import os
import shutil
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from autobuild.utils import get_websocket_manager


async def clone_repo_tool(github_url: str, target_dir: str = "repos", ws_manager: Optional[Any] = None) -> Dict[str, Any]:
    """
    Clone a GitHub repository locally for future usage.
    
    Args:
        github_url (str): The GitHub repository URL to clone
        target_dir (str): Directory where to clone the repository (default: "repos")
        websocket (Optional[Any]): WebSocket connection for streaming output
        
    Returns:
        Dict[str, Any]: Dictionary containing clone results and local path
    """

    try:
        # Parse the GitHub URL to extract repo name
        parsed_url = urlparse(github_url)
        if not parsed_url.netloc == "github.com":
            return {
                "success": False,
                "error": "Only GitHub URLs are supported",
                "url": github_url
            }
        
        # Extract repo name from URL path
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            return {
                "success": False,
                "error": "Invalid GitHub URL format",
                "url": github_url
            }
        
        owner, repo_name = path_parts[0], path_parts[1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Full local path for the cloned repository
        local_path = os.path.join(target_dir, f"{owner}_{repo_name}")
        
        # Remove existing directory if it exists with enhanced retry mechanism
        if os.path.exists(local_path):
            if ws_manager:
                await ws_manager.send_status(f"🗑️  Removing existing directory: {local_path}")
            for attempt in range(5):  # Try up to 5 times
                try:
                    # Change permissions to ensure we can delete
                    # if os.path.isdir(local_path):
                    #     for root, dirs, files in os.walk(local_path):
                    #         for d in dirs:
                    #             os.chmod(os.path.join(root, d), 0o777)
                    #         for f in files:
                    #             os.chmod(os.path.join(root, f), 0o777)
                    #     os.chmod(local_path, 0o777)
                    
                    shutil.rmtree(local_path)
                    
                    # Wait and verify deletion
                    time.sleep(0.5)  # Increase wait time
                    if not os.path.exists(local_path):
                        if ws_manager:
                            await ws_manager.send_status(f"✅ Successfully removed existing directory")
                        # Add extra wait to ensure file system operations are completed
                        time.sleep(0.5)
                        break
                    else:
                        if ws_manager:
                            await ws_manager.send_status(f"⚠️  Directory still exists after deletion attempt {attempt + 1}")
                        
                    # If still exists, try again
                    if attempt == 4:  # Last attempt
                        return {
                            "success": False,
                            "error": f"Failed to remove existing directory after 5 attempts. Directory might be locked by another process.",
                            "url": github_url
                        }
                except PermissionError as e:
                    if ws_manager:
                        await ws_manager.send_status(f"⚠️  Permission denied on attempt {attempt + 1}: {str(e)}")
                    if attempt == 4:  # Last attempt
                        return {
                            "success": False,
                            "error": f"Permission denied when removing directory: {str(e)}. Directory might be locked by another process.",
                            "url": github_url
                        }
                    time.sleep(0.5)  # Wait before retrying
                except Exception as e:
                    if ws_manager:
                        await ws_manager.send_status(f"⚠️  Error on attempt {attempt + 1}: {str(e)}")
                    if attempt == 4:  # Last attempt
                        return {
                            "success": False,
                            "error": f"Failed to remove existing directory after 5 attempts: {str(e)}",
                            "url": github_url
                        }
                    time.sleep(0.5)  # Wait before retrying
            # If we're here, it means we successfully removed the directory
            # Wait a bit more to ensure file system operations are completed
            time.sleep(0.5)
        
        # Clone the repository using git command
        # First try normal clone
        clone_commands = [
            f"git clone --recursive {github_url} {local_path}",  # Normal clone
            f"git clone --recursive -c http.sslVerify=false {github_url} {local_path}"  # Clone without SSL verification
        ]
        
        success = False
        last_error = ""
        
        for attempt, clone_command in enumerate(clone_commands):
            if ws_manager:
                if attempt == 0:
                    await ws_manager.send_status(f"📥 Cloning repository to: {local_path}")
                else:
                    await ws_manager.send_status(f"🔄 Retrying clone without SSL verification...")
            
            # Run the git clone command with real-time output
            process = await asyncio.create_subprocess_shell(
                clone_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Stream output in real-time
            stdout_lines = []
            stderr_lines = []
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode('utf-8').strip()
                if line_str:
                    stdout_lines.append(line_str)
                    await ws_manager.send_chunk(f"[CLONE] {line_str}\n")
            
            # Also read stderr
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                line_str = line.decode('utf-8').strip()
                stderr_lines.append(line_str)
                if line_str:
                    await ws_manager.send_chunk(f"[CLONE] {line_str}\n")
            
            await process.wait()
            
            if process.returncode == 0:
                success = True
                break
            else:
                last_error = "\n".join(stderr_lines) if stderr_lines else "Clone failed"
                if ws_manager:
                    await ws_manager.send_status(f"⚠️  Clone attempt {attempt + 1} failed. {last_error}")
        
        if not success:
            # Try HTTP fallback if HTTPS is failing
            if "https://" in github_url:
                http_url = github_url.replace("https://", "http://")
                if ws_manager:
                    await ws_manager.send_status(f"🔄 Retrying with HTTP instead of HTTPS...")
                
                clone_command = f"git clone --recursive {http_url} {local_path}"
                process = await asyncio.create_subprocess_shell(
                    clone_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout_lines = []
                stderr_lines = []
                
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        stdout_lines.append(line_str)
                        await ws_manager.send_chunk(f"[CLONE] {line_str}\n")
                
                while True:
                    line = await process.stderr.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8').strip()
                    stderr_lines.append(line_str)
                    if line_str:
                        await ws_manager.send_chunk(f"[CLONE] {line_str}\n")
                
                await process.wait()
                
                if process.returncode == 0:
                    success = True
                else:
                    last_error = "\n".join(stderr_lines) if stderr_lines else "Clone failed"
            
            if not success:
                return {
                    "success": False,
                    "error": last_error,
                    "url": github_url,
                    "troubleshooting": "SSL连接错误。可能是网络问题、防火墙限制或证书验证问题。请检查网络连接或尝试使用VPN。"
                }
        
        stdout_lines = []  # Reinitialize for success case
        
        # Stream output in real-time
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode('utf-8').strip()
            if line_str:
                await ws_manager.send_chunk(f"[CLONE] {line_str}\n")
        
        # Also read stderr
        stderr_lines = []
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            line_str = line.decode('utf-8').strip()
            stderr_lines.append(line_str)
            if line_str:
                await ws_manager.send_chunk(f"[CLONE] {line_str}\n")
        
        await process.wait()
        
        # Get repository info
        repo_size = get_directory_size(local_path)
        file_count = count_files(local_path)
        
        await ws_manager.send_chunk(f"✅ Successfully cloned {owner}/{repo_name}\n")
        
        return {
            "success": True,
            "local_path": local_path,
            "repo_name": f"{owner}/{repo_name}",
            "repo_size_mb": round(repo_size / (1024 * 1024), 2),
            "file_count": file_count,
            "url": github_url,
            "message": f"Successfully cloned {owner}/{repo_name} to {local_path}"
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": github_url
        }


def get_directory_size(path: str) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception:
        pass
    return total_size


def count_files(path: str) -> int:
    """Count total number of files in directory."""
    file_count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            file_count += len(filenames)
    except Exception:
        pass
    return file_count


def run_clone_repo(github_url: str, target_dir: str = "repos") -> Dict[str, Any]:
    """
    Synchronous wrapper for the clone repo tool.
    
    Args:
        github_url (str): The GitHub repository URL to clone
        target_dir (str): Directory where to clone the repository
        
    Returns:
        Dict[str, Any]: Dictionary containing clone results
    """
    return asyncio.run(clone_repo_tool(github_url, target_dir))


# Tool definition for OpenAI Agents SDK
git_operations_function = {
    "type": "function",
    "function": {
        "name": "clone_github_repo",
        "description": "Clone a GitHub repository locally for analysis and future usage",
        "parameters": {
            "type": "object",
            "properties": {
                "github_url": {
                    "type": "string",
                    "description": "The GitHub repository URL to clone (e.g., https://github.com/user/repo)"
                },
                "target_dir": {
                    "type": "string",
                    "description": "Directory where to clone the repository (default: 'repos')",
                    "default": "repos"
                }
            },
            "required": ["github_url"]
        }
    }
}