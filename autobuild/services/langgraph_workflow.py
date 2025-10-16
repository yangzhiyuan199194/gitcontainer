"""
LangGraph Workflow Service for Gitcontainer application.

This module provides an enhanced workflow implementation using LangGraph with automatic
websocket message forwarding for real-time frontend updates.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable
from typing_extensions import Annotated, TypedDict

from fastapi import WebSocket
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from autobuild.core.config import Settings
from autobuild.prompts.dockerfile import create_reflection_prompt
from autobuild.services.llm_client import LLMClient
from autobuild.tools import (
    clone_repo_tool,
    gitingest_tool,
    create_container_tool,
    build_docker_image,
    wiki_generator_tool
)
from autobuild.utils.websocket_manager import WebSocketManager, MessageType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create settings instance
settings = Settings()


class LangGraphWorkflowState(TypedDict):
    """Enhanced workflow state with automatic message handling"""
    repo_url: Annotated[str, "append"]
    additional_instructions: str
    model: Optional[str]
    generate_wiki: bool  # Add generate_wiki flag
    clone_result: Dict[str, Any]
    analysis_result: Dict[str, Any]
    dockerfile_result: Dict[str, Any]
    build_result: Dict[str, Any]
    test_result: Dict[str, Any]  # Add test_result field
    reflection_result: Dict[str, Any]
    iteration: int
    max_iterations: int
    final_result: Dict[str, Any]
    ws_log_file_path: Optional[str]
    websocket: Optional[WebSocket]
    ws_manager: Optional[WebSocketManager]
    messages: Annotated[List[Dict[str, Any]], add_messages]
    current_node: Optional[str]
    node_status: Dict[str, str]
    # Wiki generation fields
    wiki_result: Dict[str, Any]


def auto_message_handler(func: Callable) -> Callable:
    """
    Decorator that automatically sends node status messages to frontend.
    
    This decorator wraps workflow nodes to automatically send:
    1. Node start message
    2. Node completion message with status
    3. Any errors that occur during execution
    """

    async def wrapper(state: LangGraphWorkflowState) -> Dict[str, Any]:
        node_name = func.__name__.replace("_", " ").title()
        logger.info(f"Starting node: {node_name}")

        # Initialize WebSocket manager if not already present
        if "ws_manager" not in state or state["ws_manager"] is None:
            ws_manager = WebSocketManager(
                websocket=state.get("websocket"),
                log_file=state.get("ws_log_file_path")
            )
            state["ws_manager"] = ws_manager
        else:
            ws_manager = state["ws_manager"]

        # Update current node in state
        updated_state = {"current_node": node_name, "node_status": state.get("node_status", {})}

        # Get node metadata if available
        node_metadata = None
        if hasattr(state, 'workflow_metadata') and 'nodes' in state.workflow_metadata:
            node_metadata = state.workflow_metadata['nodes'].get(node_name.lower(), {})

        # 节点ID映射：从函数名映射到工作流元数据中的节点ID
        node_id_mapping = {
            'clone_repository': 'clone',
            'analyze_repository': 'analyze',
            'generate_dockerfile': 'generate',
            'build_docker': 'build',
            'reflect_on_failure': 'reflect',
            'improve_dockerfile': 'improve',
            'generate_wiki': 'wiki'
        }

        # 获取正确的节点ID
        node_id = node_id_mapping.get(func.__name__, func.__name__)

        # Send node start message using the correct node ID
        await ws_manager.send_node_start(node_id, node_metadata)
        await ws_manager.send_status(f"🔄 Processing {node_name}...")

        try:
            # Execute the actual node function
            result = await func(state)

            # 更新节点状态为completed
            updated_state["node_status"][node_id] = "completed"

            # 发送节点完成消息（使用正确的node_id）
            await ws_manager.send_node_complete(node_id, {"success": True})

            # Merge results
            result.update(updated_state)
            return result

        except Exception as e:
            error_msg = f"Error in {node_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # 更新节点状态为failed
            updated_state["node_status"][node_id] = "failed"

            # 发送错误消息
            await ws_manager.send_error(error_msg)
            # 发送节点完成消息（使用正确的node_id）
            await ws_manager.send_node_complete(node_id, {"success": False, "error": str(e)})

            # Return state with error information
            error_info = {"error": error_msg, "error_node": node_name}
            updated_state.update(error_info)
            return updated_state

    wrapper.__name__ = func.__name__
    return wrapper


@auto_message_handler
async def clone_repository(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Clone repository node"""
    ws_manager = state.get("ws_manager")

    await ws_manager.send_build_log("🚀 开始克隆仓库...\n")
    clone_result = await clone_repo_tool(
        github_url=state["repo_url"],
        ws_manager=ws_manager
    )

    if clone_result.get("success"):
        await ws_manager.send_build_log(f"✅ 仓库克隆成功: {clone_result.get('repo_name', 'unknown')}\n")
    else:
        await ws_manager.send_build_log(f"❌ 仓库克隆失败: {clone_result.get('error', 'Unknown error')}\n")

    return {"clone_result": clone_result}


@auto_message_handler
async def analyze_repository(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Analyze repository node"""
    ws_manager = state.get("ws_manager")
    clone_result = state["clone_result"]

    if not clone_result.get("success"):
        error_msg = "无法分析仓库：克隆失败"
        troubleshooting = clone_result.get("troubleshooting", "请检查网络连接和仓库URL是否正确")
        if ws_manager:
            await ws_manager.send_error(f"{error_msg}: {troubleshooting}")
        raise Exception(f"{error_msg}: {troubleshooting}")

    try:
        await ws_manager.send_build_log("🔍 开始分析仓库结构和内容...\n")
        analysis_result = await gitingest_tool(
            clone_result["local_path"],
            state["model"],
            ws_manager=ws_manager
        )

        # 确保analysis_result包含必要字段
        if "summary" not in analysis_result:
            analysis_result["summary"] = "仓库分析摘要缺失"
        if "tree" not in analysis_result:
            analysis_result["tree"] = "仓库结构缺失"
        if "content" not in analysis_result:
            analysis_result["content"] = "仓库内容缺失"

        return {"analysis_result": analysis_result}
    except KeyError as e:
        error_msg = f"分析仓库时遇到KeyError: {str(e)}"
        if ws_manager:
            await ws_manager.send_error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"分析仓库时出错: {str(e)}"
        if ws_manager:
            await ws_manager.send_error(error_msg)
        raise Exception(error_msg)


@auto_message_handler
async def generate_dockerfile(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Generate Dockerfile node"""
    ws_manager = state.get("ws_manager")

    # Check if we have valid analysis result
    if not state.get("analysis_result") or not state.get("clone_result"):
        raise Exception("Cannot generate Dockerfile: missing repository analysis or clone data")

    # Determine if the selected model supports streaming
    stream_support = settings.get_model_stream_support(state["model"]) if state["model"] else True

    # Check if we have reflection results to use as additional instructions
    additional_instructions = state["additional_instructions"] or ""
    if state.get("reflection_result", {}).get("needed"):
        if "root_cause" in state["reflection_result"]:
            improvement_points = "\n".join(state["reflection_result"].get("suggestions", []))
            additional_instructions += f"\n\nBased on the following analysis results, improve the Dockerfile:\n{improvement_points}"
        elif "error_message" in state["reflection_result"]:
            additional_instructions += f"\nBuild error message: {state['reflection_result']['error_message']}"

    await ws_manager.send_build_log("🐳 开始生成Dockerfile...\n")
    try:
        dockerfile_result = await create_container_tool(
            gitingest_summary=state["analysis_result"].get("summary", ""),
            gitingest_tree=state["analysis_result"].get("tree", ""),
            gitingest_content=state["analysis_result"].get("content", ""),
            project_name=state["clone_result"].get("repo_name", "project"),
            additional_instructions=additional_instructions,
            model=state["model"],
            ws_manager=ws_manager,
            stream=stream_support
        )

        # 确保返回的结果中包含verification_code
        result = {"dockerfile_result": dockerfile_result}

        return result
    except Exception as e:
        # Return a failed result instead of raising exception
        logger.error(f"Dockerfile generation failed: {e}")
        return {"dockerfile_result": {"success": False, "error": str(e)}}


@auto_message_handler
async def build_docker(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Build Docker image node"""
    ws_manager = state.get("ws_manager")

    try:
        # Send build log header
        if ws_manager:
            await ws_manager.send_build_log("🚀 开始构建Docker镜像...\n")

        # 安全获取dockerfile_result和clone_result
        dockerfile_result = state.get("dockerfile_result", {})
        clone_result = state.get("clone_result", {})

        # Check if we have a valid dockerfile_result
        if not dockerfile_result:
            build_result = {
                "success": False,
                "error": "Dockerfile生成结果缺失",
                "build_logs": "无法获取Dockerfile生成结果"
            }
            if ws_manager:
                await ws_manager.send_error("构建失败: Dockerfile生成结果缺失")
            return {"build_result": build_result}

        # Check if Dockerfile generation failed
        if not dockerfile_result.get("success"):
            build_result = {
                "success": False,
                "error": dockerfile_result.get("error", "Dockerfile生成失败"),
                "build_logs": f"Dockerfile生成失败: {dockerfile_result.get('error', '未知错误')}"
            }
            if ws_manager:
                await ws_manager.send_error(f"构建失败: {dockerfile_result.get('error', 'Dockerfile生成失败')}")
            return {"build_result": build_result}

        # Check if we have a Dockerfile
        if not dockerfile_result.get("dockerfile"):
            build_result = {
                "success": False,
                "error": "无法获取Dockerfile内容，构建失败",
                "build_logs": "Dockerfile生成失败: 缺少必要的dockerfile字段"
            }
            if ws_manager:
                await ws_manager.send_error("构建失败: 缺少Dockerfile内容")
            return {"build_result": build_result}

        # Check if we have a valid clone result
        if not clone_result or not clone_result.get("repo_name"):
            build_result = {
                "success": False,
                "error": "仓库克隆信息缺失",
                "build_logs": "无法获取仓库信息"
            }
            if ws_manager:
                await ws_manager.send_error("构建失败: 仓库克隆信息缺失")
            return {"build_result": build_result}

        # Build the Docker image
        try:
            # Check if we have local_path in clone_result
            if clone_result.get("local_path"):
                build_result = await build_docker_image(
                    dockerfile_content=dockerfile_result["dockerfile"],
                    project_name=clone_result.get("repo_name", "project"),
                    local_path=clone_result["local_path"],
                    ws_manager=ws_manager
                )
            else:
                build_result = {
                    "success": False,
                    "error": f"构建镜像时出错: we have not local_path in clone_result",
                    "build_logs": f"构建镜像失败: we have not local_path in clone_result"
                }

            # 确保build_result包含必要字段
            if not isinstance(build_result, dict):
                build_result = {"success": False, "error": "构建结果格式错误"}
            if "success" not in build_result:
                build_result["success"] = False

            return {"build_result": build_result}
        except Exception as e:
            logger.error(f"Docker image build failed: {e}")
            build_result = {
                "success": False,
                "error": f"构建镜像时出错: {str(e)}",
                "build_logs": f"构建镜像失败: {str(e)}"
            }
            if ws_manager:
                await ws_manager.send_error(f"构建失败: {str(e)}")
            return {"build_result": build_result}
    except KeyError as e:
        error_msg = f"构建Docker镜像时遇到KeyError: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ws_manager:
            await ws_manager.send_error(error_msg)
        return {
            "build_result": {
                "success": False,
                "error": error_msg,
                "build_logs": error_msg
            }
        }
    except Exception as e:
        error_msg = f"构建Docker镜像时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ws_manager:
            await ws_manager.send_error(error_msg)
        return {
            "build_result": {
                "success": False,
                "error": error_msg,
                "build_logs": error_msg
            }
        }
@auto_message_handler
async def test_env(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Test ENV node - Run tests in Kubernetes cluster"""
    ws_manager = state.get("ws_manager")
    
    try:
        # Check if we have a successful build result
        build_result = state.get("build_result", {})
        if not build_result.get("success"):
            error_msg = "无法进行测试：镜像构建失败"
            if ws_manager:
                await ws_manager.send_error(error_msg)
            return {"test_result": {"success": False, "error": error_msg}}
        
        # Get the built image from build result
        image = build_result.get("image_tag", "")
        if not image:
            error_msg = "无法进行测试：找不到构建的镜像"
            if ws_manager:
                await ws_manager.send_error(error_msg)
            return {"test_result": {"success": False, "error": error_msg}}
        
        # Import k8s operations
        from autobuild.tools.k8s_operations import run_test_in_k8s
        
        # Determine test command based on repository type (can be enhanced)
        test_command = None
        
        # # Check if we have repository analysis
        # analysis_result = state.get("analysis_result", {})
        # if analysis_result:
        #     # You can enhance this logic to determine appropriate test commands
        #     # based on the repository analysis results
        #     summary = analysis_result.get("summary", "").lower()
        #     if "python" in summary:
        #         # Python project detection
        #         test_command = ["/bin/sh", "-c", "pip install -e . && pytest || echo 'No tests found'"]
        #     elif "node" in summary or "javascript" in summary or "typescript" in summary:
        #         # Node.js project detection
        #         test_command = ["/bin/sh", "-c", "npm install && npm test || echo 'No tests found'"]
        #     elif "golang" in summary:
        #         # Go project detection
        #         test_command = ["/bin/sh", "-c", "go test ./... || echo 'No tests found'"]
        
        # # Default test command if none detected
        # if not test_command:
        #     test_command = ["/bin/sh", "-c", "echo 'Running basic environment test' && echo 'Test completed successfully'"]

        test_command = ["/bin/sh", "-c", "echo 'Running basic environment test' && echo 'Test completed successfully'"]
        
        # Send test start message
        if ws_manager:
            await ws_manager.send_build_log(f"🧪 开始在Kubernetes中测试镜像: {image}\n")
            await ws_manager.send_build_log(f"📋 测试命令: {test_command}\n")
        
        # Run test in Kubernetes
        test_result = await run_test_in_k8s(
            image=image,
            test_command=test_command,
            ws_manager=ws_manager
        )
        
        # Update node status and return result
        return {"test_result": test_result}
        
    except Exception as e:
        error_msg = f"测试环境执行时出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ws_manager:
            await ws_manager.send_error(error_msg)
        return {"test_result": {"success": False, "error": error_msg}}



@auto_message_handler
async def generate_wiki(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Generate Wiki documentation node"""
    ws_manager = state.get("ws_manager")

    if ws_manager:
        await ws_manager.send_build_log("📚 开始生成Wiki文档...\n")

    try:
        # Extract owner and repo from repo_url with safe access
        repo_url = state.get("repo_url", "")
        owner, repo = "unknown", "unknown"
        if "github.com/" in repo_url:
            owner_repo = repo_url.split("github.com/")[-1].rstrip("/")
            if owner_repo.endswith(".git"):
                owner_repo = owner_repo[:-4]
            parts = owner_repo.split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]

        # Generate Wiki with safe field access
        local_repo_path = state.get("clone_result", {}).get("local_path", "")
        if not local_repo_path:
            raise Exception("无法获取本地仓库路径")

        wiki_result = await wiki_generator_tool(
            local_repo_path=local_repo_path,
            owner=owner,
            repo=repo,
            model=state.get("model"),
            ws_manager=ws_manager
        )

        return {"wiki_result": wiki_result}
    except Exception as e:
        error_msg = f"生成Wiki文档时出错: {str(e)}"
        if ws_manager:
            await ws_manager.send_error(error_msg)
        return {
            "wiki_result": {
                "success": False,
                "error": error_msg
            }
        }


@auto_message_handler
async def reflect_on_failure(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Reflect on build failure node"""
    ws_manager = state.get("ws_manager")

    if state["build_result"]["success"]:
        reflection_result = {
            "needed": False,
            "improvements": []
        }
        return {"reflection_result": reflection_result}

    # Build failed, analyze the reason
    build_log = state["build_result"].get("build_log", "")
    error_message = state["build_result"].get("error", "")
    dockerfile_content = state["dockerfile_result"].get("dockerfile", "")

    await ws_manager.send_build_log("💡 使用 AI 分析构建失败原因...\n")

    # Use LLM to analyze build failure
    llm_client = LLMClient()
    model = state.get("model", llm_client.default_model)

    prompt = create_reflection_prompt(
        dockerfile_content=dockerfile_content,
        build_log=build_log,
        error_message=error_message,
        gitingest_summary=state["analysis_result"].get("summary", ""),
        gitingest_tree=state["analysis_result"].get("tree", ""),
        truncated_content=state["analysis_result"].get("content", "")
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert Docker engineer specializing in analyzing build failures and providing actionable improvements."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    result = await llm_client.call_llm(
        model=model,
        messages=messages,
        stream=True
    )

    if result["success"]:
        reflection_response = result["content"]
        try:
            reflection_data = json.loads(reflection_response)
            reflection_result = {
                "needed": True,
                "improvements": reflection_data.get("improvements", []),
                "base_image_suggestion": reflection_data.get("base_image_suggestion", ""),
                "key_issues": reflection_data.get("key_issues", [])
            }
        except json.JSONDecodeError:
            reflection_result = {
                "needed": True,
                "improvements": [{"type": "general", "description": reflection_response}],
                "base_image_suggestion": "",
                "key_issues": []
            }
    else:
        reflection_result = {
            "needed": True,
            "improvements": [{"type": "general", "description": "reflection failed"}],
            "base_image_suggestion": "",
            "key_issues": []
        }

    return {"reflection_result": reflection_result}


@auto_message_handler
async def improve_dockerfile(state: LangGraphWorkflowState) -> Dict[str, Any]:
    """Improve Dockerfile node with enhanced error handling and state management"""
    ws_manager = state.get("ws_manager")

    try:
        # 安全获取迭代计数并更新
        current_iteration = state.get("iteration", 0)
        # 确保迭代次数是整数
        try:
            current_iteration = int(current_iteration)
        except (ValueError, TypeError):
            current_iteration = 0

        new_iteration = current_iteration + 1

        # Check if we have valid analysis and clone data with better error handling
        if not state.get("analysis_result") or not isinstance(state["analysis_result"], dict):
            error_msg = "仓库分析结果缺失或格式错误"
            if ws_manager:
                await ws_manager.send_error(error_msg)
            return {
                "dockerfile_result": {"success": False, "error": error_msg},
                "iteration": new_iteration
            }

        if not state.get("clone_result") or not isinstance(state["clone_result"], dict):
            error_msg = "仓库克隆结果缺失或格式错误"
            if ws_manager:
                await ws_manager.send_error(error_msg)
            return {
                "dockerfile_result": {"success": False, "error": error_msg},
                "iteration": new_iteration
            }

        # 发送迭代信息
        if ws_manager:
            await ws_manager.send_build_log(f"🔄 改进Dockerfile (第 {new_iteration} 次尝试)\n")

        # 安全获取反思结果
        reflection_result = state.get("reflection_result", {})
        if not isinstance(reflection_result, dict):
            reflection_result = {}

        # Prepare additional instructions based on reflection results
        additional_instructions = ""
        if reflection_result.get("needed", False):
            # 支持多种格式的改进建议
            improvements = []
            if "improvements" in reflection_result and isinstance(reflection_result["improvements"], list):
                improvements = reflection_result["improvements"]
            elif "suggestions" in reflection_result and isinstance(reflection_result["suggestions"], list):
                improvements = reflection_result["suggestions"]

            base_image_suggestion = reflection_result.get("base_image_suggestion", "")

            additional_instructions = "根据之前的构建失败，请考虑以下改进建议:\n"
            for improvement in improvements:
                if isinstance(improvement, dict):
                    additional_instructions += f"- {improvement.get('description', improvement.get('type', 'General improvement'))}\n"
                elif isinstance(improvement, str):
                    additional_instructions += f"- {improvement}\n"

            if base_image_suggestion:
                additional_instructions += f"\n建议的基础镜像: {base_image_suggestion}\n"

        # Add user-provided additional instructions
        user_instructions = state.get("additional_instructions", "")
        if user_instructions:
            additional_instructions += f"\n用户额外要求: {user_instructions}\n"

        # Generate improved Dockerfile with better error handling
        stream_support = True
        try:
            dockerfile_result = await create_container_tool(
                gitingest_summary=state["analysis_result"].get("summary", ""),
                gitingest_tree=state["analysis_result"].get("tree", ""),
                gitingest_content=state["analysis_result"].get("content", ""),
                project_name=state["clone_result"].get("repo_name", "project"),
                additional_instructions=additional_instructions,
                model=state.get("model"),
                ws_manager=ws_manager,
                stream=stream_support
            )
        except Exception as e:
            logger.warning(f"Stream mode failed, using non-stream mode: {e}")
            stream_support = False
            try:
                dockerfile_result = await create_container_tool(
                    gitingest_summary=state["analysis_result"].get("summary", ""),
                    gitingest_tree=state["analysis_result"].get("tree", ""),
                    gitingest_content=state["analysis_result"].get("content", ""),
                    project_name=state["clone_result"].get("repo_name", "project"),
                    additional_instructions=additional_instructions,
                    model=state.get("model"),
                    ws_manager=ws_manager,
                    stream=stream_support
                )
            except Exception as fallback_e:
                logger.error(f"Both stream and non-stream mode failed: {fallback_e}")
                dockerfile_result = {"success": False, "error": str(fallback_e)}

        # 确保dockerfile_result是有效的字典格式
        if not isinstance(dockerfile_result, dict):
            dockerfile_result = {"success": False, "error": "Dockerfile生成工具返回了无效结果格式"}

        # 确保success字段存在
        if "success" not in dockerfile_result:
            # 根据是否有dockerfile字段判断成功与否
            if "dockerfile" in dockerfile_result:
                dockerfile_result["success"] = True
            else:
                dockerfile_result["success"] = False

        # 发送改进结果信息
        if ws_manager:
            if dockerfile_result.get("success", False):
                await ws_manager.send_status(f"✅ Dockerfile改进成功 (第 {new_iteration} 次)")
                await ws_manager.send_build_log("✅ Dockerfile改进成功，准备重新构建\n")
            else:
                error_msg = dockerfile_result.get("error", "改进失败原因未知")
                await ws_manager.send_error(f"❌ Dockerfile改进失败: {error_msg}")
                await ws_manager.send_build_log(f"❌ Dockerfile改进失败: {error_msg}\n")

        return {
            "dockerfile_result": dockerfile_result,
            "iteration": new_iteration
        }
    except KeyError as e:
        error_msg = f"Dockerfile改进时遇到KeyError: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ws_manager:
            await ws_manager.send_error(error_msg)
        return {
            "dockerfile_result": {"success": False, "error": error_msg},
            "iteration": state.get("iteration", 0) + 1
        }
    except Exception as e:
        error_msg = f"Dockerfile改进过程中出错: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if ws_manager:
            await ws_manager.send_error(error_msg)
        return {
            "dockerfile_result": {"success": False, "error": error_msg},
            "iteration": state.get("iteration", 0) + 1
        }


async def should_continue(state: LangGraphWorkflowState) -> str:
    """Decision node: determine whether to continue building or end"""
    ws_manager = state.get("ws_manager")

    # Send decision node status update
    if ws_manager:
        await ws_manager.send_node_update("Decision: Continue?", 1.0, "evaluating")

    # Check for early failures (clone or analyze)
    if not state.get("clone_result") or not state["clone_result"].get("success"):
        if ws_manager:
            await ws_manager.send_status("❌ 仓库克隆失败，工作流结束")
            await ws_manager.send_error("无法继续，仓库克隆失败")
            await ws_manager.send_node_complete("Decision: Continue?", {"decision": "clone_failed"})
        return "early_failure"

    if not state.get("analysis_result"):
        if ws_manager:
            await ws_manager.send_status("❌ 仓库分析失败，工作流结束")
            await ws_manager.send_error("无法继续，仓库分析失败")
            await ws_manager.send_node_complete("Decision: Continue?", {"decision": "analysis_failed"})
        return "early_failure"

    # If build succeeded, continue to wiki
    if state.get("build_result") and state["build_result"].get("success"):
        if ws_manager:
            await ws_manager.send_status("✅ 构建成功，准备生成Wiki")
            await ws_manager.send_node_complete("Decision: Continue?", {"decision": "success"})
        return "success"

    # If max iterations reached, end
    if state["iteration"] >= state["max_iterations"]:
        if ws_manager:
            await ws_manager.send_status(f"⏹️ 已达到最大迭代次数 ({state['max_iterations']})，工作流结束")
            await ws_manager.send_error("已达到最大迭代次数，Docker镜像构建失败")
            await ws_manager.send_node_complete("Decision: Continue?", {"decision": "max_iterations_reached"})
        return "max_iterations_reached"

    # If build failed, go to reflection
    if state.get("build_result") and not state["build_result"].get("success"):
        if ws_manager:
            await ws_manager.send_status("🔄 构建失败，进入反思阶段")
            await ws_manager.send_node_complete("Decision: Continue?", {"decision": "reflect"})
        return "reflect"

    if ws_manager:
        await ws_manager.send_status("❓ 未知状态，工作流结束")
        await ws_manager.send_node_complete("Decision: Continue?", {"decision": "unknown"})
    return "end"


async def should_retry(state: LangGraphWorkflowState) -> str:
    """Decision node: determine whether to retry building"""
    ws_manager = state.get("ws_manager")

    # Send decision node status update
    if ws_manager:
        await ws_manager.send_node_update("Decision: Retry?", 1.0, "evaluating")

    # If Dockerfile was improved and iteration count is within limit, retry
    if state["dockerfile_result"]["success"] and state["iteration"] < state["max_iterations"]:
        if ws_manager:
            await ws_manager.send_status("🔄 Dockerfile已改进，重新尝试构建")
            await ws_manager.send_build_log(f"🔄 Dockerfile已改进，重新尝试构建 (第 {state['iteration'] + 1} 次尝试)\n")
            await ws_manager.send_node_complete("Decision: Retry?", {"decision": "retry"})
        return "retry"

    # Otherwise end
    decision = "end"
    if ws_manager:
        if not state["dockerfile_result"]["success"]:
            await ws_manager.send_status("❌ Dockerfile改进失败，工作流结束")
            await ws_manager.send_error("Dockerfile改进失败，已达到最大迭代次数或改进过程出错")
            decision = "dockerfile_failed"
        elif state["iteration"] >= state["max_iterations"]:
            await ws_manager.send_status("⏹️ 已达到最大迭代次数，工作流结束")
            await ws_manager.send_error("已达到最大迭代次数，Docker镜像构建失败")
            decision = "max_iterations"
        else:
            await ws_manager.send_status("🔚 工作流结束")

        await ws_manager.send_node_complete("Decision: Retry?", {"decision": decision})

    return "end"


def create_langgraph_workflow() -> StateGraph:
    """
    Create enhanced LangGraph workflow with automatic message forwarding.
    
    Returns:
        StateGraph: Configured LangGraph workflow instance
    """
    try:
        # Create workflow with error handling
        workflow = StateGraph(LangGraphWorkflowState)

        # Add nodes
        workflow.add_node("clone", clone_repository)
        workflow.add_node("analyze", analyze_repository)
        workflow.add_node("generate", generate_dockerfile)
        workflow.add_node("build", build_docker)
        workflow.add_node("test", test_env)
        workflow.add_node("wiki", generate_wiki)
        workflow.add_node("reflect", reflect_on_failure)
        workflow.add_node("improve", improve_dockerfile)

        # Set entry point
        workflow.set_entry_point("clone")

        # Set workflow edges
        workflow.add_edge("clone", "analyze")
        workflow.add_edge("analyze", "generate")
        workflow.add_edge("generate", "build")
        workflow.add_edge("build", "test")

        # Enhanced conditional edges with error handling
        async def safe_should_continue_async(state):
            """Async wrapper for should_continue that properly awaits the async function"""
            try:
                return await should_continue(state)
            except Exception as e:
                # Log error and return safe default
                logger.error(f"Error in should_continue decision: {str(e)}")
                ws_manager = state.get("ws_manager")
                if ws_manager:
                    import asyncio
                    asyncio.create_task(ws_manager.send_error(f"决策逻辑错误: {str(e)}"))
                return "end"

        def safe_should_continue(state):
            """Sync wrapper that handles the async decision node"""
            try:
                import asyncio
                # Try to get the current event loop, create one if none exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Create a new event loop for this thread if none exists
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # If loop is running, we can't use run_until_complete directly
                    # Instead, use a new loop for this operation
                    if loop._thread_id != asyncio.current_thread().ident:
                        # If current thread doesn't own the loop, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Use a new task in the existing loop
                    future = asyncio.create_task(safe_should_continue_async(state))
                    return loop.run_until_complete(future)
                else:
                    # If loop is not running, run it directly
                    return loop.run_until_complete(safe_should_continue_async(state))
            except Exception as e:
                logger.error(f"Error in safe_should_continue wrapper: {str(e)}")
                return "end"

        # Define a new decision function that considers generate_wiki flag
        async def should_continue_with_wiki(state):
            """Decision function that considers both test success and generate_wiki flag"""
            # First check test_result status
            test_result = state.get("test_result", {})
            ws_manager = state.get("ws_manager")
            
            # If test was successful
            if test_result.get("success"):
                generate_wiki = state.get("generate_wiki", True)  # Default to True if not specified
                if ws_manager:
                    await ws_manager.send_status(f"✅ 测试成功，{'准备生成Wiki' if generate_wiki else '工作流结束'}")
                if generate_wiki:
                    return "wiki"
                else:
                    return "end"
            else:
                # Test failed
                error_msg = test_result.get("error", "测试失败")
                if ws_manager:
                    await ws_manager.send_status(f"❌ 测试失败，工作流结束")
                    await ws_manager.send_error(f"测试失败: {error_msg}")
                return "end"
            
            # Fallback to original logic if needed
            result = await safe_should_continue_async(state)
            return result

        def safe_should_continue_with_wiki(state):
            """Sync wrapper for should_continue_with_wiki"""
            try:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    if loop._thread_id != asyncio.current_thread().ident:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    future = asyncio.create_task(should_continue_with_wiki(state))
                    return loop.run_until_complete(future)
                else:
                    return loop.run_until_complete(should_continue_with_wiki(state))
            except Exception as e:
                logger.error(f"Error in safe_should_continue_with_wiki: {str(e)}")
                return "end"

        # Set conditional edges with the new decision function
        workflow.add_conditional_edges(
            "test",
            safe_should_continue_with_wiki,
            {
                "wiki": "wiki",  # Only go to wiki if generate_wiki is True
                "success": END,  # Legacy support, should not be reached
                "max_iterations_reached": END,
                "reflect": "reflect",
                "early_failure": END,
                "end": END
            }
        )

        # Enhanced reflection flow with error handling
        async def safe_should_retry_async(state):
            """Async wrapper for should_retry that properly awaits the async function"""
            try:
                return await should_retry(state)
            except Exception as e:
                # Log error and return safe default
                logger.error(f"Error in should_retry decision: {str(e)}")
                ws_manager = state.get("ws_manager")
                if ws_manager:
                    import asyncio
                    asyncio.create_task(ws_manager.send_error(f"重试决策逻辑错误: {str(e)}"))
                return "end"

        def safe_should_retry(state):
            """Sync wrapper that handles the async decision node"""
            try:
                import asyncio
                # Try to get the current event loop, create one if none exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # Create a new event loop for this thread if none exists
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    # If loop is running, we can't use run_until_complete directly
                    # Instead, use a new loop for this operation
                    if loop._thread_id != asyncio.current_thread().ident:
                        # If current thread doesn't own the loop, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Use a new task in the existing loop
                    future = asyncio.create_task(safe_should_retry_async(state))
                    return loop.run_until_complete(future)
                else:
                    # If loop is not running, run it directly
                    return loop.run_until_complete(safe_should_retry_async(state))
            except Exception as e:
                logger.error(f"Error in safe_should_retry wrapper: {str(e)}")
                return "end"

        # Set reflection flow with safer decision function
        workflow.add_edge("reflect", "improve")
        workflow.add_conditional_edges(
            "improve",
            safe_should_retry,
            {
                "retry": "build",
                "end": END
            }
        )

        # Wiki node leads to end
        workflow.add_edge("wiki", END)

        # Add rich metadata for frontend visualization and automatic tracking
        workflow.metadata = {
            "name": "Gitcontainer Workflow",
            "description": "AI-powered Dockerfile generation workflow",
            "version": "1.0.0",
            "nodes": {
                "clone": {
                    "label": "克隆仓库",
                    "category": "git",
                    "description": "从GitHub克隆代码仓库到本地"
                },
                "analyze": {
                    "label": "分析仓库",
                    "category": "analysis",
                    "description": "分析仓库结构和代码内容"
                },
                "generate": {
                    "label": "生成Dockerfile",
                    "category": "generation",
                    "description": "基于仓库分析生成Dockerfile"
                },
                "build": {
                    "label": "构建镜像",
                    "category": "build",
                    "description": "使用生成的Dockerfile构建Docker镜像"
                },
                "test": {
                    "label": "环境验证",
                    "category": "test",
                    "description": "测试构建的镜像环境与验证代码"
                },
                "wiki": {
                    "label": "生成Wiki",
                    "category": "documentation",
                    "description": "为项目生成Wiki文档"
                },
                "reflect": {
                    "label": "反思失败",
                    "category": "intelligence",
                    "description": "分析构建失败原因并提供改进建议"
                },
                "improve": {
                    "label": "改进Dockerfile",
                    "category": "improvement",
                    "description": "根据反思结果改进Dockerfile"
                }
            },
            "edges": [
                {"from": "clone", "to": "analyze"},
                {"from": "analyze", "to": "generate"},
                {"from": "generate", "to": "build"},
                {"from": "build", "to": "test", "condition": "success"},
                {"from": "test", "to": "wiki", "condition": "success"},
                {"from": "build", "to": "reflect", "condition": "reflect"},
                {"from": "reflect", "to": "improve"},
                {"from": "improve", "to": "build", "condition": "retry"},
                {"from": "wiki", "to": "end"}
            ]
        }

        # Add a pre-execution hook to send workflow metadata to frontend with enhanced error handling
        original_compile = workflow.compile

        def enhanced_compile():
            try:
                compiled = original_compile()

                # Store the original invoke methods
                original_ainvoke = compiled.ainvoke
                original_invoke = compiled.invoke

                # Store workflow metadata in a variable that will be accessible in the closure
                workflow_metadata = workflow.metadata

                async def enhanced_ainvoke(state, **kwargs):
                    try:
                        # Initialize state with default values if missing
                        if not isinstance(state, dict):
                            logger.warning("State is not a dictionary, initializing default values")
                            state = {}

                        # Ensure iteration and max_iterations are set
                        if "iteration" not in state:
                            state["iteration"] = 0
                        if "max_iterations" not in state:
                            state["max_iterations"] = 2

                        # Ensure result fields exist with default values
                        for result_field in ["clone_result", "analysis_result", "dockerfile_result", "build_result",
                                             "test_result", "wiki_result"]:
                            if result_field not in state:
                                state[result_field] = {}

                        # Send workflow metadata at the start of execution with error handling
                        ws_manager = state.get("ws_manager")
                        if ws_manager:
                            try:
                                await ws_manager.send_workflow_metadata(workflow_metadata)
                                await ws_manager.send_status("🚀 工作流启动中...")
                                await ws_manager.send_build_log("=== 工作流开始执行 ===\n")
                            except Exception as metadata_error:
                                logger.error(f"Error sending workflow metadata: {str(metadata_error)}")

                        # Execute the original invoke
                        return await original_ainvoke(state, **kwargs)
                    except Exception as invoke_error:
                        logger.error(f"Error in enhanced_ainvoke: {str(invoke_error)}", exc_info=True)
                        # Try to send error via websocket if available
                        try:
                            ws_manager = state.get("ws_manager")
                            if ws_manager:
                                await ws_manager.send_error(f"工作流执行失败: {str(invoke_error)}")
                        except Exception:
                            pass
                        # Re-raise the original error
                        raise

                def enhanced_invoke(state, **kwargs):
                    # For synchronous invoke, we can't send metadata via websocket
                    # since it requires async operations, but we can initialize default values
                    if not isinstance(state, dict):
                        logger.warning("State is not a dictionary, initializing default values")
                        state = {}

                    # Ensure iteration and max_iterations are set
                    if "iteration" not in state:
                        state["iteration"] = 0
                    if "max_iterations" not in state:
                        state["max_iterations"] = 2

                    return original_invoke(state, **kwargs)

                # Replace the invoke methods with our enhanced versions
                compiled.ainvoke = enhanced_ainvoke
                compiled.invoke = enhanced_invoke

                return compiled
            except Exception as compile_error:
                logger.error(f"Error in enhanced_compile: {str(compile_error)}", exc_info=True)
                raise

        # Replace the compile method with our enhanced version
        workflow.compile = enhanced_compile

        return workflow
    except Exception as e:
        logger.error(f"Error creating LangGraph workflow: {str(e)}", exc_info=True)
        raise
