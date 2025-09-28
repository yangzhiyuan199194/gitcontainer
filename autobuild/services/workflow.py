"""
Workflow service for Gitcontainer application.

This module provides the multi-agent workflow implementation for generating and building Docker images.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from typing_extensions import Annotated

from fastapi import WebSocketDisconnect
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

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
from autobuild.utils import get_websocket_manager
from autobuild.utils.build_history import build_history_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create settings instance
settings = Settings()


class WorkflowState(TypedDict):
    """定义工作流状态"""
    repo_url: Annotated[str, "append"]
    additional_instructions: str
    model: Optional[str]
    clone_result: Dict[str, Any]
    analysis_result: Dict[str, Any]
    dockerfile_result: Dict[str, Any]
    build_result: Dict[str, Any]
    reflection_result: Dict[str, Any]
    iteration: int
    max_iterations: int
    final_result: Dict[str, Any]
    ws_log_file_path: Optional[str]
    websocket: Optional[Any]
    messages: List[Any]
    # Wiki generation fields
    wiki_result: Dict[str, Any]


# 定义工具函数
async def clone_repository(state: WorkflowState) -> Dict[str, Any]:
    """克隆仓库工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path",None)
    ws_manager = get_websocket_manager(websocket,ws_log_file_path)
    if websocket:
        await ws_manager.send_status("🔄 Cloning repository...")
        await ws_manager.send_phase_start("[克隆阶段开始]", "normal")
    logger.info("克隆Agent开始工作")
    
    clone_result = await clone_repo_tool(
        github_url=state["repo_url"],
        ws_manager=ws_manager
    )
    
    if websocket:
        await ws_manager.send_phase_end("[克隆阶段结束]", "normal")

    # 只返回实际更新的字段
    return {"clone_result": clone_result}


async def analyze_repo(state: WorkflowState) -> Dict[str, Any]:
    """分析仓库工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    if websocket:
        await ws_manager.send_status("🔍 Analyzing repository...")
        await ws_manager.send_phase_start("[分析阶段开始]", "normal")
    
    logger.info("分析Agent开始工作")
    
    clone_result = state["clone_result"]

    # 分析仓库
    analysis_result = await gitingest_tool(clone_result["local_path"],state["model"], ws_manager=ws_manager)
    
    if websocket:
        await ws_manager.send_phase_end("[分析阶段结束]", "normal")
    
    # 只返回实际更新的字段
    return {"analysis_result": analysis_result}


async def generate_dockerfile(state: WorkflowState) -> Dict[str, Any]:
    """生成Dockerfile工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    if websocket:
        await ws_manager.send_status("🐳 Generating Dockerfile...")
        await ws_manager.send_phase_start("[生成阶段开始]", "normal")
    
    logger.info("生成Agent开始工作")
    
    # Determine if the selected model supports streaming
    stream_support = settings.get_model_stream_support(state["model"]) if state["model"] else True
    
    # Check if we have reflection results to use as additional instructions
    additional_instructions = state["additional_instructions"] or ""
    if state.get("reflection_result", {}).get("needed"):
        if "root_cause" in state["reflection_result"]:
            # Use detailed analysis results
            improvement_points = "\n".join(state["reflection_result"].get("suggestions", []))
            additional_instructions += f"\n\nBased on the following analysis results, improve the Dockerfile:\n{improvement_points}"
        else:
            # Use simplified improvement information
            additional_instructions += f"\nBuild error message: {state['reflection_result']['error_message']}"
    
    dockerfile_result = await create_container_tool(
        gitingest_summary=state["analysis_result"]["summary"],
        gitingest_tree=state["analysis_result"]["tree"],
        gitingest_content=state["analysis_result"]["content"],
        # git_dockerfile=state["analysis_result"]["git_dockerfile"],
        project_name=state["clone_result"]["repo_name"],
        additional_instructions=additional_instructions,
        model=state["model"],
        ws_manager=ws_manager,
        stream=stream_support
    )
    
    if websocket:
        await ws_manager.send_phase_end("[生成阶段结束]", "normal")
    
    # 只返回实际更新的字段
    return {"dockerfile_result": dockerfile_result}


async def build_image(state: WorkflowState) -> Dict[str, Any]:
    """构建Docker镜像工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    if websocket:
        try:
            await ws_manager.send_status("🔨 Building Docker image...")
            await ws_manager.send_phase_start("[构建阶段开始]", "normal")
            await ws_manager.send_build_log("🚀 开始构建 Docker 镜像...\n")
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected during phase start messages")
            return {"build_result": {
                "success": False,
                "error": "WebSocket disconnected before build could start"
            }}
    
    logger.info("构建Agent开始工作")
    
    if not state["dockerfile_result"]["success"]:
        if websocket:
            try:
                await ws_manager.send_build_log("❌ Dockerfile生成失败，无法构建镜像\n")
                await ws_manager.send_phase_end("[构建阶段结束]", "normal")
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected during phase end message")
        return {"build_result": {
            "success": False,
            "error": "Dockerfile generation failed, cannot build image"
        }}
    
    if websocket:
        try:
            await ws_manager.send_build_log(f"📦 项目名称: {state['clone_result']['repo_name']}\n")
            await ws_manager.send_build_log(f"📁 本地路径: {state['clone_result']['local_path']}\n")
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected during build log messages")
            return {"build_result": {
                "success": False,
                "error": "WebSocket disconnected during build process"
            }}
    
    build_result = await build_docker_image(
        dockerfile_content=state["dockerfile_result"]["dockerfile"],
        project_name=state["clone_result"]["repo_name"],
        local_path=state["clone_result"]["local_path"],
        ws_manager=ws_manager
    )
    
    # Send build result information
    if websocket:
        try:
            if build_result["success"]:
                await ws_manager.send_build_log(f"✅ Docker 镜像构建完成: {build_result['image_tag']}\n")
            else:
                await ws_manager.send_build_log(f"❌ Docker 镜像构建失败: {build_result.get('error', 'Unknown error')}\n")
                # 发送错误消息，确保前端能正确更新步骤状态
                await ws_manager.send_error(f"Docker 镜像构建失败: {build_result.get('error', 'Unknown error')}")
            await ws_manager.send_phase_end("[构建阶段结束]", "normal")
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected during final build messages")
    
    # 只返回实际更新的字段
    return {"build_result": build_result}


async def generate_wiki(state: WorkflowState) -> Dict[str, Any]:
    """生成Wiki文档工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    if websocket:
        await ws_manager.send_status("📚 Generating Wiki documentation...")
        await ws_manager.send_phase_start("[Wiki生成阶段开始]", "normal")
        await ws_manager.send_chunk("🔍 Starting Wiki generation process...\n")
    
    logger.info("Wiki生成Agent开始工作")
    
    # Extract owner and repo from repo_url
    repo_url = state["repo_url"]
    if "github.com/" in repo_url:
        owner_repo = repo_url.split("github.com/")[-1].rstrip("/")
        if owner_repo.endswith(".git"):
            owner_repo = owner_repo[:-4]
        parts = owner_repo.split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
        else:
            owner, repo = "unknown", "unknown"
    else:
        owner, repo = "unknown", "unknown"
    
    # Generate wiki
    wiki_result = await wiki_generator_tool(
        local_repo_path=state["clone_result"]["local_path"],
        owner=owner,
        repo=repo,
        model=state["model"],
        ws_manager=ws_manager
    )
    
    if websocket:
        await ws_manager.send_phase_end("[Wiki生成阶段结束]", "normal")
    
    # 只返回实际更新的字段
    return {"wiki_result": wiki_result}


async def reflect_on_failure(state: WorkflowState) -> Dict[str, Any]:
    """反思构建失败原因工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    if websocket:
        await ws_manager.send_status("🤔 Reflecting on build failure...")
        await ws_manager.send_phase_start("[反思阶段开始]", "smart")
        await ws_manager.send_build_log("🔍 正在分析构建失败原因...\n")
    
    logger.info("反思Agent开始工作")
    
    if state["build_result"]["success"]:
        # 如果构建成功，不需要反思
        reflection_result = {
            "needed": False,
            "improvements": []
        }
        if websocket:
            await ws_manager.send_build_log("✅ 构建成功，无需反思\n")
            await ws_manager.send_phase_end("[反思阶段结束]", "smart")
        # 只返回实际更新的字段
        return {"reflection_result": reflection_result}
    
    # 构建失败，需要反思
    logger.info("检测到构建失败，开始分析原因")
    
    # 获取失败日志和错误信息
    build_log = state["build_result"].get("build_log", "")
    error_message = state["build_result"].get("error", "")
    
    # 获取当前 Dockerfile 内容
    dockerfile_content = state["dockerfile_result"].get("dockerfile", "")
    
    if websocket:
        await ws_manager.send_build_log("💡 正在使用 AI 分析失败原因和改进建议...\n")
    
    # 使用 LLM 分析构建失败原因
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
    
    # 生成反思消息
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
    
    # 使用流式响应以实时显示分析过程
    result = await llm_client.call_llm(
        model=model,
        messages=messages,
        stream=True
    )

    if result["success"]:
        reflection_response = result["content"]
        try:
            # 解析 LLM 响应
            reflection_data = json.loads(reflection_response)
            reflection_result = {
                "needed": True,
                "improvements": reflection_data.get("improvements", []),
                "base_image_suggestion": reflection_data.get("base_image_suggestion", ""),
                "key_issues": reflection_data.get("key_issues", [])
            }
        except json.JSONDecodeError:
            # 如果解析失败，返回原始响应
            reflection_result = {
                "needed": True,
                "improvements": [{"type": "general", "description": reflection_response}],
                "base_image_suggestion": "",
                "key_issues": []
            }

        if websocket:
            await ws_manager.send_phase_end("[反思阶段结束]", "smart")
    else:
        reflection_result = {
            "needed": True,
            "improvements": [{"type": "general", "description": "reflection failed"}],
            "base_image_suggestion": "",
            "key_issues": []
        }
    # 只返回实际更新的字段
    return {"reflection_result": reflection_result}


async def improve_dockerfile(state: WorkflowState) -> Dict[str, Any]:
    """改进Dockerfile工具"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    if websocket:
        await ws_manager.send_status("🔧 Improving Dockerfile...")
        await ws_manager.send_phase_start("[改进阶段开始]", "smart")
        await ws_manager.send_build_log(f"🔄 改进Dockerfile (第 {state['iteration'] + 1} 次尝试)\n")
    
    logger.info("改进Dockerfile Agent开始工作")
    
    # 准备额外指令，包括反思结果
    additional_instructions = ""
    if state.get("reflection_result", {}).get("needed", False):
        improvements = state["reflection_result"].get("improvements", [])
        base_image_suggestion = state["reflection_result"].get("base_image_suggestion", "")
        
        additional_instructions = "根据之前的构建失败，请考虑以下改进建议:\n"
        for improvement in improvements:
            additional_instructions += f"- {improvement.get('description', improvement.get('type', 'General improvement'))}\n"
        
        if base_image_suggestion:
            additional_instructions += f"\n建议的基础镜像: {base_image_suggestion}\n"
    
    # 添加用户提供的额外指令
    user_instructions = state.get("additional_instructions", "")
    if user_instructions:
        additional_instructions += f"\n用户额外要求: {user_instructions}\n"
    
    # 改进 Dockerfile
    stream_support = True
    try:
        # 尝试使用流模式
        dockerfile_result = await create_container_tool(
            gitingest_summary=state["analysis_result"]["summary"],
            gitingest_tree=state["analysis_result"]["tree"],
            gitingest_content=state["analysis_result"]["content"],
            # git_dockerfile=state["analysis_result"]["git_dockerfile"],
            project_name=state["clone_result"]["repo_name"],
            additional_instructions=additional_instructions,
            model=state["model"],
            ws_manager=ws_manager,
            stream=stream_support
        )
    except Exception as e:
        logger.warning(f"流模式失败，使用非流模式: {e}")
        stream_support = False
        dockerfile_result = await create_container_tool(
            gitingest_content=state["analysis_result"]["content"],
            # git_dockerfile=state["analysis_result"]["git_dockerfile"],
            project_name=state["clone_result"]["repo_name"],
            additional_instructions=additional_instructions,
            model=state["model"],
            ws_manager=ws_manager,
            stream=stream_support
        )
    
    # 只返回实际更新的字段
    return {
        "dockerfile_result": dockerfile_result,
        "iteration": state["iteration"] + 1
    }


async def should_continue(state: WorkflowState) -> str:
    """决定是否继续构建或结束"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    # 如果构建成功，结束
    if state["build_result"].get("success"):
        if websocket:
            await ws_manager.send_status("✅ 构建成功，工作流结束")
        return "success"
    
    # 如果达到最大迭代次数，结束
    if state["iteration"] >= state["max_iterations"]:
        if websocket:
            await ws_manager.send_status(f"⏹️ 已达到最大迭代次数 ({state['max_iterations']})，工作流结束")
            await ws_manager.send_build_log(f"⏹️ 已达到最大迭代次数 ({state['max_iterations']})，工作流结束\n")
            # 添加明确的错误消息类型，确保前端能正确处理
            await ws_manager.send_error("已达到最大迭代次数，Docker镜像构建失败")
        return "max_iterations_reached"
    
    # 如果需要反思，进入反思流程
    if not state["build_result"].get("success"):
        if websocket:
            await ws_manager.send_status("🔄 构建失败，进入反思阶段")
        return "reflect"
    
    if websocket:
        await ws_manager.send_status("🔚 工作流结束")
    return "end"


async def should_retry(state: WorkflowState) -> str:
    """决定是否重试构建"""
    websocket = state.get("websocket")
    ws_log_file_path = state.get("ws_log_file_path", None)
    ws_manager = get_websocket_manager(websocket, ws_log_file_path)
    
    # 如果改进后可以重试
    if state["dockerfile_result"]["success"] and state["iteration"] < state["max_iterations"]:
        if websocket:
            await ws_manager.send_status("🔄 Dockerfile已改进，重新尝试构建")
            await ws_manager.send_build_log(f"🔄 Dockerfile已改进，重新尝试构建 (第 {state['iteration'] + 1} 次尝试)\n")
        return "retry"
    
    # 否则结束
    if websocket:
        if not state["dockerfile_result"]["success"]:
            await ws_manager.send_status("❌ Dockerfile改进失败，工作流结束")
            await ws_manager.send_build_log("❌ Dockerfile改进失败，工作流结束\n")
            # 添加明确的错误消息类型，确保前端能正确处理
            await ws_manager.send_error("Dockerfile改进失败，已达到最大迭代次数或改进过程出错")
        elif state["iteration"] >= state["max_iterations"]:
            await ws_manager.send_status("⏹️ 已达到最大迭代次数，工作流结束")
            await ws_manager.send_build_log("⏹️ 已达到最大迭代次数，工作流结束\n")
            # 添加明确的错误消息类型，确保前端能正确处理
            await ws_manager.send_error("已达到最大迭代次数，Docker镜像构建失败")
        else:
            await ws_manager.send_status("🔚 工作流结束")
    return "end"


def create_workflow() -> StateGraph:
    """创建工作流图"""
    # 创建工作流
    workflow = StateGraph(WorkflowState)

    # 添加节点
    workflow.add_node("clone", clone_repository)
    workflow.add_node("analyze", analyze_repo)
    workflow.add_node("generate", generate_dockerfile)
    workflow.add_node("build", build_image)
    workflow.add_node("wiki", generate_wiki)
    workflow.add_node("reflect", reflect_on_failure)
    workflow.add_node("improve", improve_dockerfile)

    # 设置入口点
    workflow.set_entry_point("clone")

    # 设置并行流程 - clone完成后同时执行analyze和wiki
    workflow.add_edge("clone", "analyze")
    # workflow.add_edge("clone", "wiki")

    # 设置docker正常流程
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "build")

    # 设置条件边
    workflow.add_conditional_edges(
        "build",
        should_continue,
        {
            "success": END,
            "max_iterations_reached": END,
            "reflect": "reflect"
        }
    )

    # 设置反思流程
    workflow.add_edge("reflect", "improve")
    workflow.add_conditional_edges(
        "improve",
        should_retry,
        {
            "retry": "build",
            "end": END
        }
    )

    return workflow