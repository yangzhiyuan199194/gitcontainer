import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from tools import clone_repo_tool, gitingest_tool, create_container_tool, build_docker_image

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """定义工作流状态"""
    repo_url: str
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


async def clone_agent(state: WorkflowState) -> WorkflowState:
    """克隆仓库的Agent"""
    logger.info("克隆Agent开始工作")
    
    clone_result = await clone_repo_tool(
        github_url=state["repo_url"],
        websocket=None  # 在实际应用中，这里应该传递websocket连接
    )
    
    state["clone_result"] = clone_result
    return state


async def analyze_agent(state: WorkflowState) -> WorkflowState:
    """分析代码结构的Agent"""
    logger.info("分析Agent开始工作")
    
    if not state["clone_result"]["success"]:
        state["analysis_result"] = {
            "success": False,
            "error": "Repository cloning failed, cannot analyze"
        }
        return state
    
    analysis_result = await gitingest_tool(
        local_repo_path=state["clone_result"]["local_path"],
        websocket=None  # 在实际应用中，这里应该传递websocket连接
    )
    
    state["analysis_result"] = analysis_result
    return state


async def dockerfile_generation_agent(state: WorkflowState) -> WorkflowState:
    """生成Dockerfile的Agent"""
    logger.info("Dockerfile生成Agent开始工作")
    
    if not state["analysis_result"]["success"]:
        state["dockerfile_result"] = {
            "success": False,
            "error": "Repository analysis failed, cannot generate Dockerfile"
        }
        return state
    
    dockerfile_result = await create_container_tool(
        gitingest_summary=state["analysis_result"]["summary"],
        gitingest_tree=state["analysis_result"]["tree"],
        gitingest_content=state["analysis_result"]["content"],
        git_dockerfile=state["analysis_result"]["git_dockerfile"],
        project_name=state["clone_result"]["repo_name"],
        additional_instructions=state["additional_instructions"],
        model=state["model"],
        websocket=None,  # 在实际应用中，这里应该传递websocket连接
        stream=False
    )
    
    state["dockerfile_result"] = dockerfile_result
    return state


async def build_agent(state: WorkflowState) -> WorkflowState:
    """构建Docker镜像的Agent"""
    logger.info("构建Agent开始工作")
    
    if not state["dockerfile_result"]["success"]:
        state["build_result"] = {
            "success": False,
            "error": "Dockerfile generation failed, cannot build image"
        }
        return state
    
    build_result = await build_docker_image(
        dockerfile_content=state["dockerfile_result"]["dockerfile"],
        project_name=state["clone_result"]["repo_name"],
        local_path=state["clone_result"]["local_path"],
        websocket=None  # 在实际应用中，这里应该传递websocket连接
    )
    
    state["build_result"] = build_result
    return state


async def reflection_agent(state: WorkflowState) -> WorkflowState:
    """反思Agent，分析构建失败原因并提出改进方案"""
    logger.info("反思Agent开始工作")
    
    if state["build_result"]["success"]:
        # 如果构建成功，不需要反思
        state["reflection_result"] = {
            "needed": False,
            "improvements": []
        }
        return state
    
    # 构建失败，需要反思
    logger.info("检测到构建失败，开始分析原因")
    
    # 获取失败日志
    build_log = state["build_result"].get("build_log", "")
    error_message = state["build_result"].get("error", "")
    
    # 在实际应用中，这里应该调用LLM分析错误原因并提出改进建议
    # 简化版本，我们只记录错误信息
    state["reflection_result"] = {
        "needed": True,
        "build_log": build_log,
        "error_message": error_message,
        "improvements": ["需要重新生成Dockerfile以解决构建问题"]
    }
    
    return state


async def improvement_agent(state: WorkflowState) -> WorkflowState:
    """改进Agent，基于反思结果生成新的Dockerfile"""
    logger.info("改进Agent开始工作")
    
    if not state["reflection_result"]["needed"]:
        return state
    
    # 基于反思结果生成改进的Dockerfile
    # 在实际应用中，这里应该使用LLM生成新的Dockerfile
    logger.info("基于反思结果重新生成Dockerfile")
    
    # 简化版本，我们重新运行Dockerfile生成
    # 实际应用中应该将错误信息和改进建议传递给create_container_tool
    dockerfile_result = await create_container_tool(
        gitingest_summary=state["analysis_result"]["summary"],
        gitingest_tree=state["analysis_result"]["tree"],
        gitingest_content=state["analysis_result"]["content"],
        git_dockerfile=state["analysis_result"]["git_dockerfile"],
        project_name=state["clone_result"]["repo_name"],
        additional_instructions=f"{state['additional_instructions']}\n构建错误信息: {state['reflection_result']['error_message']}",
        model=state["model"],
        websocket=None,
        stream=False
    )
    
    state["dockerfile_result"] = dockerfile_result
    state["iteration"] += 1
    
    return state


def should_continue_building(state: WorkflowState) -> str:
    """决定是否继续构建或结束"""
    # 如果构建成功，结束
    if state["build_result"]["success"]:
        return "success"
    
    # 如果达到最大迭代次数，结束
    if state["iteration"] >= state["max_iterations"]:
        return "max_iterations_reached"
    
    # 如果需要反思，进入反思流程
    return "reflect"


def should_retry_building(state: WorkflowState) -> str:
    """决定是否重试构建"""
    # 如果改进后可以重试
    if state["dockerfile_result"]["success"] and state["iteration"] < state["max_iterations"]:
        return "retry"
    
    # 否则结束
    return "end"


def create_workflow() -> StateGraph:
    """创建工作流图"""
    workflow = StateGraph(WorkflowState)
    
    # 添加节点
    workflow.add_node("clone", clone_agent)
    workflow.add_node("analyze", analyze_agent)
    workflow.add_node("generate_dockerfile", dockerfile_generation_agent)
    workflow.add_node("build", build_agent)
    workflow.add_node("reflect", reflection_agent)
    workflow.add_node("improve", improvement_agent)
    
    # 设置入口点
    workflow.set_entry_point("clone")
    
    # 设置正常流程
    workflow.add_edge("clone", "analyze")
    workflow.add_edge("analyze", "generate_dockerfile")
    workflow.add_edge("generate_dockerfile", "build")
    
    # 设置条件边
    workflow.add_conditional_edges(
        "build",
        should_continue_building,
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
        should_retry_building,
        {
            "retry": "build",
            "end": END
        }
    )
    
    return workflow


async def run_multiagent_workflow(
    repo_url: str,
    additional_instructions: str = "",
    model: Optional[str] = None,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    运行多Agent工作流
    
    Args:
        repo_url: GitHub仓库URL
        additional_instructions: 附加指令
        model: 使用的模型
        max_iterations: 最大迭代次数
        
    Returns:
        工作流执行结果
    """
    # 编译工作流
    workflow = create_workflow()
    app = workflow.compile()
    
    # 初始化状态
    initial_state = WorkflowState(
        repo_url=repo_url,
        additional_instructions=additional_instructions,
        model=model,
        clone_result={},
        analysis_result={},
        dockerfile_result={},
        build_result={},
        reflection_result={},
        iteration=0,
        max_iterations=max_iterations,
        final_result={}
    )
    
    # 运行工作流
    final_state = await app.ainvoke(initial_state)
    
    # 构造最终结果
    result = {
        "success": final_state["build_result"].get("success", False),
        "clone_result": final_state["clone_result"],
        "analysis_result": final_state["analysis_result"],
        "dockerfile_result": final_state["dockerfile_result"],
        "build_result": final_state["build_result"],
        "reflection_result": final_state["reflection_result"],
        "iterations": final_state["iteration"]
    }
    
    if final_state["build_result"].get("success"):
        result["final_result"] = {
            "project_name": final_state["dockerfile_result"].get("project_name", ""),
            "technology_stack": final_state["dockerfile_result"].get("technology_stack", ""),
            "dockerfile": final_state["dockerfile_result"].get("dockerfile", ""),
            "base_image_reasoning": final_state["dockerfile_result"].get("base_image_reasoning", ""),
            "additional_notes": final_state["dockerfile_result"].get("additional_notes", ""),
            "image_build": final_state["build_result"]
        }
    
    return result