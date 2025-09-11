"""Agent-based system for GitHub to Dockerfile generation with reflection capabilities."""

import asyncio
import json
import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api_analytics.fastapi import Analytics
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from tools import clone_repo_tool, gitingest_tool, create_container_tool, build_docker_image

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="GitHub to Dockerfile Generator - Agent Version")

# Add API Analytics middleware
app.add_middleware(Analytics, api_key=os.getenv("FASTAPI_ANALYTICS_KEY"))

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Store for session data
sessions = {}


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
    websocket: Optional[Any]
    messages: List[Any]


def parse_available_models():
    """
    Parse the AVAILABLE_MODELS environment variable.
    Format: model_name|stream_support,model_name|stream_support,...
    Example: gpt-4o-mini|true,gpt-4o|true,o1-mini|false,o1|false
    """
    available_models_str = os.getenv("AVAILABLE_MODELS", "")
    if not available_models_str:
        return []
    
    models = []
    for model_entry in available_models_str.split(","):
        model_entry = model_entry.strip()
        if "|" in model_entry:
            model_name, stream_support = model_entry.split("|")
            models.append({
                "name": model_name.strip(),
                "stream": stream_support.strip().lower() == "true"
            })
        else:
            # For backward compatibility, if no | is present, assume stream is supported
            models.append({
                "name": model_entry,
                "stream": True
            })
    return models


def get_model_stream_support(model_name):
    """
    Check if a specific model supports streaming based on AVAILABLE_MODELS environment variable.
    """
    models = parse_available_models()
    for model in models:
        if model["name"] == model_name:
            return model["stream"]
    # Default to True if model not found in the list
    return True


# 定义工具函数
async def clone_repository(state: WorkflowState) -> WorkflowState:
    """克隆仓库工具"""
    websocket = state.get("websocket")
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🔄 Cloning repository..."
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_start",
            "content": "[克隆阶段开始]",
            "phase_type": "normal"
        }))
    
    logger.info("克隆Agent开始工作")
    
    clone_result = await clone_repo_tool(
        github_url=state["repo_url"],
        websocket=websocket
    )
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "phase_end",
            "content": "[克隆阶段结束]",
            "phase_type": "normal"
        }))
    
    state["clone_result"] = clone_result
    return state


async def analyze_repository(state: WorkflowState) -> WorkflowState:
    """分析代码结构工具"""
    websocket = state.get("websocket")
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "📊 Analyzing repository structure..."
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_start",
            "content": "[分析阶段开始]",
            "phase_type": "normal"
        }))
    
    logger.info("分析Agent开始工作")
    
    if not state["clone_result"]["success"]:
        state["analysis_result"] = {
            "success": False,
            "error": "Repository cloning failed, cannot analyze"
        }
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "phase_end",
                "content": "[分析阶段结束]",
                "phase_type": "normal"
            }))
        return state
    
    analysis_result = await gitingest_tool(
        local_repo_path=state["clone_result"]["local_path"],
        websocket=websocket
    )
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "phase_end",
            "content": "[分析阶段结束]",
            "phase_type": "normal"
        }))
    
    state["analysis_result"] = analysis_result
    return state


async def generate_dockerfile(state: WorkflowState) -> WorkflowState:
    """生成Dockerfile工具"""
    websocket = state.get("websocket")
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🐳 Generating Dockerfile with AI..."
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_start",
            "content": "[生成阶段开始]",
            "phase_type": "normal"
        }))
    
    logger.info("Dockerfile生成Agent开始工作")
    
    if not state["analysis_result"]["success"]:
        state["dockerfile_result"] = {
            "success": False,
            "error": "Repository analysis failed, cannot generate Dockerfile"
        }
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "phase_end",
                "content": "[生成阶段结束]",
                "phase_type": "normal"
            }))
        return state
    
    # Determine if the selected model supports streaming
    stream_support = get_model_stream_support(state["model"]) if state["model"] else True
    
    dockerfile_result = await create_container_tool(
        gitingest_summary=state["analysis_result"]["summary"],
        gitingest_tree=state["analysis_result"]["tree"],
        gitingest_content=state["analysis_result"]["content"],
        project_name=state["clone_result"]["repo_name"],
        additional_instructions=state["additional_instructions"],
        model=state["model"],
        websocket=websocket,
        stream=stream_support
    )
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "phase_end",
            "content": "[生成阶段结束]",
            "phase_type": "normal"
        }))
    
    state["dockerfile_result"] = dockerfile_result
    return state


async def build_image(state: WorkflowState) -> WorkflowState:
    """构建Docker镜像工具"""
    websocket = state.get("websocket")
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🔨 Building Docker image..."
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_start",
            "content": "[构建阶段开始]",
            "phase_type": "normal"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": "🚀 开始构建 Docker 镜像...\n"
        }))
    
    logger.info("构建Agent开始工作")
    
    if not state["dockerfile_result"]["success"]:
        state["build_result"] = {
            "success": False,
            "error": "Dockerfile generation failed, cannot build image"
        }
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": "❌ Dockerfile生成失败，无法构建镜像\n"
            }))
            await websocket.send_text(json.dumps({
                "type": "phase_end",
                "content": "[构建阶段结束]",
                "phase_type": "normal"
            }))
        return state
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"📦 项目名称: {state['clone_result']['repo_name']}\n"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"📁 本地路径: {state['clone_result']['local_path']}\n"
        }))
    
    build_result = await build_docker_image(
        dockerfile_content=state["dockerfile_result"]["dockerfile"],
        project_name=state["clone_result"]["repo_name"],
        local_path=state["clone_result"]["local_path"],
        websocket=websocket
    )
    
    # Send build result information
    if websocket:
        if build_result["success"]:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": f"✅ Docker 镜像构建完成: {build_result['image_tag']}\n"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": f"❌ Docker 镜像构建失败: {build_result.get('error', 'Unknown error')}\n"
            }))
            # 发送错误消息，确保前端能正确更新步骤状态
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Docker 镜像构建失败: {build_result.get('error', 'Unknown error')}"
            }))
        await websocket.send_text(json.dumps({
            "type": "phase_end",
            "content": "[构建阶段结束]",
            "phase_type": "normal"
        }))
    
    state["build_result"] = build_result
    return state


async def reflect_on_failure(state: WorkflowState) -> WorkflowState:
    """反思构建失败原因工具"""
    websocket = state.get("websocket")
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🤔 Reflecting on build failure..."
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_start",
            "content": "[反思阶段开始]",
            "phase_type": "smart"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": "🔍 正在分析构建失败原因...\n"
        }))
    
    logger.info("反思Agent开始工作")
    
    if state["build_result"]["success"]:
        # 如果构建成功，不需要反思
        state["reflection_result"] = {
            "needed": False,
            "improvements": []
        }
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": "✅ 构建成功，无需反思\n"
            }))
            await websocket.send_text(json.dumps({
                "type": "phase_end",
                "content": "[反思阶段结束]",
                "phase_type": "smart"
            }))
        return state
    
    # 构建失败，需要反思
    logger.info("检测到构建失败，开始分析原因")
    
    # 获取失败日志
    build_log = state["build_result"].get("build_log", "")
    error_message = state["build_result"].get("error", "")
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"❌ 错误信息: {error_message}\n"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"📝 构建日志摘要:\n{build_log[-1000:] if build_log else '无日志'}\n"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": "💡 正在生成改进建议...\n"
        }))
    
    # 在实际应用中，这里应该调用LLM分析错误原因并提出改进建议
    # 简化版本，我们只记录错误信息
    state["reflection_result"] = {
        "needed": True,
        "build_log": build_log,
        "error_message": error_message,
        "improvements": ["需要重新生成Dockerfile以解决构建问题"]
    }
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"🤔 反思构建失败原因: {error_message}\n"
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_end",
            "content": "[反思阶段结束]",
            "phase_type": "smart"
        }))
    
    return state


async def improve_dockerfile(state: WorkflowState) -> WorkflowState:
    """改进Dockerfile工具"""
    websocket = state.get("websocket")
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "status",
            "content": "🔄 Improving Dockerfile based on reflection..."
        }))
        await websocket.send_text(json.dumps({
            "type": "phase_start",
            "content": "[改进阶段开始]",
            "phase_type": "smart"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"🔧 第 {state['iteration'] + 1} 次尝试改进Dockerfile...\n"
        }))
    
    logger.info("改进Agent开始工作")
    
    if not state["reflection_result"]["needed"]:
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": "✅ 无需改进\n"
            }))
            await websocket.send_text(json.dumps({
                "type": "phase_end",
                "content": "[改进阶段结束]",
                "phase_type": "smart"
            }))
        return state
    
    # 基于反思结果生成改进的Dockerfile
    # 在实际应用中，这里应该使用LLM生成新的Dockerfile
    logger.info("基于反思结果重新生成Dockerfile")
    
    if websocket:
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": "📝 基于以下错误信息重新生成Dockerfile:\n"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"   错误: {state['reflection_result']['error_message']}\n"
        }))

    # Determine if the selected model supports streaming
    stream_support = get_model_stream_support(state["model"]) if state["model"] else True
    
    dockerfile_result = await create_container_tool(
        gitingest_summary=state["analysis_result"]["summary"],
        gitingest_tree=state["analysis_result"]["tree"],
        gitingest_content=state["analysis_result"]["content"],
        project_name=state["clone_result"]["repo_name"],
        additional_instructions=f"{state['additional_instructions']}\n构建错误信息: {state['reflection_result']['error_message']}",
        model=state["model"],
        websocket=websocket,
        stream=stream_support
    )
    
    state["dockerfile_result"] = dockerfile_result
    state["iteration"] += 1
    
    if websocket:
        if dockerfile_result["success"]:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": "✅ Dockerfile改进成功\n"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "build_log",
                "content": f"❌ Dockerfile改进失败: {dockerfile_result.get('error', 'Unknown error')}\n"
            }))
            # 添加明确的错误消息类型，确保前端能正确处理
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"Dockerfile改进失败: {dockerfile_result.get('error', 'Unknown error')}"
            }))
        await websocket.send_text(json.dumps({
            "type": "phase_end",
            "content": "[改进阶段结束]",
            "phase_type": "smart"
        }))
        await websocket.send_text(json.dumps({
            "type": "build_log",
            "content": f"🔄 重新生成Dockerfile (第 {state['iteration']} 次尝试)\n"
        }))
    
    return state


def should_continue(state: WorkflowState) -> str:
    """决定是否继续构建或结束"""
    websocket = state.get("websocket")
    
    # 如果构建成功，结束
    if state["build_result"].get("success"):
        if websocket:
            try:
                # 使用 get_event_loop().create_task 替代 create_task 来避免 RuntimeWarning
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "✅ 构建成功，工作流结束"
                })))
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
        return "success"
    
    # 如果达到最大迭代次数，结束
    if state["iteration"] >= state["max_iterations"]:
        if websocket:
            try:
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": f"⏹️ 已达到最大迭代次数 ({state['max_iterations']})，工作流结束"
                })))
                try:
                    asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                        "type": "build_log",
                        "content": f"⏹️ 已达到最大迭代次数 ({state['max_iterations']})，工作流结束\n"
                    })))
                except RuntimeError:
                    # 如果没有运行的事件循环，跳过消息发送
                    pass
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
        return "max_iterations_reached"
    
    # 如果需要反思，进入反思流程
    if not state["build_result"].get("success"):
        if websocket:
            try:
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "🔄 构建失败，进入反思阶段"
                })))
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
        return "reflect"
    
    if websocket:
        try:
            asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                "type": "status",
                "content": "🔚 工作流结束"
            })))
        except RuntimeError:
            # 如果没有运行的事件循环，跳过消息发送
            pass
    return "end"


def should_retry(state: WorkflowState) -> str:
    """决定是否重试构建"""
    websocket = state.get("websocket")
    
    # 如果改进后可以重试
    if state["dockerfile_result"]["success"] and state["iteration"] < state["max_iterations"]:
        if websocket:
            try:
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "🔄 Dockerfile已改进，重新尝试构建"
                })))
                try:
                    asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                        "type": "build_log",
                        "content": f"🔄 Dockerfile已改进，重新尝试构建 (第 {state['iteration'] + 1} 次尝试)\n"
                    })))
                except RuntimeError:
                    # 如果没有运行的事件循环，跳过消息发送
                    pass
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
        return "retry"
    
    # 否则结束
    if websocket:
        if not state["dockerfile_result"]["success"]:
            try:
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "❌ Dockerfile改进失败，工作流结束"
                })))
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "build_log",
                    "content": "❌ Dockerfile改进失败，工作流结束\n"
                })))
                # 添加明确的错误消息类型，确保前端能正确处理
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "Dockerfile改进失败，已达到最大迭代次数或改进过程出错"
                })))
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
        elif state["iteration"] >= state["max_iterations"]:
            try:
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "⏹️ 已达到最大迭代次数，工作流结束"
                })))
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "build_log",
                    "content": "⏹️ 已达到最大迭代次数，工作流结束\n"
                })))
                # 添加明确的错误消息类型，确保前端能正确处理
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "error",
                    "content": "已达到最大迭代次数，Docker镜像构建失败"
                })))
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
        else:
            try:
                asyncio.get_event_loop().create_task(websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "🔚 工作流结束"
                })))
            except RuntimeError:
                # 如果没有运行的事件循环，跳过消息发送
                pass
    return "end"


def create_workflow() -> StateGraph:
    """创建工作流图"""
    # 创建工作流
    workflow = StateGraph(WorkflowState)
    
    # 添加节点
    workflow.add_node("clone", clone_repository)
    workflow.add_node("analyze", analyze_repository)
    workflow.add_node("generate", generate_dockerfile)
    workflow.add_node("build", build_image)
    workflow.add_node("reflect", reflect_on_failure)
    workflow.add_node("improve", improve_dockerfile)
    
    # 设置入口点
    workflow.set_entry_point("clone")
    
    # 设置正常流程
    workflow.add_edge("clone", "analyze")
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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with the input form."""
    # Get available models from environment variable
    available_models = parse_available_models()
    
    # Get current model - first from available models, or fallback to environment or default
    if available_models:
        current_model = available_models[0]["name"]  # Default to first available model
    else:
        current_model = os.getenv("MODEL", "gpt-4o-mini")
    
    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": "",
        "loading": False,
        "streaming": False,
        "result": None,
        "error": None,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


@app.get("/{path:path}", response_class=HTMLResponse)
async def dynamic_github_route(request: Request, path: str):
    """Handle GitHub-style URLs by replacing gitcontainer.com with github.com."""
    # Skip certain paths that shouldn't be treated as GitHub routes
    skip_paths = {"health", "favicon.ico", "favicon-16x16.png", "favicon-32x32.png", "apple-touch-icon.png", "static", "ws"}
    
    # Split path into segments
    segments = [segment for segment in path.split('/') if segment]
    
    # If it's a skip path, let it fall through
    if segments and segments[0] in skip_paths:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    
    # Check if we have at least 2 segments (username/repo)
    if len(segments) < 2:
        # Get available models from environment variable
        available_models = parse_available_models()
        
        # Get current model - first from available models, or fallback to environment or default
        if available_models:
            current_model = available_models[0]["name"]  # Default to first available model
        else:
            current_model = os.getenv("MODEL", "gpt-4o-mini")
        
        return templates.TemplateResponse("index.jinja", {
            "request": request,
            "repo_url": "",
            "loading": False,
            "streaming": False,
            "result": None,
            "error": f"Invalid GitHub URL format. Expected format: gitcontainer.com/username/repository",
            "pre_filled": False,
            "available_models": [model["name"] for model in available_models],
            "current_model": current_model
        })
    
    # Use only the first two segments (username/repo)
    username, repo = segments[0], segments[1]
    github_url = f"https://github.com/{username}/{repo}"
    
    # Get available models from environment variable
    available_models = parse_available_models()
    
    # Get current model - first from available models, or fallback to environment or default
    if available_models:
        current_model = available_models[0]["name"]  # Default to first available model
    else:
        current_model = os.getenv("MODEL", "gpt-4o-mini")
    
    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": github_url,
        "loading": False,
        "streaming": False,
        "result": None,
        "error": None,
        "pre_filled": True,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


@app.post("/", response_class=HTMLResponse)
async def generate_dockerfile_endpoint(
    request: Request, 
    repo_url: str = Form(...),
    additional_instructions_hidden: str = Form(""),
    model: str = Form(None)
):
    """Redirect to streaming page for Dockerfile generation."""
    # Store the repo URL, additional instructions, and model in a session
    session_id = str(hash(repo_url + str(asyncio.get_event_loop().time())))
    sessions[session_id] = {
        "repo_url": repo_url,
        "additional_instructions": additional_instructions_hidden.strip() if additional_instructions_hidden else "",
        "model": model,
        "status": "pending"
    }
    
    # Redirect to streaming page
    # Get available models from environment variable
    available_models = parse_available_models()
    
    # Get current model from form, or first available model, or environment variable
    if model:
        current_model = model
    elif available_models:
        current_model = available_models[0]["name"]  # Default to first available model
    else:
        current_model = os.getenv("MODEL", "gpt-4o-mini")
    
    return templates.TemplateResponse("index.jinja", {
        "request": request,
        "repo_url": repo_url,
        "loading": False,
        "streaming": True,
        "session_id": session_id,
        "result": None,
        "error": None,
        "available_models": [model["name"] for model in available_models],
        "current_model": current_model
    })


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming Dockerfile generation with multi-agent reflection."""
    await websocket.accept()
    print(f"New WebSocket connection: {session_id}")
    
    try:
        if session_id not in sessions:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": "Invalid session ID"
            }))
            return
        
        repo_url = sessions[session_id]["repo_url"]
        additional_instructions = sessions[session_id].get("additional_instructions", "")
        model = sessions[session_id].get("model", None)
        
        # 编译工作流
        workflow = create_workflow()
        app_workflow = workflow.compile()
        
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
            max_iterations=1,
            final_result={},
            websocket=websocket,
            messages=[]
        )
        
        # 运行工作流
        final_state = await app_workflow.ainvoke(initial_state)
        
        # 构造最终结果
        if final_state["build_result"].get("success"):
            final_result = {
                "project_name": final_state["dockerfile_result"].get("project_name", ""),
                "technology_stack": final_state["dockerfile_result"].get("technology_stack", ""),
                "dockerfile": final_state["dockerfile_result"].get("dockerfile", ""),
                "base_image_reasoning": final_state["dockerfile_result"].get("base_image_reasoning", ""),
                "additional_notes": final_state["dockerfile_result"].get("additional_notes", ""),
                "image_build": final_state["build_result"],
                "repo_info": {
                    "name": final_state["clone_result"].get("repo_name", ""),
                    "size_mb": final_state["clone_result"].get("repo_size_mb", 0),
                    "file_count": final_state["clone_result"].get("file_count", 0)
                }
            }
            
            await websocket.send_text(json.dumps({
                "type": "complete",
                "content": "Generation complete!",
                "result": final_result
            }))
            
            # Store result in session for potential refresh
            sessions[session_id]["result"] = final_result
            sessions[session_id]["status"] = "complete"
        else:
            # 构建失败，发送错误信息
            error_msg = final_state["build_result"].get("error", "Unknown error occurred during build")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"构建失败: {error_msg}"
            }))
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        # Check if websocket is still open before trying to send error message
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "content": error_msg
                }))
        except Exception as send_error:
            print(f"Could not send error message, WebSocket likely closed: {send_error}")
    finally:
        # Clean up session data
        try:
            if session_id in sessions:
                sessions[session_id]["status"] = "disconnected"
        except:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    # Load environment variables
    load_dotenv()
    PORT = int(os.getenv("PORT", 8000))  # Different port from main app
    uvicorn.run(app, host="0.0.0.0", port=PORT)