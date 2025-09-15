import logging
import os
from typing import Dict, Any, Optional, List, Callable

from openai import AsyncOpenAI

from tools.utils import emit_ws_message

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    通用LLM调用客户端，支持多种模型和流式/非流式响应
    """
    
    def __init__(self):
        """初始化LLM客户端"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.inf_api_key = os.getenv("INF_API_KEY")
        self.default_model = os.getenv("MODEL", "gpt-4o-mini")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = AsyncOpenAI(api_key=self.inf_api_key, base_url=self.base_url)
    
    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 20000,
        stream: bool = True,
        websocket: Optional[Any] = None,
        response_handler: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        调用LLM生成响应
        
        Args:
            messages: 对话消息列表
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大令牌数
            stream: 是否流式响应
            websocket: WebSocket连接用于流式传输
            response_handler: 响应处理函数
            
        Returns:
            包含响应内容和元数据的字典
        """
        try:
            # 使用提供的模型或回退到默认模型
            model_to_use = model or self.default_model
            
            # 发送状态消息
            websocket_active = await emit_ws_message(websocket, "status", "🤖 正在调用AI模型...")
            
            print(f"Debug - About to make API call")
            print(f"Debug - Model: {model_to_use}")
            print(f"Debug - Messages count: {len(messages)}")
            print(f"Debug - Temperature: {temperature}")
            print(f"Debug - Max tokens: {max_tokens}")
            print(f"Debug - Stream: {stream}")
            
            # 调用LLM
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra_headers={'apikey': self.api_key} if self.api_key else None,
            )
            
            print("Debug - API call initiated successfully")
            
            # 收集响应
            response_content = ""
            if stream:
                # 处理流式响应
                if websocket_active:
                    websocket_active = await emit_ws_message(websocket, "stream_start", "开始生成...")
                print("📝 Response:")
                print("-" * 50)
                
                async for chunk in response:
                    if len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            print(content, end="", flush=True)
                            response_content += content
                            # 只有WebSocket仍然活跃时才发送块
                            if websocket_active:
                                websocket_active = await emit_ws_message(websocket, "chunk", content)
                
                print("\n" + "-" * 50)
                print("✅ Generation complete!\n")
                if websocket_active:
                    await emit_ws_message(websocket, "status", "✅ 生成完成!")
            else:
                # 处理非流式响应
                response_content = response.choices[0].message.content
                print("📝 Response:")
                print("-" * 50)
                print(response_content)
                print("-" * 50)
                print("✅ Generation complete!\n")
                
                # 如果WebSocket活跃，则发送整个响应
                if websocket_active:
                    await emit_ws_message(websocket, "stream_start", "开始生成...")
                    await emit_ws_message(websocket, "chunk", response_content)
                    await emit_ws_message(websocket, "status", "✅ 生成完成!")
            
            # 如果提供了响应处理函数，则使用它处理响应
            if response_handler:
                return await response_handler(response_content)
            
            return {
                "success": True,
                "content": response_content
            }
            
        except Exception as e:
            error_msg = f"LLM调用失败: {str(e)}"
            print(f"Debug - API call failed with error: {str(e)}")
            
            # 发送错误消息
            await emit_ws_message(websocket, "error", error_msg)
            
            return {
                "success": False,
                "error": error_msg
            }