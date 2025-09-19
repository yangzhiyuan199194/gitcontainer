"""
LLM client service for Gitcontainer application.

This module provides a unified interface for interacting with various LLMs.
"""

import logging
from typing import Dict, Any, Optional, List, Callable

from openai import AsyncOpenAI

from autobuild.core.config import Settings
from autobuild.utils import get_websocket_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
settings = Settings()


class LLMClient:
    """
    Universal LLM calling client, supporting multiple models and streaming/non-streaming responses.
    """
    
    def __init__(self):
        """Initialize LLM client."""
        self.api_key = settings.openai_api_key
        self.base_url = settings.base_url
        self.inf_api_key = settings.inf_api_key
        self.default_model = settings.model
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = AsyncOpenAI(api_key=self.inf_api_key, base_url=self.base_url)
    
    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 16384,
        stream: bool = True,
        websocket: Optional[Any] = None,
        response_handler: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Call LLM to generate response.
        
        Args:
            messages: Conversation messages list
            model: Model to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens
            stream: Whether to stream response
            websocket: WebSocket connection for streaming
            response_handler: Response handling function
            
        Returns:
            Dictionary containing response content and metadata
        """
        # Initialize WebSocket manager
        ws_manager = get_websocket_manager(websocket)
        
        try:
            # Use provided model or fallback to default model
            model_to_use = model or self.default_model
            
            # Send status message
            await ws_manager.send_status("🤖 正在调用AI模型...")
            
            print(f"Debug - About to make API call")
            print(f"Debug - Model: {model_to_use}")
            print(f"Debug - Messages count: {len(messages)}")
            print(f"Debug - Temperature: {temperature}")
            print(f"Debug - Max tokens: {max_tokens}")
            print(f"Debug - Stream: {stream}")
            
            # Call LLM
            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra_headers={'apikey': self.api_key} if self.api_key else None,
            )
            
            print("Debug - API call initiated successfully")
            
            # Collect response
            response_content = ""
            if stream:
                # Handle streaming response
                await ws_manager.send_stream_start("开始生成...")
                print("📝 Response:")
                print("-" * 50)
                
                async for chunk in response:
                    if len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            print(content, end="", flush=True)
                            response_content += content
                            # Send chunk
                            await ws_manager.send_chunk(content)
                
                print("\n" + "-" * 50)
                print("✅ Generation complete!\n")
                await ws_manager.send_status("✅ 生成完成!")
            else:
                # Handle non-streaming response
                response_content = response.choices[0].message.content
                print("📝 Response:")
                print("-" * 50)
                print(response_content)
                print("-" * 50)
                print("✅ Generation complete!\n")
                
                # Send entire response
                await ws_manager.send_stream_start("开始生成...")
                await ws_manager.send_chunk(response_content)
                await ws_manager.send_status("✅ 生成完成!")
            
            # If response handler is provided, use it to process response
            if response_handler:
                return await response_handler(response_content)
            
            return {
                "success": True,
                "content": response_content
            }
            
        except Exception as e:
            error_msg = f"LLM调用失败: {str(e)}"
            print(f"Debug - API call failed with error: {str(e)}")
            
            # Send error message
            await ws_manager.send_error(error_msg)
            
            return {
                "success": False,
                "error": error_msg
            }