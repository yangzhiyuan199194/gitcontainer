"""
WebSocket Message Manager for Gitcontainer application.

This module provides a centralized manager for WebSocket message handling,
making it easier to send different types of messages consistently.
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Enumeration of supported message types."""
    STATUS = "status"
    CHUNK = "chunk"
    STREAM_START = "stream_start"
    BUILD_LOG = "build_log"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    ERROR = "error"
    COMPLETE = "complete"


class WebSocketManager:
    """
    Centralized WebSocket message manager.
    
    This class provides a clean interface for sending various types of messages
    over WebSocket connections, handling connection state checks and error handling.
    """
    
    def __init__(self, websocket: Optional[Any] = None, log_file: Optional[str] = None):
        """
        Initialize WebSocket manager.
        
        Args:
            websocket (Optional[Any]): WebSocket connection
            log_file (Optional[str]): File to log messages to
        """
        self.websocket = websocket
        self.log_file = log_file
        self.log_buffer = []
    
    async def send_message(self, message_type: Union[MessageType, str], content: str, **kwargs) -> bool:
        """
        Send a message through WebSocket.
        
        Args:
            message_type (Union[MessageType, str]): Type of message to send
            content (str): Message content
            **kwargs: Additional data to include in the message
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        # Log message
        self._log_message(message_type, content, **kwargs)
        
        if self.websocket is None:
            return False
            
        try:
            # Check if WebSocket is still open
            if hasattr(self.websocket, 'client_state') and self.websocket.client_state.name != 'CONNECTED':
                logger.debug(f"WebSocket is not connected. State: {self.websocket.client_state}")
                return False
            
            # Prepare message
            message = {
                "type": message_type.value if isinstance(message_type, MessageType) else message_type,
                "content": content,
                "timestamp": asyncio.get_event_loop().time(),
                **kwargs
            }
            
            # Send message
            await self.websocket.send_text(json.dumps(message))
            return True
            
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            return False
    
    def _log_message(self, message_type: Union[MessageType, str], content: str, **kwargs):
        """
        Log message to file if log_file is specified.
        
        Args:
            message_type (Union[MessageType, str]): Type of message
            content (str): Message content
            **kwargs: Additional data
        """
        if self.log_file:
            log_entry = {
                "type": message_type.value if isinstance(message_type, MessageType) else message_type,
                "content": content,
                "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop() else None,
                **kwargs
            }
            self.log_buffer.append(log_entry)
            
            # Write to file immediately for real-time logging
            try:
                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Append to file
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log message to file: {e}")
    
    async def send_status(self, content: str) -> bool:
        """Send status message."""
        return await self.send_message(MessageType.STATUS, content)
    
    async def send_chunk(self, content: str) -> bool:
        """Send chunk message."""
        return await self.send_message(MessageType.CHUNK, content)
    
    async def send_stream_start(self, content: str = "开始生成...") -> bool:
        """Send stream start message."""
        return await self.send_message(MessageType.STREAM_START, content)
    
    async def send_build_log(self, content: str) -> bool:
        """Send build log message."""
        return await self.send_message(MessageType.BUILD_LOG, content)
    
    async def send_phase_start(self, content: str, phase_type: str = "normal") -> bool:
        """Send phase start message."""
        return await self.send_message(MessageType.PHASE_START, content, phase_type=phase_type)
    
    async def send_phase_end(self, content: str, phase_type: str = "normal") -> bool:
        """Send phase end message."""
        return await self.send_message(MessageType.PHASE_END, content, phase_type=phase_type)
    
    async def send_error(self, content: str) -> bool:
        """Send error message."""
        return await self.send_message(MessageType.ERROR, content)
    
    async def send_complete(self, content: str, result: Optional[dict] = None) -> bool:
        """Send complete message."""
        kwargs = {"result": result} if result else {}
        return await self.send_message(MessageType.COMPLETE, content, **kwargs)
    
    def update_websocket(self, websocket: Any) -> None:
        """
        Update WebSocket connection.
        
        Args:
            websocket (Any): New WebSocket connection
        """
        self.websocket = websocket
    
    def save_log_buffer(self, file_path: str) -> bool:
        """
        Save the log buffer to a file.
        
        Args:
            file_path (str): Path to save the log buffer
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            return True
        except Exception as e:
            logger.warning(f"Failed to save log buffer: {e}")
            return False


# Global WebSocket manager instance
ws_manager = WebSocketManager()


def get_websocket_manager(websocket: Optional[Any] = None, log_file: Optional[str] = None) -> WebSocketManager:
    """
    Get WebSocket manager instance.
    
    Args:
        websocket (Optional[Any]): WebSocket connection to use
        log_file (Optional[str]): File to log messages to
        
    Returns:
        WebSocketManager: WebSocket manager instance
    """
    if websocket is not None or log_file is not None:
        return WebSocketManager(websocket, log_file)
    return ws_manager


# Backward compatibility functions
async def send_websocket_message(websocket: Optional[Any], message_type: str, content: str) -> bool:
    """
    Helper function to send WebSocket messages safely (backward compatibility).
    
    Args:
        websocket (Optional[Any]): WebSocket connection
        message_type (str): Type of message
        content (str): Message content
        
    Returns:
        bool: True if message was sent successfully, False if WebSocket is closed or error occurred
    """
    manager = get_websocket_manager(websocket)
    return await manager.send_message(message_type, content)