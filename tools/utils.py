import asyncio
import json
from typing import Any, Optional


async def emit_ws_message(websocket: Optional[Any], message_type: str, content: str) -> bool:
    """
    Helper function to emit WebSocket messages safely.
    Returns True if message was sent successfully, False if WebSocket is closed or error occurred.
    """
    if websocket is not None:
        try:
            # Check if WebSocket is still open
            if hasattr(websocket, 'client_state') and websocket.client_state.name != 'CONNECTED':
                print(f"WebSocket is not connected. State: {websocket.client_state}")
                return False

            message = {
                "type": message_type,
                "content": content,
                "timestamp": asyncio.get_event_loop().time()
            }
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            # WebSocket is likely closed, stop trying to send messages
            print(f"WebSocket closed or error occurred: {e}")
            return False
    return False
