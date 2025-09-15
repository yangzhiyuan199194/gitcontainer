import asyncio
import os
import json
from typing import Any, Optional


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


async def send_websocket_message(websocket: Optional[Any], message_type: str, content: str) -> bool:
    """
    Helper function to send WebSocket messages safely.
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