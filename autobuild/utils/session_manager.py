"""
Session management utility for Gitcontainer application.

This module provides functionality for managing user sessions during Dockerfile generation.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional


class SessionManager:
    """Manages user sessions for the application."""
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
    
    def create_session(self, data: Dict[str, Any]) -> str:
        """
        Create a new session.
        
        Args:
            data (Dict[str, Any]): Session data
            
        Returns:
            str: Session ID
        """
        session_id = str(hash(str(data) + str(time.time())))
        self.sessions[session_id] = {
            **data,
            "status": "pending",
            "created_at": time.time()
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Optional[Dict[str, Any]]: Session data or None if not found
        """
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id (str): Session identifier
            data (Dict[str, Any]): Data to update
            
        Returns:
            bool: True if session was updated, False if not found
        """
        if session_id in self.sessions:
            self.sessions[session_id].update(data)
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: True if session was deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    async def cleanup_expired_sessions(self, max_age: int = 3600) -> None:
        """
        Clean up expired sessions.
        
        Args:
            max_age (int): Maximum session age in seconds (default: 1 hour)
        """
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, session_data in self.sessions.items()
            if current_time - session_data.get("created_at", 0) > max_age
        ]
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
    
    async def start_cleanup_task(self, interval: int = 600) -> None:
        """
        Start periodic cleanup task.
        
        Args:
            interval (int): Cleanup interval in seconds (default: 10 minutes)
        """
        async def _cleanup_loop():
            while True:
                await self.cleanup_expired_sessions()
                await asyncio.sleep(interval)
        
        self._cleanup_task = asyncio.create_task(_cleanup_loop())
    
    def stop_cleanup_task(self) -> None:
        """Stop the cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()