"""
Build history management for Gitcontainer application.

This module provides functionality for saving and retrieving build history.
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class BuildHistoryManager:
    """Manages build history for the application."""
    
    def __init__(self, history_dir: str = "build_history"):
        """Initialize build history manager."""
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
    
    def _get_repo_hash(self, repo_url: str) -> str:
        """
        Generate a hash for the repository URL to use as identifier.
        
        Args:
            repo_url (str): Repository URL
            
        Returns:
            str: Hash of the repository URL
        """
        # 使用SHA256算法以匹配实际文件名
        return hashlib.sha256(repo_url.encode()).hexdigest()
    
    def _get_build_path(self, repo_url: str) -> Path:
        """
        Get the path for a build record.
        
        Args:
            repo_url (str): Repository URL
            
        Returns:
            Path: Path to the build record
        """
        repo_hash = self._get_repo_hash(repo_url)
        return self.history_dir / f"{repo_hash}.json"
    
    def save_build_record(self, repo_url: str, build_data: Dict[str, Any]) -> bool:
        """
        Save a build record.
        
        Args:
            repo_url (str): Repository URL
            build_data (Dict[str, Any]): Build data to save
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            build_path = self._get_build_path(repo_url)
            
            # Add metadata
            record = {
                "repo_url": repo_url,
                "build_time": datetime.now().isoformat(),
                "status": "success" if build_data.get("success", False) else "failed",
                **build_data
            }
            
            # Save to file
            with open(build_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving build record: {e}")
            return False
    
    def get_build_record(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """
        Get a build record.
        
        Args:
            repo_url (str): Repository URL
            
        Returns:
            Optional[Dict[str, Any]]: Build record or None if not found
        """
        try:
            build_path = self._get_build_path(repo_url)
            
            if not build_path.exists():
                return None
            
            with open(build_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading build record: {e}")
            return None
    
    def get_build_log(self, repo_url: str) -> Optional[str]:
        """
        Get build log content.
        
        Args:
            repo_url (str): Repository URL
            
        Returns:
            Optional[str]: Build log content or None if not found
        """
        try:
            # Get repo hash
            repo_hash = self._get_repo_hash(repo_url)
            
            # Log file path
            log_file_path = self.history_dir / "logs" / f"{repo_hash}.log"
            
            if not log_file_path.exists():
                return None
            
            with open(log_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading build log: {e}")
            return None
    
    def get_all_build_records(self) -> List[Dict[str, Any]]:
        """
        Get all build records.
        
        Returns:
            List[Dict[str, Any]]: List of all build records
        """
        records = []
        
        try:
            for file_path in self.history_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        record = json.load(f)
                        records.append(record)
                except Exception as e:
                    print(f"Error reading build record {file_path}: {e}")
        except Exception as e:
            print(f"Error reading build records: {e}")
        
        # Sort by build time (newest first)
        records.sort(key=lambda x: x.get("build_time", ""), reverse=True)
        return records
    
    def get_successful_builds(self) -> List[Dict[str, Any]]:
        """
        Get all successful build records.
        
        Returns:
            List[Dict[str, Any]]: List of successful build records
        """
        all_records = self.get_all_build_records()
        return [record for record in all_records if record.get("status") == "success"]
    
    def repo_already_built_successfully(self, repo_url: str) -> bool:
        """
        Check if a repository has already been built successfully.
        
        Args:
            repo_url (str): Repository URL
            
        Returns:
            bool: True if repo has been built successfully, False otherwise
        """
        record = self.get_build_record(repo_url)
        return record is not None and record.get("status") == "success"


# Global build history manager instance
build_history_manager = BuildHistoryManager()