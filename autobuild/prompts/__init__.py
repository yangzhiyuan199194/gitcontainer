"""
Prompts package for Gitcontainer application.

This package contains all the prompts used in the application in a centralized location.
"""

from .dockerfile import create_dockerfile_prompt, create_reflection_prompt
from .gitingest import create_dockerfile_selection_prompt

__all__ = [
    "create_dockerfile_prompt",
    "create_reflection_prompt",
    "create_dockerfile_selection_prompt"
]