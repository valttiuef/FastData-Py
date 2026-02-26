"""
Backend prompt manager for loading and resolving prompt templates from YAML files.

This module provides the PromptManager class that loads prompt templates from a file
and retrieves them for use throughout the application. This allows users to edit
prompts without modifying Python code.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Manages prompt templates loaded from YAML or JSON files.
    
    Provides methods to retrieve prompt templates for different contexts
    (features analysis, help system, etc.).
    """

    def __init__(self, file_path: Optional[str | Path] = None):
        """
        Initialize the prompt manager.
        
        Args:
            file_path: Path to the prompts YAML/JSON file. If None, no file is loaded initially.
        """
        self._file_path: Optional[Path] = Path(file_path) if file_path else None
        self._data: Dict[str, Any] = {}
        
        if self._file_path:
            self.reload()

    def reload(self) -> None:
        """
        (Re)load the prompt templates from disk.

        Supports a single YAML/JSON file. When a directory is provided, 
        the first `prompts.yaml` or `prompts.json` file found is used.

        Raises:
            FileNotFoundError: If the file or directory doesn't exist
            ValueError: If the file format is invalid
        """
        if not self._file_path:
            raise ValueError("No file path configured")

        if not self._file_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self._file_path}")

        if self._file_path.is_dir():
            # Look for prompts.yaml or prompts.json
            for filename in ("prompts.yaml", "prompts.yml", "prompts.json"):
                file = self._file_path / filename
                if file.exists():
                    self._data = self._load_prompts_file(file)
                    logger.info(f"Loaded prompts from {file}")
                    return
            raise FileNotFoundError(
                f"No prompts.yaml or prompts.json found in directory: {self._file_path}"
            )
        else:
            self._data = self._load_prompts_file(self._file_path)
            logger.info(f"Loaded prompts from {self._file_path}")

    def _load_prompts_file(self, path: Path) -> Dict[str, Any]:
        """Load a single YAML/JSON prompts file."""
        suffix = path.suffix.lower()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if suffix in ('.yaml', '.yml'):
                    if not YAML_AVAILABLE:
                        raise ImportError(
                            "PyYAML is not installed. Install it with: pip install pyyaml"
                        )
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    import json
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported file format: {suffix}")
        except Exception as e:
            logger.error(f"Error loading prompts file {path}: {e}")
            raise

    def get_features_prompt(self, summary_text: str) -> str:
        """
        Get the feature analysis prompt template, filled with provided summary.
        
        Args:
            summary_text: The feature summary to include in the prompt
            
        Returns:
            Complete prompt text, or empty string if template not found
        """
        if not summary_text:
            return ""
        
        try:
            guidance = self._get_nested(["features", "analyze", "guidance"], "")
            instructions = self._get_nested(["features", "analyze", "instructions"], "")
            
            if not guidance:
                return ""
            
            # Replace placeholder with actual summary text
            instructions_filled = instructions.replace("{summary_text}", summary_text)
            
            return f"{guidance}\n\n{instructions_filled}"
        except Exception as e:
            logger.error(f"Error building features prompt: {e}")
            return ""

    def get_help_prompt_steps(self) -> list[str]:
        """
        Get the help system prompt progression steps.
        
        Returns:
            List of prompt templates for progressive help explanation
        """
        try:
            steps = self._get_nested(["help", "steps"], [])
            if isinstance(steps, list):
                return steps
            return []
        except Exception as e:
            logger.error(f"Error getting help prompt steps: {e}")
            return [
                "Provide a concise overview of the subject below.",
                "Add more detail and practical guidance about the subject.",
                "Give deeper explanations with examples for the subject.",
                "Highlight edge cases, pitfalls, and tips for the subject.",
                "Summarize the subject with actionable next steps and references.",
            ]

    def get_help_prompt_footer(self) -> str:
        """
        Get the footer text appended to all help prompts.
        
        Returns:
            Footer text for help prompts
        """
        try:
            footer = self._get_nested(["help", "footer"], "")
            if footer:
                return footer
            return "Explain in plain language so the user can act on it."
        except Exception as e:
            logger.error(f"Error getting help prompt footer: {e}")
            return "Explain in plain language so the user can act on it."

    def get_som_maps_prompt(self, summary_text: str) -> str:
        """
        Get the SOM analysis prompt template, filled with provided summary.
        
        Args:
            summary_text: SOM summary to include in the prompt
            
        Returns:
            Complete prompt text, or empty string if template not found
        """
        if not summary_text:
            return ""
        
        try:
            guidance = self._get_nested(["som_maps", "analyze", "guidance"], "")
            instructions = self._get_nested(["som_maps", "analyze", "instructions"], "")
            
            if not guidance:
                return ""
            
            instructions_filled = instructions.replace("{summary_text}", summary_text)
            
            return f"{guidance}\n\n{instructions_filled}"
        except Exception as e:
            logger.error(f"Error building SOM prompt: {e}")
            return ""

    def get_model_details_prompt(self, summary_text: str) -> str:
        """
        Get the model run analysis prompt template, filled with provided summary.

        Args:
            summary_text: Model run summary to include in the prompt

        Returns:
            Complete prompt text, or empty string if template not found
        """
        if not summary_text:
            return ""

        try:
            guidance = self._get_nested(["model_details", "analyze", "guidance"], "")
            instructions = self._get_nested(["model_details", "analyze", "instructions"], "")

            if not guidance:
                return ""

            instructions_filled = instructions.replace("{summary_text}", summary_text)

            return f"{guidance}\n\n{instructions_filled}"
        except Exception as e:
            logger.error(f"Error building model details prompt: {e}")
            return ""

    def _get_nested(self, keys: list[str], default: Any = None) -> Any:
        """
        Navigate nested dictionary structure.
        
        Args:
            keys: List of keys to traverse
            default: Default value if path not found
            
        Returns:
            Value at nested path, or default if not found
        """
        current = self._data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return default
            else:
                return default
        return current
