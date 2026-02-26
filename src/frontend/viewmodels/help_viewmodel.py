from __future__ import annotations
"""
View model layer for the help system.

Mediates between help widgets (frontend) and the help manager (backend).
"""


import logging
import re
from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

from backend.help_manager import HelpManager
from backend.prompts_manager import PromptManager
from core.paths import get_resource_path

logger = logging.getLogger(__name__)


class HelpViewModel(QObject):
    """
    View model for context-sensitive help system.
    
    Provides a Qt-friendly interface between help widgets and the backend help manager.
    Emits signals when help content is loaded or errors occur.
    """

    # Signals
    help_loaded = Signal()  # Emitted when help file is (re)loaded
    help_error = Signal(str)  # Emitted when an error occurs

    def __init__(self, help_manager: HelpManager, parent: Optional[QObject] = None):
        """
        Initialize the help view model.
        
        Args:
            help_manager: The backend help manager instance
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._help_manager = help_manager
        self._ask_counts: dict[str, int] = {}
        self._body_extensions: dict[str, Callable[[], str]] = {}
        
        # Initialize the prompt manager for loading prompt templates
        self._prompt_manager: Optional[PromptManager] = None
        try:
            prompts_path = get_resource_path("prompts")
            self._prompt_manager = PromptManager(prompts_path)
        except Exception as e:
            logger.warning(f"Could not load prompts manager: {e}. Using fallback prompts.")

    def get_tooltip(self, help_key: str) -> str:
        """
        Get tooltip text for a help key.
        
        Args:
            help_key: The help key to look up
            
        Returns:
            Tooltip text, or empty string if not found
        """
        try:
            tooltip = self._help_manager.get_tooltip(help_key)
            return tooltip or ""
        except Exception as e:
            logger.error(f"Error getting tooltip for '{help_key}': {e}")
            return ""

    def get_title(self, help_key: str) -> str:
        """
        Get title for a help key.
        
        Args:
            help_key: The help key to look up
            
        Returns:
            Title text, or empty string if not found
        """
        try:
            title = self._help_manager.get_title(help_key)
            return title or ""
        except Exception as e:
            logger.error(f"Error getting title for '{help_key}': {e}")
            return ""

    def get_body(self, help_key: str) -> str:
        """
        Get body HTML for a help key.
        
        Args:
            help_key: The help key to look up
            
        Returns:
            Body HTML text, or empty string if not found
        """
        try:
            body = self._help_manager.get_body(help_key) or ""
        except Exception as e:
            logger.error(f"Error getting body for '{help_key}': {e}")
            body = ""

        extra_body = ""
        provider = self._body_extensions.get(help_key)
        if provider is not None:
            try:
                extra_body = provider() or ""
            except Exception as e:
                logger.warning(f"Error building extra help body for '{help_key}': {e}")
                extra_body = ""

        if body and extra_body:
            return f"{body}\n{extra_body}"
        return body or extra_body

    def has_help(self, help_key: str) -> bool:
        """
        Check if help exists for a given key.
        
        Args:
            help_key: The help key to check
            
        Returns:
            True if help content exists, False otherwise
        """
        try:
            entry = self._help_manager.get_entry(help_key)
            return entry is not None
        except Exception as e:
            logger.error(f"Error checking help for '{help_key}': {e}")
            return False

    def set_body_extension(self, help_key: str, provider: Optional[Callable[[], str]]) -> None:
        """Attach a callable that can append extra body content for a help key."""
        if not help_key:
            return
        if provider is None:
            self._body_extensions.pop(help_key, None)
        else:
            self._body_extensions[help_key] = provider

    def reload(self) -> bool:
        """
        Reload the help file from disk.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self._help_manager.reload()
            self.help_loaded.emit()
            return True
        except Exception as e:
            error_msg = f"Failed to reload help: {e}"
            logger.error(error_msg)
            self.help_error.emit(error_msg)
            return False

    # ------------------------------------------------------------------
    def build_ai_prompt(self, help_key: str) -> tuple[str, int]:
        """Build a progressive prompt for the given help topic."""

        if not help_key:
            return "", 0

        title = self.get_title(help_key)
        short = self.get_tooltip(help_key)
        body = self.get_body(help_key)

        if not title or not short or not body:
            entry = self._get_entry(help_key)
            if not title:
                title = entry.get("title") or title
            if not short:
                short = entry.get("short") or short
            if not body:
                body = entry.get("body") or body

        context_lines = [f"Help key: {help_key}"]
        if title:
            context_lines.append(f"Title: {title}")
        if short:
            context_lines.append(f"Summary: {short}")
        if body:
            context_lines.append(f"Details: {self._strip_html(body)}")

        context = "\n".join(context_lines)

        count = self._ask_counts.get(help_key, 0) + 1
        self._ask_counts[help_key] = count

        prompt_steps = self._get_prompt_steps()
        prompt_template = prompt_steps[min(count - 1, len(prompt_steps) - 1)]
        footer = self._get_prompt_footer()
        prompt = f"{prompt_template}\n\nExisting information:\n{context}\n\n{footer}"
        prompt = self._append_regression_suggestions(help_key, prompt)
        return prompt, count

    # ------------------------------------------------------------------
    def _get_prompt_steps(self) -> list[str]:
        """Get help prompt steps from manager or fallback."""
        if self._prompt_manager:
            try:
                steps = self._prompt_manager.get_help_prompt_steps()
                if steps:
                    return steps
            except Exception as e:
                logger.warning(f"Error getting prompt steps from manager: {e}")
        
        # Fallback prompts
        return [
            "Provide a concise overview of the subject below.",
            "Add more detail and practical guidance about the subject.",
            "Give deeper explanations with examples for the subject.",
            "Highlight edge cases, pitfalls, and tips for the subject.",
            "Summarize the subject with actionable next steps and references.",
        ]
    
    def _get_prompt_footer(self) -> str:
        """Get help prompt footer from manager or fallback."""
        if self._prompt_manager:
            try:
                footer = self._prompt_manager.get_help_prompt_footer()
                if footer:
                    return footer
            except Exception as e:
                logger.warning(f"Error getting prompt footer from manager: {e}")
        
        # Fallback footer
        return "Explain in plain language so the user can act on it."

    @property
    def _prompt_steps(self) -> list[str]:
        """Deprecated: Use _get_prompt_steps() instead."""
        return self._get_prompt_steps()

    def _get_entry(self, help_key: str) -> dict:
        try:
            entry = self._help_manager.get_entry(help_key)
            return entry or {}
        except Exception:
            return {}

    @staticmethod
    def _strip_html(value: str) -> str:
        if not value:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    @staticmethod
    def _append_regression_suggestions(help_key: str, prompt: str) -> str:
        if help_key not in {
            "controls.regression.feature_selection",
            "controls.regression.models",
        }:
            return prompt
        guidance = (
            "Also suggest suitable regression models or feature selection methods based on the dataset size "
            "(rows and selected feature count)."
        )
        return f"{prompt}\n\n{guidance}"

    @property
    def version(self) -> Optional[str]:
        """Get the help file version."""
        try:
            return self._help_manager.version
        except Exception:
            return None

    @property
    def metadata(self) -> dict:
        """Get the help file metadata."""
        try:
            return self._help_manager.metadata
        except Exception:
            return {}


# Singleton instance
_shared_help_viewmodel: Optional[HelpViewModel] = None


def get_help_viewmodel(help_manager: Optional[HelpManager] = None, parent: Optional[QObject] = None) -> HelpViewModel:
    """
    Get or create the shared HelpViewModel instance.
    
    Args:
        help_manager: The help manager to use. Only used on first call.
        parent: Optional parent QObject
        
    Returns:
        The shared HelpViewModel instance
        
    Raises:
        ValueError: If no help_manager is provided on first call
    """
    global _shared_help_viewmodel
    
    if _shared_help_viewmodel is None:
        if help_manager is None:
            raise ValueError("help_manager must be provided on first call to get_help_viewmodel")
        _shared_help_viewmodel = HelpViewModel(help_manager, parent=parent)
    elif parent is not None and _shared_help_viewmodel.parent() is None:
        _shared_help_viewmodel.setParent(parent)
    
    return _shared_help_viewmodel
