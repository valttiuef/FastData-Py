"""
Backend help manager for loading and resolving help entries from YAML/JSON files.

This module provides the HelpManager class that loads help content from a file
and resolves help entries based on exact matches, aliases, and wildcard patterns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class HelpManager:
    """
    Manages help content loaded from YAML or JSON files.
    
    Resolves help entries by priority:
    1. Exact key match
    2. Alias match
    3. Wildcard/pattern match (e.g., feature:*)
    """

    def __init__(self, file_path: Optional[str | Path] = None):
        """
        Initialize the help manager.
        
        Args:
            file_path: Path to the help YAML/JSON file. If None, no file is loaded initially.
        """
        self._file_path: Optional[Path] = Path(file_path) if file_path else None
        self._data: Dict[str, Any] = {}
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._alias_map: Dict[str, str] = {}  # alias -> primary_key
        self._wildcard_entries: List[tuple[str, str, Dict[str, Any]]] = []  # (pattern, prefix, entry)
        
        if self._file_path:
            self.reload()

    def reload(self) -> None:
        """
        (Re)load the help content from disk.

        Supports a single YAML/JSON file or a directory tree containing
        multiple help files. When a directory is provided, all ``.yaml``,
        ``.yml`` and ``.json`` files are loaded recursively and merged. Later
        files (alphabetical order) override entries from earlier ones.

        Raises:
            FileNotFoundError: If the file or directory doesn't exist, or no
                help files are found inside the directory.
            ValueError: If the file format is invalid
        """
        if not self._file_path:
            raise ValueError("No file path configured")

        if not self._file_path.exists():
            raise FileNotFoundError(f"Help file not found: {self._file_path}")

        if self._file_path.is_dir():
            files = sorted(
                p for p in self._file_path.rglob('*')
                if p.is_file() and p.suffix.lower() in ('.yaml', '.yml', '.json')
            )
            if not files:
                raise FileNotFoundError(
                    f"No help files found in directory: {self._file_path}"
                )

            aggregated: Dict[str, Any] = {"version": None, "metadata": {}, "entries": {}}
            for file in files:
                data = self._load_help_file(file)
                self._merge_data(aggregated, data)
            self._data = aggregated
        else:
            self._data = self._load_help_file(self._file_path)

        # Validate and parse entries
        self._parse_entries()
        logger.info(f"Loaded {len(self._entries)} help entries from {self._file_path}")

    def _load_help_file(self, path: Path) -> Dict[str, Any]:
        """Load a single YAML/JSON help file."""
        suffix = path.suffix.lower()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if suffix in ('.yaml', '.yml'):
                    if not YAML_AVAILABLE:
                        raise ImportError(
                            "PyYAML is not installed. Install it with: pip install pyyaml"
                        )
                    data = yaml.safe_load(f)
                elif suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json"
                    )
        except Exception as e:
            logger.error(f"Failed to load help file {path}: {e}")
            raise

        if not isinstance(data, dict):
            raise ValueError(f"Help file must contain a dictionary at the root: {path}")
        return data

    def _merge_data(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Merge a loaded help file into the aggregated structure."""
        if 'version' in source and source['version'] is not None:
            target['version'] = source['version']

        metadata = source.get('metadata')
        if isinstance(metadata, dict):
            target_metadata = target.setdefault('metadata', {})
            target_metadata.update(metadata)

        entries = source.get('entries')
        if entries is None:
            # Allow files that contain only entries without a wrapper key
            # as long as the structure is a mapping.
            non_meta_keys = set(source.keys()) - {'version', 'metadata'}
            if non_meta_keys:
                entries = {k: v for k, v in source.items() if k in non_meta_keys}
        if isinstance(entries, dict):
            target_entries = target.setdefault('entries', {})
            target_entries.update(entries)

    def _parse_entries(self) -> None:
        """Parse and index the entries from the loaded data."""
        self._entries.clear()
        self._alias_map.clear()
        self._wildcard_entries.clear()
        
        if not isinstance(self._data, dict):
            raise ValueError("Help file must contain a dictionary at the root")
        
        entries = self._data.get('entries', {})
        if not isinstance(entries, dict):
            raise ValueError("'entries' must be a dictionary")
        
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                logger.warning(f"Skipping invalid entry '{key}': not a dictionary")
                continue
            
            # Store the entry
            self._entries[key] = entry
            
            # Build alias map
            aliases = entry.get('aliases', [])
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str):
                        self._alias_map[alias] = key
            
            # Check for wildcard patterns
            if '*' in key:
                # Extract the prefix (part before the *)
                prefix = key.split('*')[0]
                self._wildcard_entries.append((key, prefix, entry))

    def get_entry(self, help_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a help entry by key.
        
        Resolution order:
        1. Exact key match
        2. Alias match
        3. Wildcard/pattern match
        
        Args:
            help_key: The help key to look up
            
        Returns:
            Dictionary containing the help entry, or None if not found
        """
        if not help_key:
            return None
        
        # 1. Try exact match
        if help_key in self._entries:
            return self._entries[help_key].copy()
        
        # 2. Try alias match
        if help_key in self._alias_map:
            primary_key = self._alias_map[help_key]
            if primary_key in self._entries:
                return self._entries[primary_key].copy()
        
        # 3. Try wildcard match
        for pattern, prefix, entry in self._wildcard_entries:
            if help_key.startswith(prefix):
                return entry.copy()
        
        return None

    def get_tooltip(self, help_key: str) -> Optional[str]:
        """
        Get the short tooltip text for a help key.
        
        Args:
            help_key: The help key to look up
            
        Returns:
            The short tooltip string, or None if not found
        """
        entry = self.get_entry(help_key)
        if entry:
            return entry.get('short')
        return None

    def get_body(self, help_key: str) -> Optional[str]:
        """
        Get the full body text (HTML allowed) for a help key.
        
        Args:
            help_key: The help key to look up
            
        Returns:
            The body text string, or None if not found
        """
        entry = self.get_entry(help_key)
        if entry:
            return entry.get('body')
        return None

    def get_title(self, help_key: str) -> Optional[str]:
        """
        Get the title for a help key.
        
        Args:
            help_key: The help key to look up
            
        Returns:
            The title string, or None if not found
        """
        entry = self.get_entry(help_key)
        if entry:
            return entry.get('title')
        return None

    @property
    def version(self) -> Optional[str]:
        """Get the version from metadata."""
        return self._data.get('version')

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self._data.get('metadata', {})

    def list_keys(self) -> List[str]:
        """Return a list of all available help keys (excluding aliases)."""
        return list(self._entries.keys())


_shared_help_manager: HelpManager | None = None

def get_help_manager(file_path: str | Path | None = None) -> HelpManager:
    global _shared_help_manager

    if _shared_help_manager is None:
        if file_path is None:
            raise RuntimeError("HelpManager not initialized. Provide file_path on first call.")
        _shared_help_manager = HelpManager(file_path)
        return _shared_help_manager

    if file_path is not None and Path(file_path) != _shared_help_manager._file_path:
        raise RuntimeError("HelpManager already initialized with a different file_path.")

    return _shared_help_manager
