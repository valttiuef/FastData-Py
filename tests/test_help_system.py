"""
Tests for the help system (manager, viewmodel, and widgets).
"""

import os
import sys
import tempfile
from pathlib import Path

# Avoid GUI crashes in headless CI - MUST be set before any Qt imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from backend.help_manager import HelpManager, get_help_manager
from frontend.viewmodels.help_viewmodel import HelpViewModel, get_help_viewmodel
from frontend.widgets.help_widgets import InfoButton, HelpPopup, attach_help, get_help_key


# Test data
SAMPLE_HELP_YAML = """
version: 1
metadata:
  app: "TestApp"
  updated: "2025-12-14"

entries:
  test.key:
    title: "Test Title"
    short: "Test tooltip"
    body: "Test body content"
    aliases: ["test.alias", "another.alias"]

  test.noalias:
    title: "No Alias Test"
    short: "Short text"
    body: "Body text"

  feature:*:
    title: "Wildcard Feature"
    short: "A wildcard match"
    body: "Matches any feature:* key"
"""

SAMPLE_HELP_JSON = """
{
  "version": 1,
  "metadata": {
    "app": "TestApp",
    "updated": "2025-12-14"
  },
  "entries": {
    "test.key": {
      "title": "Test Title",
      "short": "Test tooltip",
      "body": "Test body content"
    }
  }
}
"""


@pytest.fixture
def yaml_help_file():
    """Create a temporary YAML help file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(SAMPLE_HELP_YAML)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def json_help_file():
    """Create a temporary JSON help file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(SAMPLE_HELP_JSON)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def yaml_help_dir(tmp_path):
    """Create a temporary help directory with multiple files."""
    root = tmp_path / "help"
    root.mkdir()

    # Metadata file
    (root / "metadata.yaml").write_text(
        """version: 2
metadata:
  app: "TestApp"
  updated: "2025-12-14"
""",
        encoding="utf-8",
    )

    # Entries grouped by folder
    tabs_dir = root / "tabs"
    tabs_dir.mkdir()
    (tabs_dir / "core.yaml").write_text(
        """entries:
  test.key:
    title: "Test Title"
    short: "Test tooltip"
    body: "Test body content"
    aliases: ["test.alias", "another.alias"]

  test.noalias:
    title: "No Alias Test"
    short: "Short text"
    body: "Body text"
""",
        encoding="utf-8",
    )

    widgets_dir = root / "widgets"
    widgets_dir.mkdir()
    (widgets_dir / "wildcards.yaml").write_text(
        """entries:
  feature:*:
    title: "Wildcard Feature"
    short: "A wildcard match"
    body: "Matches any feature:* key"
""",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def help_manager(yaml_help_dir):
    """Create a HelpManager instance with test data."""
    return HelpManager(yaml_help_dir)


@pytest.fixture
def qapp():
    """Ensure QApplication exists for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestHelpManager:
    """Tests for HelpManager backend."""

    def test_load_yaml_file(self, yaml_help_file):
        """Test loading a YAML help file."""
        manager = HelpManager(yaml_help_file)
        assert manager.version == 1
        assert manager.metadata['app'] == "TestApp"
        assert len(manager.list_keys()) >= 3

    def test_load_directory(self, yaml_help_dir):
        """Test loading a directory of YAML files."""
        manager = HelpManager(yaml_help_dir)
        assert manager.version == 2
        assert manager.metadata['app'] == "TestApp"
        assert set(manager.list_keys()) == {"test.key", "test.noalias", "feature:*"}

    def test_load_json_file(self, json_help_file):
        """Test loading a JSON help file."""
        manager = HelpManager(json_help_file)
        assert manager.version == 1
        assert manager.metadata['app'] == "TestApp"

    def test_get_entry_exact_match(self, help_manager):
        """Test exact key match."""
        entry = help_manager.get_entry("test.key")
        assert entry is not None
        assert entry['title'] == "Test Title"
        assert entry['short'] == "Test tooltip"
        assert entry['body'] == "Test body content"

    def test_get_entry_alias_match(self, help_manager):
        """Test alias matching."""
        entry = help_manager.get_entry("test.alias")
        assert entry is not None
        assert entry['title'] == "Test Title"
        
        entry2 = help_manager.get_entry("another.alias")
        assert entry2 is not None
        assert entry2['title'] == "Test Title"

    def test_get_entry_wildcard_match(self, help_manager):
        """Test wildcard pattern matching."""
        entry = help_manager.get_entry("feature:AE_RMS")
        assert entry is not None
        assert entry['title'] == "Wildcard Feature"
        
        entry2 = help_manager.get_entry("feature:another_column")
        assert entry2 is not None
        assert entry2['title'] == "Wildcard Feature"

    def test_get_entry_not_found(self, help_manager):
        """Test that non-existent keys return None."""
        entry = help_manager.get_entry("nonexistent.key")
        assert entry is None

    def test_get_tooltip(self, help_manager):
        """Test getting tooltip text."""
        tooltip = help_manager.get_tooltip("test.key")
        assert tooltip == "Test tooltip"
        
        tooltip2 = help_manager.get_tooltip("nonexistent")
        assert tooltip2 is None

    def test_get_body(self, help_manager):
        """Test getting body text."""
        body = help_manager.get_body("test.key")
        assert body == "Test body content"

    def test_get_title(self, help_manager):
        """Test getting title text."""
        title = help_manager.get_title("test.key")
        assert title == "Test Title"

    def test_reload(self, yaml_help_dir):
        """Test reloading when new files are added to the help directory."""
        manager = HelpManager(yaml_help_dir)
        initial_keys = manager.list_keys()

        # Add a new file
        new_file = yaml_help_dir / "extra.yaml"
        new_file.write_text(
            """entries:
  new.entry:
    title: "New Entry"
    short: "New short"
    body: "New body"
""",
            encoding="utf-8",
        )

        # Reload
        manager.reload()
        new_keys = manager.list_keys()
        
        assert len(new_keys) > len(initial_keys)
        assert "new.entry" in new_keys

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            manager = HelpManager("/nonexistent/path/help.yaml")

    def test_no_file_path(self):
        """Test creating manager without a file path."""
        manager = HelpManager()
        assert manager._file_path is None
        
        with pytest.raises(ValueError):
            manager.reload()

    def test_list_keys(self, help_manager):
        """Test listing all available keys."""
        keys = help_manager.list_keys()
        assert "test.key" in keys
        assert "test.noalias" in keys
        assert "feature:*" in keys


class TestHelpViewModel:
    """Tests for HelpViewModel."""

    def test_get_tooltip(self, help_manager, qapp):
        """Test getting tooltip through view model."""
        viewmodel = HelpViewModel(help_manager)
        tooltip = viewmodel.get_tooltip("test.key")
        assert tooltip == "Test tooltip"

    def test_get_title(self, help_manager, qapp):
        """Test getting title through view model."""
        viewmodel = HelpViewModel(help_manager)
        title = viewmodel.get_title("test.key")
        assert title == "Test Title"

    def test_get_body(self, help_manager, qapp):
        """Test getting body through view model."""
        viewmodel = HelpViewModel(help_manager)
        body = viewmodel.get_body("test.key")
        assert body == "Test body content"

    def test_has_help(self, help_manager, qapp):
        """Test checking if help exists."""
        viewmodel = HelpViewModel(help_manager)
        assert viewmodel.has_help("test.key") is True
        assert viewmodel.has_help("nonexistent") is False

    def test_version_property(self, help_manager, qapp):
        """Test version property."""
        viewmodel = HelpViewModel(help_manager)
        assert viewmodel.version == 2

    def test_metadata_property(self, help_manager, qapp):
        """Test metadata property."""
        viewmodel = HelpViewModel(help_manager)
        metadata = viewmodel.metadata
        assert metadata['app'] == "TestApp"

    def test_reload_success(self, help_manager, qapp):
        """Test successful reload."""
        viewmodel = HelpViewModel(help_manager)
        result = viewmodel.reload()
        assert result is True

    def test_empty_key_handling(self, help_manager, qapp):
        """Test handling of empty/None keys."""
        viewmodel = HelpViewModel(help_manager)
        assert viewmodel.get_tooltip("") == ""
        assert viewmodel.get_title("") == ""
        assert viewmodel.get_body("") == ""


class TestHelpWidgets:
    """Tests for help widgets (InfoButton and HelpPopup)."""

    def test_info_button_creation(self, help_manager, qapp):
        """Test creating an InfoButton."""
        viewmodel = HelpViewModel(help_manager)
        button = InfoButton("test.key", viewmodel)
        
        assert button is not None
        assert button.text() == "ⓘ"
        assert button.toolTip() == "Test tooltip"

    def test_info_button_set_help_key(self, help_manager, qapp):
        """Test changing the help key on InfoButton."""
        viewmodel = HelpViewModel(help_manager)
        button = InfoButton("test.key", viewmodel)
        
        button.set_help_key("test.noalias")
        assert button.toolTip() == "Short text"

    def test_help_popup_creation(self, help_manager, qapp):
        """Test creating a HelpPopup."""
        viewmodel = HelpViewModel(help_manager)
        popup = HelpPopup("test.key", viewmodel)
        
        assert popup is not None
        assert popup._title_label.text() == "Test Title"

    def test_help_popup_set_help_key(self, help_manager, qapp):
        """Test changing help key on HelpPopup."""
        viewmodel = HelpViewModel(help_manager)
        popup = HelpPopup("test.key", viewmodel)

        popup.set_help_key("test.noalias")
        assert popup._title_label.text() == "No Alias Test"

    def test_help_prompt_progression(self, help_manager, qapp):
        """Test AI prompt builder progresses for a help topic."""
        viewmodel = HelpViewModel(help_manager)

        prompt1, count1 = viewmodel.build_ai_prompt("test.key")
        prompt2, count2 = viewmodel.build_ai_prompt("test.key")

        assert count1 == 1
        assert count2 == 2
        assert "concise overview" in prompt1
        assert "Add more detail" in prompt2
        assert "Help key: test.key" in prompt1

    def test_attach_help(self, qapp):
        """Test attaching help key to a widget."""
        widget = QWidget()
        attach_help(widget, "test.key")
        
        help_key = get_help_key(widget)
        assert help_key == "test.key"

    def test_get_help_key_not_set(self, qapp):
        """Test getting help key from widget without one."""
        widget = QWidget()
        help_key = get_help_key(widget)
        assert help_key is None


@pytest.fixture
def reset_singletons():
    """Reset singleton instances before and after each test."""
    # Reset before test
    import backend.help_manager
    import frontend.viewmodels.help_viewmodel
    backend.help_manager._shared_help_manager = None
    frontend.viewmodels.help_viewmodel._shared_help_viewmodel = None
    yield
    # Reset after test
    backend.help_manager._shared_help_manager = None
    frontend.viewmodels.help_viewmodel._shared_help_viewmodel = None


class TestSingletons:
    """Test singleton getters."""

    def test_get_help_manager_singleton(self, yaml_help_file, reset_singletons):
        """Test that get_help_manager returns the same instance."""
        manager1 = get_help_manager(yaml_help_file)
        manager2 = get_help_manager()
        
        assert manager1 is manager2

    def test_get_help_viewmodel_singleton(self, help_manager, qapp, reset_singletons):
        """Test that get_help_viewmodel returns the same instance."""
        viewmodel1 = get_help_viewmodel(help_manager)
        viewmodel2 = get_help_viewmodel()
        
        assert viewmodel1 is viewmodel2

    def test_get_help_viewmodel_no_manager_error(self, qapp, reset_singletons):
        """Test that get_help_viewmodel raises error without manager on first call."""
        with pytest.raises(ValueError):
            get_help_viewmodel()


class TestWildcardPatterns:
    """Test wildcard pattern matching in detail."""

    def test_multiple_wildcard_matches(self, help_manager):
        """Test that multiple feature keys all match the wildcard."""
        keys_to_test = [
            "feature:column1",
            "feature:column2",
            "feature:AE_RMS",
            "feature:Temperature",
            "feature:some_long_name_123"
        ]
        
        for key in keys_to_test:
            entry = help_manager.get_entry(key)
            assert entry is not None
            assert entry['title'] == "Wildcard Feature"

    def test_wildcard_no_match_different_prefix(self, help_manager):
        """Test that keys with different prefix don't match."""
        entry = help_manager.get_entry("other:column")
        assert entry is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
