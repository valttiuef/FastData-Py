
from __future__ import annotations
import html
import json
import re
from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (

    QAbstractItemView,
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from ..localization import tr

from ..models.log_model import LogEvent, LogModel
from ..models.settings_model import SettingsModel
from ..viewmodels.log_view_model import LogViewModel
from ..viewmodels.help_viewmodel import HelpViewModel
from ..widgets.collapsible_section import CollapsibleSection
from ..widgets.help_widgets import InfoButton



@dataclass
class _ChatMessage:
    role: str  # "user" | "assistant" | "error"
    content: str


class ChatWindow(QWidget):
    """Standalone ChatGPT-style chat window, mirroring chat/llm logs with formatting."""

    closed = Signal()

    def __init__(
        self,
        log_model: LogModel,
        log_view_model: LogViewModel,
        settings_model: SettingsModel,
        help_viewmodel: Optional[HelpViewModel] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._log_model = log_model
        self._log_view_model = log_view_model
        self._settings_model = settings_model
        self._help_viewmodel = help_viewmodel
        self._stored_api_key: Optional[str] = None

        self._messages: list[_ChatMessage] = []
        self._streaming_index: Optional[int] = None
        self._saw_llm_log_for_stream = False
        self._suppress_next_llm_log: Optional[str] = None
        self._suppress_next_llm_error_log: Optional[str] = None
        self._pending_stream_append: list[str] = []
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._flush_stream_updates)

        self.setObjectName("chatWindow")
        self.setWindowTitle(tr("Chat"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(840, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        settings_section = CollapsibleSection(tr("Settings"), collapsed=True, parent=self)
        settings_layout = settings_section.body_layout()
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        settings_layout.addLayout(grid)

        provider_label = QLabel(tr("LLM Provider:"), self)
        self._provider_selector = QComboBox(self)
        self._provider_selector.addItem(tr("ChatGPT (OpenAI)"), "openai")
        self._provider_selector.addItem(tr("Ollama (Local)"), "ollama")
        self._provider_selector.currentIndexChanged.connect(self._on_provider_changed)
        grid.addWidget(provider_label, 0, 0)
        grid.addWidget(self._with_info(self._provider_selector, "chat.settings.provider"), 0, 1)

        model_label = QLabel(tr("Model name:"), self)
        self._model_input = QLineEdit(self)
        self._model_input.setPlaceholderText(tr("Model identifier (e.g. gpt-4o-mini, llama3.2)"))
        self._model_input.editingFinished.connect(self._persist_model_choice)
        grid.addWidget(model_label, 1, 0)
        grid.addWidget(self._with_info(self._model_input, "chat.settings.model_name"), 1, 1)

        self._api_label = QLabel(tr("API Key:"), self)
        self._api_key_row = QWidget(self)
        api_layout = QHBoxLayout(self._api_key_row)
        api_layout.setContentsMargins(0, 0, 0, 0)
        api_layout.setSpacing(8)
        self._api_key_input = QLineEdit(self)
        self._api_key_input.setPlaceholderText(tr("OpenAI API key"))
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.editingFinished.connect(self._persist_api_key)
        api_layout.addWidget(self._api_key_input, stretch=1)
        self._clear_api_key_button = QPushButton(tr("Clear"), self)
        self._clear_api_key_button.setToolTip(tr("Clear stored API key"))
        self._clear_api_key_button.clicked.connect(self._confirm_clear_api_key)
        api_layout.addWidget(self._clear_api_key_button)
        self._api_key_row_container = self._with_info(self._api_key_row, "chat.settings.api_key")
        grid.addWidget(self._api_label, 2, 0)
        grid.addWidget(self._api_key_row_container, 2, 1)

        labels = [provider_label, model_label, self._api_label]
        max_width = max(label.sizeHint().width() for label in labels)
        grid.setColumnMinimumWidth(0, max_width)
        for label in labels:
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        clear_row = QHBoxLayout()
        clear_row.setContentsMargins(0, 0, 0, 0)
        clear_row.setSpacing(6)
        clear_row.addStretch(1)
        self._clear_chats_button = QPushButton(tr("Clear chats"), self)
        self._clear_chats_button.clicked.connect(self._confirm_clear_chats)
        clear_row.addWidget(self._clear_chats_button)
        settings_layout.addLayout(clear_row)

        layout.addWidget(settings_section)

        self._chat_list = QListWidget(self)
        self._chat_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._chat_list.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._chat_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._chat_list.setSpacing(10)
        # Configure smooth scrollbar behavior: smaller steps for less jumping
        scrollbar = self._chat_list.verticalScrollBar()
        scrollbar.setSingleStep(10)  # Smooth wheel scrolling (10 pixels per scroll event)
        scrollbar.setPageStep(50)   # Moderate jumps when clicking scrollbar buttons
        self._chat_list.viewport().installEventFilter(self)
        layout.addWidget(self._chat_list, stretch=1)

        input_row = QWidget(self)
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self._input = QLineEdit(self)
        self._input.setPlaceholderText(tr("Ask anything…"))
        self._input.returnPressed.connect(self._send_message)
        input_layout.addWidget(self._input, stretch=1)

        self._send_button = QPushButton(tr("Send"), self)
        self._send_button.clicked.connect(self._send_message)
        input_layout.addWidget(self._send_button)

        layout.addWidget(input_row)

        self._log_model.entry_added.connect(self._on_log_entry_added)
        self._log_model.cleared.connect(self._on_log_cleared)

        self._log_view_model.llm_response_started.connect(self._on_llm_response_started)
        self._log_view_model.llm_token_received.connect(self._on_llm_token_received)
        self._log_view_model.llm_response_finished.connect(self._on_llm_response_finished)
        self._log_view_model.llm_error.connect(self._on_llm_error)

        self._restore_llm_settings()
        self._rebuild_from_log()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self._chat_list.viewport() and event.type() == QEvent.Type.Resize:
            self._relayout_chat_items()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Behave like "close to hide" so it can be reopened from the menu.
        self.closed.emit()
        self.hide()
        event.ignore()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._chat_list.count() > 0:
            QTimer.singleShot(0, self._relayout_chat_items)
            QTimer.singleShot(50, self._relayout_chat_items)

    # ------------------------------------------------------------------
    def _with_info(self, widget: QWidget, help_key: str) -> QWidget:
        if self._help_viewmodel is None:
            return widget

        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        size_policy = widget.sizePolicy()
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        widget.setSizePolicy(size_policy)
        layout.addWidget(widget, 1)
        layout.addWidget(InfoButton(help_key, self._help_viewmodel, parent=container))
        return container

    # ------------------------------------------------------------------
    def _restore_llm_settings(self) -> None:
        provider = self._settings_model.llm_provider or "openai"
        provider_index = self._provider_selector.findData(provider)
        if provider_index >= 0:
            self._provider_selector.setCurrentIndex(provider_index)
        else:
            self._provider_selector.setCurrentIndex(0)

        self._stored_api_key = self._settings_model.openai_api_key or None
        self._api_key_input.setText(self._stored_api_key or "")

        model = self._settings_model.llm_model(provider)
        self._model_input.setText(model)

        self._log_view_model.set_provider(provider)
        self._log_view_model.set_api_key(self._stored_api_key)
        self._log_view_model.set_model_name(model or None)
        self._on_provider_changed(self._provider_selector.currentIndex())

    def _persist_model_choice(self) -> None:
        provider = self._provider_selector.currentData(Qt.ItemDataRole.UserRole)
        model_name = self._model_input.text().strip()
        if not model_name:
            self._log_view_model.set_model_name(None)
            return
        self._settings_model.set_llm_model(model_name, provider)
        self._log_view_model.set_model_name(model_name)

    def _persist_api_key(self) -> None:
        value = self._api_key_input.text().strip()
        if not value:
            return
        self._stored_api_key = value
        self._settings_model.set_openai_api_key(value)
        self._log_view_model.set_api_key(value)

    def _confirm_clear_api_key(self) -> None:
        reply = QMessageBox.question(
            self,
            tr("Clear API key"),
            tr("This will remove the stored OpenAI API key. Continue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._api_key_input.clear()
        self._stored_api_key = None
        self._settings_model.set_openai_api_key("")
        self._log_view_model.set_api_key(None)

    def _confirm_clear_chats(self) -> None:
        reply = QMessageBox.question(
            self,
            tr("Clear chat history"),
            tr("This will clear all chat history from the current database. This cannot be undone. Continue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._log_view_model.clear_chat_history()

    def _on_provider_changed(self, _index: int) -> None:
        provider = self._provider_selector.currentData(Qt.ItemDataRole.UserRole)
        is_openai = provider == "openai"
        self._api_label.setVisible(is_openai)
        self._api_key_row_container.setVisible(is_openai)

        self._settings_model.set_llm_provider(provider)
        self._log_view_model.set_provider(provider)

        if is_openai:
            self._model_input.setPlaceholderText(tr("OpenAI model (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)"))
        else:
            self._model_input.setPlaceholderText(tr("Local model (llama3.2, mistral, codellama)"))

        default_model = self._settings_model.llm_model(provider)
        self._model_input.setText(default_model)
        self._log_view_model.set_model_name(default_model or None)

        if is_openai:
            api_value = self._api_key_input.text().strip() or self._stored_api_key or ""
            self._api_key_input.setText(api_value)
            self._log_view_model.set_api_key(api_value or None)
        else:
            self._log_view_model.set_api_key(None)

    # ------------------------------------------------------------------
    def _send_message(self) -> None:
        text = self._input.text().strip()
        if not text:
            return

        provider = self._provider_selector.currentData(Qt.ItemDataRole.UserRole)
        self._log_view_model.set_provider(provider)

        model_name = self._model_input.text().strip() or self._settings_model.llm_model(provider)
        self._log_view_model.set_model_name(model_name or None)
        if model_name:
            self._settings_model.set_llm_model(model_name, provider)

        api_key = self._api_key_input.text().strip() or self._stored_api_key or None
        self._stored_api_key = api_key
        if provider == "openai":
            if api_key:
                self._settings_model.set_openai_api_key(api_key)
            self._log_view_model.set_api_key(api_key)
        else:
            self._log_view_model.set_api_key(None)

        self._set_llm_busy(True)
        self._log_view_model.ask_llm(text)
        self._input.clear()

    # ------------------------------------------------------------------
    def _on_llm_response_started(self) -> None:
        self._saw_llm_log_for_stream = False
        self._streaming_index = len(self._messages)
        self._append_message("assistant", "…")

    def _on_llm_token_received(self, token: str) -> None:
        if self._streaming_index is None:
            return
        self._pending_stream_append.append(token)
        if not self._render_timer.isActive():
            self._render_timer.start(120)

    def _flush_stream_updates(self) -> None:
        if self._streaming_index is None or not self._pending_stream_append:
            return
        idx = self._streaming_index
        if idx < 0 or idx >= len(self._messages):
            return
        content = self._messages[idx].content
        if content.strip() == "…":
            content = ""
        content += "".join(self._pending_stream_append)
        self._pending_stream_append.clear()
        self._messages[idx].content = content
        self._update_message_item(idx)

    def _on_llm_response_finished(self, text: str) -> None:
        self._flush_stream_updates()
        if self._streaming_index is not None and 0 <= self._streaming_index < len(self._messages):
            self._messages[self._streaming_index].content = text or ""
            if not self._saw_llm_log_for_stream:
                self._suppress_next_llm_log = (text or "").strip()
            self._update_message_item(self._streaming_index)
        self._streaming_index = None
        self._saw_llm_log_for_stream = False
        self._pending_stream_append.clear()
        self._set_llm_busy(False)

    def _on_llm_error(self, message: str) -> None:
        self._streaming_index = None
        self._saw_llm_log_for_stream = False
        self._pending_stream_append.clear()
        self._set_llm_busy(False)
        if message:
            self._append_message("error", message)
            self._suppress_next_llm_error_log = message.strip()

    # ------------------------------------------------------------------
    def _on_log_cleared(self) -> None:
        self._rebuild_from_log()

    def _on_log_entry_added(self, event: LogEvent) -> None:
        if event.origin == "chat":
            self._append_message("user", event.message)
            return

        if event.origin != "llm":
            return

        message = (event.message or "").strip()
        if event.level >= 40:
            if self._suppress_next_llm_error_log and message == self._suppress_next_llm_error_log:
                self._suppress_next_llm_error_log = None
                return
            self._append_message("error", event.message)
            return

        if self._suppress_next_llm_log and message == self._suppress_next_llm_log:
            self._suppress_next_llm_log = None
            return

        if self._streaming_index is not None and 0 <= self._streaming_index < len(self._messages):
            self._messages[self._streaming_index].content = event.message
            self._saw_llm_log_for_stream = True
            self._update_message_item(self._streaming_index)
            return

        self._append_message("assistant", event.message)

    def _rebuild_from_log(self) -> None:
        self._messages = []
        self._streaming_index = None
        self._saw_llm_log_for_stream = False
        self._suppress_next_llm_log = None
        self._suppress_next_llm_error_log = None
        self._pending_stream_append.clear()

        for entry in self._log_model.entries():
            if entry.origin == "chat":
                self._messages.append(_ChatMessage(role="user", content=entry.message))
            elif entry.origin == "llm":
                role = "error" if entry.level >= 40 else "assistant"
                self._messages.append(_ChatMessage(role=role, content=entry.message))

        self._rebuild_chat_list()

    def _append_message(self, role: str, content: str) -> None:
        self._messages.append(_ChatMessage(role=role, content=content))
        self._append_message_item(len(self._messages) - 1)

    def _set_llm_busy(self, value: bool) -> None:
        self._send_button.setDisabled(value)
        self._provider_selector.setDisabled(value)
        self._model_input.setDisabled(value)
        self._api_key_input.setDisabled(value)
        self._clear_api_key_button.setDisabled(value)
        self._clear_chats_button.setDisabled(value)

    # ------------------------------------------------------------------
    def _chat_is_at_bottom(self) -> bool:
        sb = self._chat_list.verticalScrollBar()
        return sb.value() >= sb.maximum() - 2

    def _scroll_chat_to_bottom(self) -> None:
        self._chat_list.scrollToBottom()

    def _rebuild_chat_list(self) -> None:
        self._chat_list.setUpdatesEnabled(False)
        try:
            self._chat_list.clear()
            for idx in range(len(self._messages)):
                self._append_message_item(idx, preserve_scroll=False)
        finally:
            self._chat_list.setUpdatesEnabled(True)
        self._scroll_chat_to_bottom()

    def _append_message_item(self, idx: int, *, preserve_scroll: bool = True) -> None:
        if idx < 0 or idx >= len(self._messages):
            return

        at_bottom = self._chat_is_at_bottom() if preserve_scroll else False
        available_width = self._chat_list.viewport().width()
        msg = self._messages[idx]

        item = QListWidgetItem(self._chat_list)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)

        widget = _ChatBubbleWidget(
            role=msg.role,
            content_css=self._content_css(),
            parent=self._chat_list,
        )
        widget.set_available_width(available_width)
        widget.set_message(msg.role, msg.content, self._format_assistant_content)
        widget.adjustSize()

        self._chat_list.setItemWidget(item, widget)
        item.setSizeHint(widget.sizeHint())

        if at_bottom:
            self._scroll_chat_to_bottom()

    def _update_message_item(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._messages):
            return
        if idx >= self._chat_list.count():
            self._rebuild_chat_list()
            return

        item = self._chat_list.item(idx)
        widget = self._chat_list.itemWidget(item)
        if not isinstance(widget, _ChatBubbleWidget):
            self._rebuild_chat_list()
            return

        at_bottom = self._chat_is_at_bottom()
        self._chat_list.setUpdatesEnabled(False)
        try:
            widget.set_available_width(self._chat_list.viewport().width())
            msg = self._messages[idx]
            widget.set_message(msg.role, msg.content, self._format_assistant_content)
            widget.adjustSize()
            item.setSizeHint(widget.sizeHint())
            self._chat_list.doItemsLayout()
        finally:
            self._chat_list.setUpdatesEnabled(True)
        if at_bottom:
            self._scroll_chat_to_bottom()

    def _relayout_chat_items(self) -> None:
        if self._chat_list.count() == 0:
            return
        available_width = self._chat_list.viewport().width()
        self._chat_list.setUpdatesEnabled(False)
        try:
            for idx in range(self._chat_list.count()):
                item = self._chat_list.item(idx)
                widget = self._chat_list.itemWidget(item)
                if not isinstance(widget, _ChatBubbleWidget):
                    continue
                widget.set_available_width(available_width)
                widget.adjustSize()
                item.setSizeHint(widget.sizeHint())
            self._chat_list.doItemsLayout()
        finally:
            self._chat_list.setUpdatesEnabled(True)

    @staticmethod
    def _format_plain_text(text: str) -> str:
        escaped = html.escape(text)
        return escaped.replace("\n", "<br>")

    def _format_assistant_content(self, text: str) -> str:
        raw = text.strip("\n")
        if not raw:
            return ""

        json_html = self._try_format_as_json(raw)
        if json_html is not None:
            return json_html

        return self._format_markdownish(raw)

    def _format_markdownish(self, text: str) -> str:
        """Render a small markdown-like subset (headings, lists, emphasis, code)."""

        lines = text.splitlines()
        blocks: list[str] = []
        paragraph_lines: list[str] = []
        list_kind: Optional[str] = None  # "ul" | "ol"
        list_items: list[str] = []
        list_values: list[Optional[int]] = []
        in_code = False
        code_lang = ""
        code_lines: list[str] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if not paragraph_lines:
                return
            joined = "\n".join(paragraph_lines).strip("\n")
            if joined:
                blocks.append(f"<p>{self._inline_format(joined)}</p>")
            paragraph_lines = []

        def flush_list() -> None:
            nonlocal list_kind, list_items, list_values
            if not list_kind or not list_items:
                list_kind = None
                list_items = []
                list_values = []
                return

            if list_kind == "ol":
                numeric_values = [value for value in list_values if value is not None]
                use_explicit_values = not (numeric_values and all(value == 1 for value in numeric_values))
                rendered_items: list[str] = []
                for value, item in zip(list_values, list_items, strict=True):
                    if use_explicit_values and value is not None:
                        rendered_items.append(f"<li value='{value}'>{self._inline_format(item)}</li>")
                    else:
                        rendered_items.append(f"<li>{self._inline_format(item)}</li>")
                items_html = "".join(rendered_items)
            else:
                items_html = "".join(f"<li>{self._inline_format(item)}</li>" for item in list_items)

            blocks.append(f"<{list_kind}>{items_html}</{list_kind}>")
            list_kind = None
            list_items = []
            list_values = []

        def flush_code() -> None:
            nonlocal code_lines, code_lang
            if not code_lines:
                return
            code = "\n".join(code_lines)
            lang_class = f" language-{html.escape(code_lang)}" if code_lang else ""
            blocks.append(f"<pre><code class='{lang_class.strip()}'>{html.escape(code)}</code></pre>")
            code_lines = []
            code_lang = ""

        for line in lines:
            fence = line.strip()
            if fence.startswith("```"):
                if in_code:
                    flush_code()
                    in_code = False
                    continue
                flush_paragraph()
                flush_list()
                in_code = True
                code_lang = fence[3:].strip()
                continue

            if in_code:
                code_lines.append(line)
                continue

            if not line.strip():
                flush_paragraph()
                if list_kind and list_items:
                    # Keep ordered/unordered lists across blank lines like markdown does.
                    list_items[-1] = f"{list_items[-1]}\n"
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                flush_paragraph()
                flush_list()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                blocks.append(f"<h{level}>{self._inline_format(title)}</h{level}>")
                continue

            ul_match = re.match(r"^\s*[-*]\s+(.+)$", line)
            ol_match = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
            if ul_match or ol_match:
                flush_paragraph()
                next_kind = "ul" if ul_match else "ol"
                if list_kind and list_kind != next_kind:
                    flush_list()
                list_kind = next_kind
                item_text = (ul_match.group(1) if ul_match else ol_match.group(2)).strip()  # type: ignore[union-attr]
                list_items.append(item_text)
                list_values.append(int(ol_match.group(1)) if ol_match else None)  # type: ignore[union-attr]
                continue

            if list_kind:
                # Continue list item wrapping for indented/continuation lines.
                if line.startswith(" ") or line.startswith("\t"):
                    list_items[-1] = f"{list_items[-1]}\n{line.strip()}"
                    continue
                flush_list()

            paragraph_lines.append(line)

        if in_code:
            flush_code()
        flush_paragraph()
        flush_list()

        if not blocks:
            return ""
        return "".join(blocks)

    _INLINE_CODE_RE = re.compile(r"`([^`]+)`")
    _BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
    # Avoid lookbehind (some environments error on variable-width lookbehind).
    # Group 1 keeps the leading char when the asterisk isn't at the start.
    _ITALIC_RE = re.compile(r"(^|[^*])\*([^*]+)\*(?!\*)")

    def _inline_format(self, text: str) -> str:
        placeholders: list[str] = []

        def _code_sub(match: re.Match[str]) -> str:
            placeholders.append(html.escape(match.group(1)))
            return f"@@CODE{len(placeholders) - 1}@@"

        escaped = html.escape(text)
        escaped = self._INLINE_CODE_RE.sub(_code_sub, escaped)
        escaped = self._BOLD_RE.sub(r"<strong>\1</strong>", escaped)
        escaped = self._ITALIC_RE.sub(r"\1<em>\2</em>", escaped)

        for idx, code in enumerate(placeholders):
            escaped = escaped.replace(f"@@CODE{idx}@@", f"<code class='inline'>{code}</code>")
        escaped = escaped.replace("\n", "<br>")
        return escaped

    @staticmethod
    def _try_format_as_json(text: str) -> Optional[str]:
        candidate = text.strip()
        if not candidate or candidate[0] not in "[{":
            return None
        try:
            obj = json.loads(candidate)
        except Exception:
            return None
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        return f"<pre><code class='language-json'>{html.escape(pretty)}</code></pre>"

    @staticmethod
    def _content_css() -> str:
        return """
        body {
            margin: 0;
            padding: 0;
        }
        h1, h2, h3, h4, h5, h6 {
            margin: 0.2em 0 0.4em 0;
            line-height: 1.2;
        }
        h1 { font-size: 1.35em; }
        h2 { font-size: 1.25em; }
        h3 { font-size: 1.15em; }
        h4 { font-size: 1.05em; }
        h5, h6 { font-size: 1.0em; }
        ul, ol {
            margin: 0.3em 0 0.3em 1.25em;
            padding: 0;
        }
        li {
            margin: 0.18em 0;
        }
        pre {
            background: rgba(0, 0, 0, 0.25);
            padding: 10px 12px;
            border-radius: 12px;
            overflow-x: auto;
            white-space: pre;
        }
        code {
            font-family: Consolas, 'Cascadia Mono', 'Courier New', monospace;
            font-size: 0.95em;
        }
        code.inline {
            background: rgba(0, 0, 0, 0.18);
            padding: 0.05em 0.35em;
            border-radius: 8px;
        }
        p {
            margin: 0;
        }
        """


class _ChatBubbleWidget(QWidget):
    def __init__(self, role: str, content_css: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._role = role
        self._content_css = content_css
        self._available_width = 800
        self._raw_content = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._bubble = QFrame(self)
        self._bubble.setObjectName("chatBubble")
        self._bubble.setProperty("role", role)
        self._bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        bubble_layout = QVBoxLayout(self._bubble)
        bubble_layout.setContentsMargins(12, 9, 12, 9)
        bubble_layout.setSpacing(0)

        header = QWidget(self._bubble)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        header_layout.addStretch(1)
        self._copy_button = QPushButton(tr("Copy"), header)
        self._copy_button.setObjectName("chatBubbleCopyButton")
        self._copy_button.setToolTip(tr("Copy message to clipboard"))
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.setFixedHeight(18)
        self._copy_button.clicked.connect(self._copy_to_clipboard)
        header_layout.addWidget(self._copy_button, 0, Qt.AlignmentFlag.AlignRight)
        bubble_layout.addWidget(header)

        self._text_label: Optional[QLabel] = None
        self._rich_view: Optional[QTextBrowser] = None

        if role == "assistant":
            view = QTextBrowser(self._bubble)
            view.setOpenExternalLinks(True)
            view.setReadOnly(True)
            view.setFrameStyle(QFrame.Shape.NoFrame)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            view.document().setDefaultStyleSheet(content_css)
            view.setStyleSheet("background: transparent;")
            self._rich_view = view
            bubble_layout.addWidget(view)
        else:
            label = QLabel(self._bubble)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            self._text_label = label
            bubble_layout.addWidget(label)

        self.setStyleSheet(
            """
            QFrame#chatBubble[role="user"] { background: #2563eb; color: #ffffff; border-radius: 14px; }
            QFrame#chatBubble[role="assistant"] { background: rgba(127, 127, 127, 0.18); border-radius: 14px; }
            QFrame#chatBubble[role="error"] { background: rgba(240, 98, 146, 0.20); color: #f06292; border-radius: 14px; }
            QPushButton#chatBubbleCopyButton {
                padding: 0 6px;
                border: none;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.12);
                color: rgba(255, 255, 255, 0.90);
                font-size: 11px;
            }
            QFrame#chatBubble[role=\"assistant\"] QPushButton#chatBubbleCopyButton {
                background: rgba(0, 0, 0, 0.18);
                color: rgba(255, 255, 255, 0.85);
            }
            QFrame#chatBubble[role=\"error\"] QPushButton#chatBubbleCopyButton {
                background: rgba(240, 98, 146, 0.25);
                color: rgba(240, 98, 146, 0.95);
            }
            QPushButton#chatBubbleCopyButton:hover { background: rgba(255, 255, 255, 0.18); }
            """
        )

        if role == "user":
            layout.addStretch(1)
            layout.addWidget(self._bubble, 0, Qt.AlignmentFlag.AlignRight)
        else:
            layout.addWidget(self._bubble, 0, Qt.AlignmentFlag.AlignLeft)
            layout.addStretch(1)

    def _reflow_rich_view_height(self) -> None:
        if self._rich_view is None:
            return
        doc = self._rich_view.document()
        doc.setTextWidth(self._rich_view.viewport().width())
        height = int(doc.size().height() + 2)
        self._rich_view.setFixedHeight(max(22, height))
        self._bubble.adjustSize()

    def _copy_to_clipboard(self) -> None:
        clipboard = QApplication.clipboard()
        if self._rich_view is not None:
            clipboard.setText(self._rich_view.document().toPlainText())
        elif self._text_label is not None:
            clipboard.setText(self._text_label.text())
        else:
            clipboard.setText(self._raw_content)

    def set_available_width(self, width: int) -> None:
        self._available_width = max(240, width)
        ratio = 0.60 if self._role == "user" else 0.78
        bubble_max = int(self._available_width * ratio)
        self._bubble.setMaximumWidth(bubble_max)
        if self._rich_view is not None:
            # 24 = left+right bubble layout margins.
            self._rich_view.setFixedWidth(max(120, bubble_max - 24))
            self._reflow_rich_view_height()
        if self._text_label is not None:
            self._text_label.setMaximumWidth(max(120, bubble_max - 24))
            self._text_label.adjustSize()

    def set_message(self, role: str, content: str, assistant_formatter: Callable[[str], str]) -> None:
        self._role = role
        self._raw_content = content
        self._bubble.setProperty("role", role)
        self._bubble.style().unpolish(self._bubble)
        self._bubble.style().polish(self._bubble)

        if self._rich_view is not None:
            html_body = assistant_formatter(content)
            html_doc = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"
            self._rich_view.setHtml(html_doc)
            self._reflow_rich_view_height()
            self._bubble.adjustSize()
            return

        if self._text_label is not None:
            self._text_label.setText(content)
            self._text_label.adjustSize()
            self._bubble.adjustSize()
