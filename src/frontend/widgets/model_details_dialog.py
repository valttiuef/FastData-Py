from __future__ import annotations
from ..localization import tr

from typing import Callable


from .ask_ai_dialog import AskAiDialog



class ModelDetailsDialog(AskAiDialog):
    """Popup dialog showing model run summary with editable text."""

    def __init__(
        self,
        *,
        summary_text: str,
        on_ask_ai: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__(
            title=tr("Model run details"),
            header_text=tr("Review the selected model runs. Add notes before asking the AI if you like."),
            summary_text=summary_text,
            on_ask_ai=on_ask_ai,
            minimum_size=(520, 360),
            parent=parent,
        )
