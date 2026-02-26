from __future__ import annotations
from ...localization import tr

from typing import Callable


from ...widgets.ask_ai_dialog import AskAiDialog



class FeaturesInfoDialog(AskAiDialog):
    """Popup dialog showing feature summary with editable text."""

    def __init__(
        self,
        *,
        summary_text: str,
        on_ask_ai: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__(
            title=tr("Feature details"),
            header_text=tr("Review the current features and stats. Add notes before asking the AI if you like."),
            summary_text=summary_text,
            on_ask_ai=on_ask_ai,
            minimum_size=(420, 260),
            parent=parent,
        )
