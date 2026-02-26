from __future__ import annotations
from ...localization import tr

from typing import Callable


from ...widgets.ask_ai_dialog import AskAiDialog



class SomDetailsDialog(AskAiDialog):
    """Popup dialog showing SOM summary with editable text."""

    def __init__(
        self,
        *,
        summary_text: str,
        on_ask_ai: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__(
            title=tr("SOM details"),
            header_text=tr("Review the map summary. Add notes before asking the AI if you like."),
            summary_text=summary_text,
            on_ask_ai=on_ask_ai,
            minimum_size=(480, 320),
            parent=parent,
        )
