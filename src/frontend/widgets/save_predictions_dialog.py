
from __future__ import annotations
from typing import Optional

from .save_output_dialog import SaveOutputDialog


class SavePredictionsDialog(SaveOutputDialog):
    def __init__(
        self,
        *,
        model_label: str,
        defaults: dict[str, object],
        parent: Optional["QWidget"] = None,
    ) -> None:
        super().__init__(
            mode="predictions",
            model_label=model_label,
            defaults=defaults,
            parent=parent,
        )


__all__ = ["SavePredictionsDialog"]
