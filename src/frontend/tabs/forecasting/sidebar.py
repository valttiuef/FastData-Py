import logging
import re
from typing import Callable, Optional, TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (

    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)
from ...localization import tr

if TYPE_CHECKING:
    from .viewmodel import ForecastingViewModel

from ...widgets.collapsible_section import CollapsibleSection
from ...widgets.data_selector_widget import DataSelectorWidget
from ...widgets.multi_check_combo import MultiCheckCombo
from ...widgets.sidebar_widget import SidebarWidget
from ...widgets.help_widgets import InfoButton
from ...viewmodels.help_viewmodel import HelpViewModel, get_help_viewmodel
from ...param_specs.forecasting import FORECASTING_MODEL_PARAM_SPECS
from ...utils import toast_error



class ForecastingSidebar(SidebarWidget):
    """Sidebar housing feature selection and forecasting options."""

    def __init__(
        self,
        *,
        view_model: "ForecastingViewModel",
        parent: Optional[QWidget] = None,
        help_viewmodel: Optional[HelpViewModel] = None,
    ) -> None:
        super().__init__(title=tr("Forecasting"), parent=parent)

        self._view_model = view_model
        resolved_help = help_viewmodel
        if resolved_help is None:
            try:
                resolved_help = get_help_viewmodel()
            except Exception:
                resolved_help = None
        self._help_viewmodel = resolved_help
        if self._view_model is None:
            logger.warning("ForecastingSidebar initialised without view_model.")

        self._model_items: list[tuple[str, str, dict[str, object]]] = []
        self._model_param_getters: dict[str, dict[str, Callable[[], object]]] = {}
        self._feature_payload_lookup: dict[int, dict] = {}
        self._selected_target_id: Optional[int] = None

        controls_layout = self.content_layout()

        actions_group = QGroupBox(tr("Actions"), self)
        actions_layout = QVBoxLayout(actions_group)
        self.run_button = QPushButton(tr("Run forecasting"), actions_group)
        self.auto_save_checkbox = QCheckBox(tr("Auto-save models to database"), actions_group)
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(self.auto_save_checkbox)
        self.set_sticky_actions(actions_group)

        self.data_selector = DataSelectorWidget(
            parent=self, data_model=view_model.data_model, help_viewmodel=self._help_viewmodel
        )
        controls_layout.addWidget(self.data_selector, 1)

        self.features_widget = self.data_selector.features_widget

        params_group = QGroupBox(tr("Forecast parameters"), self)
        params_form = QFormLayout(params_group)
        self.horizon = QSpinBox(params_group)
        self.horizon.setRange(1, 10_000)
        self.horizon.setValue(24)
        params_form.addRow(
            tr("Horizon (steps):"),
            self._wrap_with_info(self.horizon, "controls.forecasting.horizon"),
        )

        # Window strategy selection
        self.window_strategy_combo = QComboBox(params_group)
        self.window_strategy_combo.addItem(tr("Single Train/Test Split"), "single")
        self.window_strategy_combo.addItem(tr("Sliding Window"), "sliding")
        self.window_strategy_combo.addItem(tr("Expanding Window"), "expanding")
        self.window_strategy_combo.setCurrentIndex(1)  # Default to sliding
        params_form.addRow(
            tr("Window strategy:"),
            self._wrap_with_info(
                self.window_strategy_combo, "controls.forecasting.window_strategy"
            ),
        )

        # Initial window size (for sliding/expanding)
        self.initial_window = QSpinBox(params_group)
        self.initial_window.setRange(-1, 100_000)
        self.initial_window.setValue(-1)  # -1 means auto-calculate
        self.initial_window.setSpecialValueText(tr("Auto"))
        params_form.addRow(
            tr("Window size:"),
            self._wrap_with_info(
                self.initial_window, "controls.forecasting.initial_window"
            ),
        )
        
        controls_layout.addWidget(params_group)

        models_group = QGroupBox(tr("Models"), self)
        models_layout = QVBoxLayout(models_group)
        self.model_combo = MultiCheckCombo(models_group, placeholder=tr("Select models"), summary_max=3)
        models_layout.addWidget(
            self._wrap_with_info(self.model_combo, "controls.forecasting.models")
        )
        controls_layout.addWidget(models_group)

        # Optional target feature for time series regression models
        target_group = QGroupBox(tr("Target feature (optional)"), self)
        target_layout = QVBoxLayout(target_group)
        target_placeholder = tr("None (univariate)")
        self.target_combo = MultiCheckCombo(target_group, placeholder=target_placeholder, summary_max=1)
        self.target_combo.set_max_checked(1)
        self.target_combo.set_summary_formatter(
            lambda labels, checked, total, placeholder=target_placeholder: labels[0] if labels else placeholder
        )
        target_layout.addWidget(
            self._wrap_with_info(self.target_combo, "controls.forecasting.target")
        )
        controls_layout.addWidget(target_group)

        self.hyperparams_section = CollapsibleSection(tr("Hyperparameters"), collapsed=True, parent=self)
        hyperparams_layout = self.hyperparams_section.body_layout()
        self.model_params_group = QGroupBox(tr("Models"), self.hyperparams_section)
        self.model_params_layout = QVBoxLayout(self.model_params_group)
        hyperparams_layout.addWidget(self.model_params_group)
        controls_layout.addWidget(self.hyperparams_section)

        self.run_button.clicked.connect(self._start_run)
        self.model_combo.selection_changed.connect(self._build_model_param_forms)
        self.target_combo.selection_changed.connect(self._on_target_selection_changed)
        self._wire_signals()

        # Set up features widget connection to update target combo
        try:
            self.features_widget._view_model.features_loaded.connect(
                lambda *_args: self.refresh_target_list()
            )
        except AttributeError:
            # features_widget may not have _view_model attribute in some configurations
            logger.warning("Exception in __init__", exc_info=True)
        self.refresh_target_list()

    def set_models(self, items: list[tuple[str, str, dict[str, object]]]):
        self._model_items = list(items)
        combo_items = [(label, key) for key, label, _defaults in items]
        self.model_combo.set_items(combo_items, check_all=False)
        default_key = "linear_regression" if any(key == "linear_regression" for key, _, _ in items) else (items[0][0] if items else None)
        self.model_combo.set_selected_values([default_key] if default_key else [])
        self._build_model_param_forms()

    def selected_model_keys(self) -> list[str]:
        return [str(v) for v in self.model_combo.selected_values()]

    def selected_payloads(self) -> list[dict]:
        payloads = self.features_widget.selected_payloads()
        self._feature_payload_lookup = {int(p.get("feature_id")): p for p in payloads if p.get("feature_id")}
        return payloads

    def forecast_horizon(self) -> int:
        return int(self.horizon.value())
    
    def window_strategy(self) -> str:
        """Get the selected window strategy: 'single', 'sliding', or 'expanding'."""
        return str(self.window_strategy_combo.currentData())
    
    def initial_window_size(self) -> Optional[int]:
        """Get the initial window size, or None if set to auto (-1)."""
        val = int(self.initial_window.value())
        return None if val < 0 else val

    def target_payload(self) -> Optional[dict]:
        """Get the selected target feature payload, or None if not set."""
        if self._selected_target_id is None:
            return None
        return self._feature_payload_lookup.get(self._selected_target_id)

    def refresh_target_list(self) -> None:
        """Refresh the target feature dropdown from available features."""
        self._feature_payload_lookup = {}
        items: list[tuple[str, int]] = []
        for payload in self.features_widget.all_payloads():
            if not isinstance(payload, dict):
                continue
            fid = self._payload_id(payload)
            if fid is None:
                continue
            self._feature_payload_lookup[fid] = payload
            items.append((self._payload_label(payload), fid))

        block = self.target_combo.blockSignals(True)
        try:
            self.target_combo.set_items(items, check_all=False)
            if self._selected_target_id is not None and self._selected_target_id in self._feature_payload_lookup:
                self.target_combo.set_selected_values([self._selected_target_id])
            else:
                self._selected_target_id = None
                self.target_combo.set_selected_values([])
        finally:
            self.target_combo.blockSignals(block)

    def _payload_id(self, payload: Optional[dict]) -> Optional[int]:
        """Extract the feature_id from a payload dictionary."""
        if not isinstance(payload, dict):
            return None
        fid = payload.get("feature_id")
        try:
            return int(fid) if fid is not None else None
        except (ValueError, TypeError):
            return None

    def _payload_label(self, payload: dict) -> str:
        """Generate a display label for a payload."""
        label = payload.get("label")
        feature_type = payload.get("type")
        parts = [p for p in (label, feature_type) if p]
        text = " · ".join(parts)
        fid = payload.get("feature_id")
        return text or f"Feature {fid}"

    def _on_target_selection_changed(self) -> None:
        """Handle target feature selection changes."""
        values = [v for v in self.target_combo.selected_values() if isinstance(v, int)]
        self._selected_target_id = values[0] if values else None

    def preprocessing_parameters(self) -> dict[str, object]:
        return self.data_selector.preprocessing_widget.parameters()

    def model_parameters(self) -> dict[str, dict[str, object]]:
        params: dict[str, dict[str, object]] = {}
        for key, getters in self._model_param_getters.items():
            values: dict[str, object] = {}
            for name, getter in getters.items():
                try:
                    value = getter()
                    # Filter out None values to avoid passing unsupported parameters
                    if value is not None:
                        values[name] = value
                except Exception:
                    continue
            if values:
                params[key] = values
        return params

    def _start_run(self) -> None:
        if self._view_model.is_running():
            return

        features = self.selected_payloads()
        models = self.selected_model_keys()
        if not features or not models:
            return

        model_params = self.model_parameters()
        data_filters = self.data_selector.build_data_filters()
        if data_filters is None:
            return
        preprocessing = self.preprocessing_parameters()
        target_feature = self.target_payload()

        try:
            all_payloads = list(features)
            if isinstance(target_feature, dict) and target_feature not in all_payloads:
                all_payloads.append(target_feature)
            data_frame = self.data_selector.fetch_base_dataframe_for_features(
                all_payloads,
            )
            if data_frame is None:
                raise RuntimeError(tr("Failed to load data for selected inputs."))
            self._view_model.start_forecasts(
                data_frame=data_frame,
                features=features,
                models=models,
                data_filters=data_filters,
                model_params=model_params,
                forecast_horizon=self.forecast_horizon(),
                preprocessing=preprocessing,
                window_strategy=self.window_strategy(),
                initial_window=self.initial_window_size(),
                target_feature=target_feature,
            )
        except Exception as exc:
            logger = logging.getLogger(__name__)
            error_message = tr("Failed to start forecasting: {error}").format(error=exc)
            logger.error(error_message, exc_info=True)
            try:
                toast_error(str(error_message), title=tr("Forecasting failed"), tab_key="forecasting")
            except Exception:
                logger.warning("Exception in _start_run", exc_info=True)
            self._view_model.run_failed.emit(error_message)

    def auto_save_enabled(self) -> bool:
        return bool(self.auto_save_checkbox.isChecked())

    def _build_model_param_forms(self) -> None:
        self._clear_layout(self.model_params_layout)
        self._model_param_getters.clear()
        for key, label, defaults in self._model_items:
            if key not in self.model_combo.selected_values():
                continue
            specs = FORECASTING_MODEL_PARAM_SPECS.get(key)
            if not specs:
                continue
            box = QGroupBox(label, self.model_params_group)
            form = QFormLayout(box)
            getters: dict[str, Callable[[], object]] = {}
            for spec in specs:
                name = str(spec.get("name"))
                widget, getter = self._create_param_widget(spec, defaults.get(name), box)
                getters[name] = getter
                help_key = spec.get("help_key")
                label = tr(str(spec.get("label", name.replace("_", " ").title())))
                form.addRow(
                    label,
                    self._wrap_with_info(widget, str(help_key) if help_key else None),
                )
            self.model_params_layout.addWidget(box)
            self._model_param_getters[key] = getters
        self.model_params_layout.addStretch(1)

    def _create_param_widget(self, spec: dict[str, object], default: object, parent: QWidget) -> tuple[QWidget, Callable[[], object]]:
        ptype = str(spec.get("type", "text"))

        if ptype == "bool":
            widget = QCheckBox(parent)
            widget.setChecked(bool(default))
            getter = lambda w=widget: bool(w.isChecked())
            return widget, getter

        if ptype in {"int", "int_optional", "int_all"}:
            widget = QSpinBox(parent)
            minimum = int(spec.get("min", -1 if ptype != "int" else 0))
            maximum = int(spec.get("max", 1000))
            widget.setRange(minimum, maximum)
            widget.setSingleStep(int(spec.get("step", 1)))
            if ptype == "int_optional":
                widget.setSpecialValueText(tr("None"))
                if default is None:
                    widget.setValue(minimum)
                else:
                    widget.setValue(int(default))
                getter = lambda w=widget, min_val=minimum: None if w.value() == min_val else int(w.value())
            elif ptype == "int_all":
                widget.setSpecialValueText(tr("All"))
                if default in (None, "all"):
                    widget.setValue(minimum)
                else:
                    widget.setValue(int(default))
                getter = lambda w=widget, min_val=minimum: "all" if w.value() == min_val else int(w.value())
            else:
                widget.setValue(int(default) if default is not None else minimum)
                getter = lambda w=widget: int(w.value())
            return widget, getter

        if ptype in {"float", "float_optional"}:
            widget = QDoubleSpinBox(parent)
            widget.setRange(float(spec.get("min", 0.0)), float(spec.get("max", 10_000.0)))
            widget.setSingleStep(float(spec.get("step", 0.1)))
            widget.setDecimals(int(spec.get("decimals", 3)))
            if ptype == "float_optional":
                widget.setSpecialValueText(tr("None"))
                if default is None:
                    widget.setValue(widget.minimum())
                else:
                    widget.setValue(float(default))
                getter = lambda w=widget: None if w.value() == w.minimum() else float(w.value())
            else:
                widget.setValue(float(default) if default is not None else widget.minimum())
                getter = lambda w=widget: float(w.value())
            return widget, getter

        if ptype == "choice":
            widget = QComboBox(parent)
            for choice in spec.get("choices", []):
                widget.addItem(tr(str(choice)), choice)
            if default is not None:
                idx = widget.findData(default)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            getter = lambda w=widget: w.currentData()
            return widget, getter

        if ptype == "int_list":
            line = QLineEdit(parent)
            placeholder = spec.get("placeholder")
            if placeholder:
                line.setPlaceholderText(tr(str(placeholder)))
            if isinstance(default, (list, tuple)):
                line.setText(", ".join(str(v) for v in default))
            elif default is None:
                line.setText("")
            else:
                line.setText(str(default))

            def _parse(value: str) -> Optional[tuple[int, ...]]:
                parts = [p for p in re.split(r"[,\\s]+", value.strip()) if p]
                if not parts:
                    return None
                numbers: list[int] = []
                for part in parts:
                    try:
                        numbers.append(int(part))
                    except ValueError:
                        return None
                return tuple(numbers)

            getter = lambda w=line: _parse(w.text())
            return line, getter

        line = QLineEdit(parent)
        line.setText("" if default is None else str(default))
        getter = lambda w=line: (w.text().strip() or None)
        return line, getter

    def _wrap_with_info(self, widget: QWidget, help_key: Optional[str]) -> QWidget:
        if help_key and self._help_viewmodel is not None:
            container = QWidget(widget.parent() or self)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addWidget(widget, 1)
            layout.addWidget(
                InfoButton(help_key, self._help_viewmodel, parent=container),
                0,
                Qt.AlignmentFlag.AlignVCenter,
            )
            return container
        return widget

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)  # type: ignore[arg-type]

    def _wire_signals(self) -> None:
        self._view_model.database_changed.connect(lambda _db: self._on_database_changed())
        self._view_model.run_started.connect(lambda *_: self._on_run_started())
        self._view_model.run_finished.connect(lambda *_: self._on_run_finished())
        self._view_model.run_failed.connect(lambda *_: self._on_run_finished())

    def _on_database_changed(self) -> None:
        self.set_models(self._view_model.available_models())

    def _on_run_started(self) -> None:
        self.run_button.setEnabled(False)

    def _on_run_finished(self) -> None:
        self.run_button.setEnabled(True)


__all__ = ["ForecastingSidebar"]
