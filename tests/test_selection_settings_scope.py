from frontend.models.selection_settings import (
    FILTER_SCOPE_GLOBAL,
    FILTER_SCOPE_LOCAL,
    FILTER_SCOPE_SYSTEM,
    FeatureLabelFilter,
    FeatureValueFilter,
    normalize_filter_scope,
)


def test_feature_value_filter_reads_legacy_apply_globally_flags() -> None:
    assert FeatureValueFilter.from_dict({"feature_id": 1, "apply_globally": True}).scope == FILTER_SCOPE_GLOBAL
    assert FeatureValueFilter.from_dict({"feature_id": 1, "apply_globally": False}).scope == FILTER_SCOPE_LOCAL


def test_feature_filters_round_trip_scope_values() -> None:
    value_filter = FeatureValueFilter(feature_id=7, min_value=1.0, max_value=2.0, scope="dataset")
    label_filter = FeatureLabelFilter(label="Gate", min_value=0.0, max_value=5.0, scope="import")

    assert FeatureValueFilter.from_dict(value_filter.to_dict()).scope == "dataset"
    assert FeatureLabelFilter.from_dict(label_filter.to_dict()).scope == "import"


def test_filter_scope_defaults_to_system_for_invalid_values() -> None:
    assert normalize_filter_scope(None) == FILTER_SCOPE_SYSTEM
    assert normalize_filter_scope("unexpected") == FILTER_SCOPE_SYSTEM
