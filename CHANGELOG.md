# Changelog

All notable end-user changes are documented in this file.

The format is based on Keep a Changelog, with entries grouped by release.

## [Unreleased]

### Changed
- Data tab filter dependency flow now separates feature-affecting and data-affecting filter updates more clearly, and group/tag filter options are scoped to selected systems/datasets/imports.

### Breaking
- Data database schema now includes `group_label_scopes` for explicit group-to-system/dataset/import linking.
- Existing databases must be recreated to fully use scoped group/tag filtering behavior.

## [0.1.1] - 2026-02-28

### Added
- Faster table experience across the app with improved responsiveness on larger datasets.

### Changed
- Improved table sorting and filtering speed to make everyday browsing and analysis smoother.
- Expanded fast table behavior to more views for a more consistent interaction pattern.
- Improved default table ordering so data appears in a more predictable way after loading.
- Improved startup window behavior and table sizing defaults for better first-use readability.

### Fixed
- Reduced chart redraw and timeline display issues that could cause visual jumping in some time-based views.
- Fixed chart and bar-visual cleanup behavior in repeated interactions.
- Fixed several UI layout/margin inconsistencies across tabs.
- Improved reliability of status/progress feedback in long-running analysis actions.

## [0.1.0] - 2026-02-06

### Added
- Initial public release of FastData for Windows.
- Core data analysis workflow with data import, selections, statistics, charts, SOM, and regression features.
