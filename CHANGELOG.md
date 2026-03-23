# Changelog

All notable end-user changes are documented in this file.

The format is based on Keep a Changelog, with entries grouped by release.

## [0.2.2] - 2026-03-23

### Added
- Monthly grouped bar charts now support hover and click interactions, making it easier to inspect and select grouped values directly.

### Changed
- Chart zooming and panning now keep legends more stable and reduce visible flicker during navigation.
- Time-series gaps are rendered more naturally when changing zoom levels.
- Timeline section titles now use consistent spacing in the SOM timeline view.
- Regression feature selections now use cleaner labels (internal notes are no longer shown in the selector text).
- Feature and neuron cluster refreshes are now cancellable and superseded by newer actions, which keeps timeline/feature views aligned with your latest selection.

### Fixed
- Cluster timeline rendering now respects pixel-to-time ratios better, so short/dense cluster ranges stay visible instead of disappearing.
- Monthly chart group ordering now matches ordering used in other grouped chart views.
- Monthly chart signaling warnings were removed.
- Feature list selection is now preserved more reliably during refreshes, especially when data content has not actually changed.
- Restoring large feature selections no longer triggers a noisy burst of selection-change signals.
- Startup and window-state recovery now handle in-use database files more safely: the app informs you and still builds the UI cleanly.

### Performance
- Data fetching and drawing in the Data tab are faster, especially during large-table browsing and repeated viewport updates.
- Zoom/pan chart updates reduce duplicate fetch/render work, improving responsiveness in interactive chart workflows.
- SOM neuron clustering/training paths were optimized to reduce UI jamming on heavier runs.
- SOM post-processing is faster for large datasets by preparing feature/timeline tables off the UI thread.
- Group timeline overlays now use display-aware box limits, improving chart responsiveness and avoiding unreadable overdraw.
- Feature summary statistics are computed with a more efficient vectorized path for large feature sets.
- Feature table refresh avoids unnecessary re-sort/reselect work when incoming data is unchanged.

## [0.2.1] - 2026-03-16
- Fix issues with icons + small styling changes

## [0.2.0] - 2026-03-13

### Added
- Chat session management: save, load, and manage multiple conversation sessions
- AI thinking mode toggle for more detailed analysis responses
- LLM provider selection and model listing (OpenAI, Ollama)
- API key testing flow for LLM providers
- Stop button for chat streaming responses
- CSV group labeling: link CSV columns as group labels for better data organization
- Dynamic theme switching at runtime (no restart needed)
- Language switching at runtime
- Session-aware logging in the database
- Import/export file dialog remembers last used folder

### Changed
- Filter dependency flow is now more intuitive: datasets filter by systems, imports filter by datasets, and groups/tags scope to your selection
- Better data selection performance with async fetching that doesn't freeze the UI
- Chart exports now match the GUI appearance for professional reports
- Statistics tab can plot and export multiple features at once
- Regression and SOM tabs now use unified data fetching for consistency
- Selection settings are now more secure with explicit preset saving options
- Improved chart styling across both light and dark themes for a more cohesive look
- More user-friendly error messages in regression and SOM analysis
- Reset buttons readily available on all chart views

### Fixed
- Filter state restoration now respects data dependencies (datasets available for systems, imports available for datasets)
- Chart feature combo now updates properly when database changes
- SOM saved timeline cluster groups now show correctly in custom selections
- Regression predictions properly preserve source import scope when saving
- Stratify options update correctly when user changes inputs/targets
- Chart selections work correctly even when table rows are sorted
- Timeline cluster groups and statistics properly use scoped filters
- Feature selections are preserved across UI refreshes
- Empty and static features are automatically cleaned during model training
- CSV imports now retain non-numeric columns as features instead of silently dropping them

### Performance
- Significantly faster data filtering and selection operations
- Improved chart responsiveness with better refresh optimization
- More efficient database queries for filtered data access
- Optimized statistics calculations with preprocessed data

### Breaking
- Data database schema now includes `group_label_scopes` for explicit group-to-system/dataset/import linking
- Existing databases should be recreated to fully support scoped group/tag filtering and CSV group linking

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
