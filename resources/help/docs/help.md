# FastData Help

Context-sensitive help entries for FastData application

Updated: 2026-01-01

Version: 2

## Overview

#### FastData Help

How this help system is organized.

<p>This help content is organized to match how you work in the app.</p>
<ul>
  <li><b>Basics</b> covers core concepts and definitions.</li>
  <li><b>Features</b> groups each major workflow area and its related controls.</li>
</ul>


## Basics

#### Files and Sheets (Imports)

Files/sheets are stored as imports; for users these are effectively the same thing.

<p>The first concept is the <b>import</b>. In practice, users select files and sheets, and each selected
file/sheet becomes an import.</p>
<p>Technically, an import is represented as a database table. From the user perspective, it is fine to think
of <b>file/sheet = import</b>.</p>
<p>Typical import sources:</p>
<ul>
  <li>CSV files</li>
  <li>Excel files and individual sheets</li>
  <li>Other supported tabular sources</li>
</ul>
<p>Imports are the atomic source units that later get grouped into datasets.</p>


#### Datasets

A dataset is a collection of one or more imports used together for analysis.

<p>A <b>dataset</b> groups related imports into one analysis scope.</p>
<p>One dataset can contain many imports (for example, multiple files/sheets from the same process).</p>
<p>The active dataset drives most work in the app: filtering, preprocessing, visualization, and modeling.</p>
<p>Use datasets to separate contexts like production lines, time periods, or raw vs cleaned data.</p>


#### Systems

A system contains multiple datasets and its own feature set.

<p>A <b>system</b> is a higher-level container above datasets.</p>
<p>Each system can hold multiple datasets, and each system owns its own feature set.</p>
<p>This means feature definitions are managed in system context, even when data is split across multiple
datasets in that same system.</p>
<p>Use systems to separate fundamentally different processes, plants, machines, or product families.</p>


#### Features

Features are columns used for analysis and modeling inside a system.

<p><b>Features</b> are the measurable variables (columns) used as model inputs or analysis dimensions.</p>
<p>Because systems own feature sets, feature metadata is managed consistently across datasets within the same
system.</p>
<p>For each feature, users should understand at least:</p>
<ul>
  <li><b>Source</b>: where the measurement comes from (sensor, calculated field, instrument, etc.)</li>
  <li><b>Unit</b>: physical/logical unit (for example C, bar, kg/h, %)</li>
  <li><b>Type</b>: value type like how it was measured (statistical, raw, etc.)</li>
</ul>
<p>Clear feature metadata improves filtering, preprocessing, model quality, and interpretation.</p>


#### Tags

Tags label features for organization, filtering, and workflow clarity.

<p><b>Tags</b> are lightweight labels you assign to features (and optionally other entities) to keep large
models understandable.</p>
<p>Typical tag examples: <i>target</i>, <i>input</i>, <i>quality</i>, <i>temperature</i>, <i>critical</i>, <i>lab</i>, <i>calculated</i>.</p>
<p>Use tags to:</p>
<ul>
  <li>Group related features quickly</li>
  <li>Filter feature lists for modeling or monitoring</li>
  <li>Keep naming simple while still preserving context</li>
</ul>
<p>Tags are metadata only. They do not change raw measured values.</p>


#### Data Flow

Canonical structure: files/sheets -> datasets -> systems -> features -> tags.

<p>The conceptual model is:</p>
<ol>
  <li><b>Files/Sheets</b> are ingested as imports (database tables)</li>
  <li><b>Datasets</b> group one or many imports</li>
  <li><b>Systems</b> group multiple datasets and own feature definitions</li>
  <li><b>Features</b> define the usable variables with source/unit/type metadata</li>
  <li><b>Tags</b> label features for organization and filtering</li>
</ol>
<p>Operationally, you import files first, assign/group them into datasets, work inside a system context,
then manage features and tags for efficient analysis.</p>


#### Target Variable

The variable you want to predict or model.

<p>The <b>Target Variable</b> (also called the dependent variable or label) is what your model
is trying to predict.</p>
<p>Examples:</p>
<ul>
  <li>In a sales forecasting model, the target is <i>sales amount</i></li>
  <li>In a classification problem, the target is the <i>category or class</i></li>
  <li>In regression, the target is a <i>continuous numeric value</i></li>
</ul>
<p>The target variable should be clearly defined before beginning model training.</p>


#### Imported Feature

A feature column with source, unit, type, and optional tags.

<p>This is a <b>feature column</b> available in the current system/dataset context.</p>
<p>Check and maintain feature metadata:</p>
<ul>
  <li><b>Source</b> - Origin of this feature value</li>
  <li><b>Unit</b> - Unit of the feature value (e.g., %, $, °C)</li>
  <li><b>Type</b> - How the feature value was measured (e.g., raw, normalized, statistical)</li>
  <li><b>Tags</b> - Labels for grouping and filtering</li>
</ul>
<p>To use this feature effectively:</p>
<ul>
  <li>Check for missing values and outliers</li>
  <li>Understand its distribution and data type</li>
  <li>Consider if preprocessing (scaling, encoding) is needed</li>
  <li>Evaluate its correlation with the target variable</li>
</ul>
<p>You can view detailed statistics and visualizations for this feature in the data exploration tab.</p>


#### Model Selection

Choose the best algorithm for your prediction task.

<p><b>Model Selection</b> is the process of choosing the most appropriate machine learning
algorithm for your specific problem.</p>
<p>Key considerations:</p>
<ul>
  <li><b>Problem Type</b> - Regression, classification, clustering, or forecasting?</li>
  <li><b>Data Size</b> - Some algorithms work better with small vs. large datasets</li>
  <li><b>Interpretability</b> - Do you need to explain predictions? (Linear models > Neural networks)</li>
  <li><b>Performance Metrics</b> - Use cross-validation to compare models objectively</li>
  <li><b>Training Time</b> - Balance accuracy vs. computational cost</li>
</ul>
<p>FastData provides automatic model selection tools to help you find the best fit.</p>


## Features

### Data

#### Data Filters

Apply conditions to subset your dataset.

<p><b>Filters</b> allow you to select specific rows from your dataset based on conditions.</p>
<p>Use filters to:</p>
<ul>
  <li>Focus on specific time periods</li>
  <li>Exclude outliers or invalid data</li>
  <li>Analyze specific categories or groups</li>
  <li>Compare subsets of data</li>
</ul>
<p>Filters can be combined using AND/OR logic for complex queries.</p>
<p>Filtered data is used for all downstream analysis and visualizations.</p>


#### Date range

Limit data between a start and end timestamp.

<p>Pick a starting and ending date/time to bound the records that are loaded, previewed, and charted.</p>
<ul>
  <li>Set only the start to include everything after that point.</li>
  <li>Set only the end to include everything up to that timestamp.</li>
  <li>Leave both empty to include the full available history.</li>
</ul>


#### Imports and tags

Filter by selected imports and feature tags together.

<p>Use <b>Imports</b> to restrict data to specific ingestion events in the currently selected dataset.</p>
<ul>
  <li>Each import entry corresponds to a file ingestion (file name + import timestamp).</li>
  <li>Select one or many imports to compare runs or isolate a single load.</li>
  <li>Leave imports empty to include all imports in scope.</li>
</ul>
<p>Use <b>Tags</b> to filter which features are available/selected for analysis.</p>
<ul>
  <li>Tags are feature metadata labels (for example, domain or equipment class).</li>
  <li>Select multiple tags to include features that match those labels.</li>
  <li>Combine imports + tags to narrow both rows (data provenance) and columns (feature set).</li>
</ul>


#### Months and groups

Filter by calendar months and predefined groups.

<p>Select calendar months to focus on seasonal patterns, and choose database groups to restrict which entities are processed.</p>
<ul>
  <li>Pick multiple months to compare seasons such as winter vs. summer.</li>
  <li>Combine month and group filters to study specific cohorts during chosen periods.</li>
  <li>Leave a list empty to include all options for that filter.</li>
</ul>


#### Systems and datasets

Scope results by system and dataset in one place.

<p>Use these selectors together to narrow the data to specific systems and the datasets they contain.</p>
<ul>
  <li>Check multiple systems or datasets to compare them side by side.</li>
  <li>Leave either list empty to include every option for that category.</li>
  <li>Combine both filters to isolate behavior for chosen datasets within selected systems.</li>
</ul>


#### Assume day-first dates

Interpret 03/04 as 3 April instead of 4 March.

<p>Enable this if your date strings use <b>day/month</b> order.</p>
<ul>
  <li>Useful for European date formats.</li>
  <li>Disable for US month-first formats.</li>
</ul>


#### CSV decimal

Decimal separator for numeric values.

<p>Specify the decimal character for numeric values.</p>
<ul>
  <li>Common values are <b>.</b> and <b>,</b>.</li>
  <li>Leave as <b>auto</b> to use the default parser behavior.</li>
</ul>


#### CSV delimiter

Character used to separate CSV columns.

<p>Set the delimiter between columns (for example <b>,</b> or <b>;</b>).</p>
<ul>
  <li>Leave as <b>auto</b> to let the importer detect the delimiter.</li>
  <li>Overrides the quick guess if you enter a value.</li>
</ul>


#### CSV encoding

Text encoding for CSV files.

<p>Provide an encoding name (for example <i>utf-8</i> or <i>latin-1</i>).</p>
<ul>
  <li>Leave as <b>auto</b> to try common encodings automatically.</li>
  <li>Set explicitly when you see garbled characters.</li>
</ul>


#### Dataset name

Attach the data to a specific dataset.

<p>Pick a <b>dataset</b> that belongs to the selected system.</p>
<ul>
  <li>Use the dropdown to pick an existing dataset or type a new one.</li>
  <li><i>&lt;Use sheet name&gt;</i> uses the Excel sheet name as the dataset value.</li>
</ul>


#### Date column

Choose which column contains timestamps.

<p>Select the column that holds datetime values.</p>
<ul>
  <li>Leave as <b>auto</b> to let the importer detect a timestamp column.</li>
  <li>Set explicitly if the detection picks the wrong column.</li>
</ul>


#### Datetime formats

Provide explicit datetime formats (optional).

<p>Comma-separated list of strftime-compatible formats.</p>
<ul>
  <li>Use <b>auto</b> to let the importer guess formats.</li>
  <li>Supplying formats can speed up parsing and avoid ambiguities.</li>
</ul>


#### Dot time formatting

Parse 9.00 as 09:00.

<p>Convert dot-separated times to colon-separated times while parsing.</p>
<ul>
  <li>Example: <i>13.47</i> becomes <i>13:47:00</i>.</li>
  <li>Disable if dots are not used as time separators.</li>
</ul>


#### Header row amount

Number of header rows in Excel files.

<p>Define how many rows at the top of the sheet are header rows.</p>
<ul>
  <li>Use <b>0</b> if the sheet has no header row.</li>
  <li>This setting is only used for Excel-style files.</li>
</ul>


#### Force meta columns

Treat specific columns as metadata (not measurements).

<p>Comma-separated list of column names to force into metadata.</p>
<ul>
  <li>Use this to prevent ID or label columns from becoming features.</li>
  <li>Matching is case-insensitive.</li>
</ul>


#### Base name row

Choose which header row contains the base feature name.

<p>Select the header row that contains the base feature name (e.g., <i>Temperature</i>).</p>
<ul>
  <li>Set to <b>None</b> if the file does not contain that row.</li>
  <li>Combine with the header delimiter to split values into parts.</li>
</ul>


#### Header delimiter

Split multi-part headers by a delimiter.

<p>Split header text into multiple parts using a delimiter (for example <b>"_"</b> or <b>" | "</b>).</p>
<ul>
  <li>Use this when the header contains concatenated fields (e.g., <i>Temp_Outdoor_C</i>).</li>
  <li>Leave empty to keep the header text as-is.</li>
</ul>


#### Type row

Select the header row for qualifiers or annotations.

<p>Qualifiers capture extra descriptors like <i>min/max</i>, <i>status</i>, or <i>quality</i>.</p>
<ul>
  <li>Use when headers include additional descriptors beyond base name and unit.</li>
  <li>Leave unset if not applicable.</li>
</ul>


#### Source row

Select the header row for source/series labels.

<p>Use this when a header row identifies the source or series for a feature.</p>
<ul>
  <li>Examples: phase identifiers or channel names.</li>
  <li>Leave unset if the file does not include source rows.</li>
</ul>


#### Unit row

Select the header row containing units.

<p>Point to the header row that contains measurement units (e.g., <i>°C</i>, <i>kW</i>).</p>
<ul>
  <li>Units are stored with the feature metadata.</li>
  <li>Leave unset if units are already embedded in names.</li>
</ul>


#### Ignore column prefixes

Skip columns that start with listed prefixes.

<p>Comma-separated prefixes to ignore when importing columns.</p>
<ul>
  <li>Example: <i>Time, AE_</i> will ignore columns starting with those values.</li>
  <li>Use this to exclude helper columns from imports.</li>
</ul>


#### System name

Label the imported data with a system name.

<p>Choose the <b>system</b> that owns the data you are importing.</p>
<ul>
  <li>Existing systems appear in the list; you can type a new name to create one.</li>
  <li>The system is used to organize datasets and imports in the database.</li>
</ul>


#### Use DuckDB CSV import

Load large CSVs with DuckDB's fast parser.

<p>Enable DuckDB's CSV import pipeline for large files.</p>
<ul>
  <li>Recommended for very large CSVs (100MB+).</li>
  <li>Disable if you need strict pandas-style parsing behavior.</li>
</ul>


#### Data

Load datasets and browse their contents.

<p>The <b>Data</b> tab is the starting point for working with FastData.</p>
<ul>
  <li>Connect to DuckDB/DB files and pick the active table.</li>
  <li>Preview rows with paging to verify that the dataset loaded correctly.</li>
  <li>Inspect schema information such as column names, data types, and row counts.</li>
  <li>Refresh or swap datasets without restarting the application.</li>
</ul>
<p>Use this tab to confirm your source data looks correct before you create selections or train models.</p>


### Selections

#### Selections

Create reusable feature selections and filter presets.

<p>The <b>Selections</b> tab lets you curate which features to keep for downstream tasks.</p>
<ul>
  <li>Toggle columns on or off to build a focused feature set.</li>
  <li>Right-click a saved feature to convert its measurement values into database groups.</li>
  <li>Use <b>Reload features</b> to restore the table from the database and discard unsaved edits.</li>
  <li>Apply saved filters to limit rows before modeling.</li>
  <li>Store multiple presets and quickly switch between them for experimentation.</li>
  <li>Export or import selection databases to share with teammates.</li>
</ul>
<p>Adjusting selections here keeps your preprocessing and modeling steps consistent.</p>


### Statistics

#### Data Preprocessing

Clean and transform data before analysis.

<p><b>Preprocessing</b> is the critical step of preparing raw data for machine learning models.</p>
<p>Common preprocessing operations in FastData:</p>
<ul>
  <li><b>Missing Value Handling</b> - Impute or remove rows with missing data</li>
  <li><b>Normalization/Scaling</b> - Standardize feature ranges (e.g., StandardScaler, MinMaxScaler)</li>
  <li><b>Encoding</b> - Convert categorical variables to numeric (one-hot, label encoding)</li>
  <li><b>Outlier Removal</b> - Filter extreme values that may skew results</li>
  <li><b>Feature Engineering</b> - Create new derived features from existing ones</li>
</ul>
<p>Proper preprocessing often has more impact on model performance than algorithm selection.</p>


#### Aggregation

Summarize multiple points that fall within a timestep.

<p>When multiple measurements land in the same resampled bucket, aggregation decides how they are combined.</p>
<ul>
  <li><b>avg</b> (default) computes the mean value.</li>
  <li><b>min</b>/<b>max</b> keep the extremes.</li>
  <li><b>first</b>/<b>last</b> preserve ordering-sensitive data.</li>
  <li><b>median</b> is robust against outliers.</li>
</ul>
<p>Pick the method that best matches how you would summarize overlapping readings.</p>


#### Fill empty

Choose how to handle missing timestamps after resampling.

<p>Determines how gaps created during resampling are filled.</p>
<ul>
  <li><b>none</b> leaves gaps as-is.</li>
  <li><b>zero</b> inserts 0 for missing values.</li>
  <li><b>prev</b> carries the previous value forward.</li>
  <li><b>next</b> uses the next known value.</li>
</ul>
<p>Forward or backward filling is useful when signals change slowly and occasional gaps appear.</p>


#### Moving average

Smooth measurements with an optional rolling window.

<p>Applies a rolling mean over the selected window to reduce noise.</p>
<ul>
  <li><b>none</b> leaves the data untouched.</li>
  <li>Preset windows (e.g., <b>5 minutes</b>) smooth short-term spikes.</li>
  <li>Type a custom window in seconds for finer control.</li>
</ul>
<p>Use smoothing when charts or models are sensitive to rapid fluctuations.</p>


#### Timestep

Resample data to a fixed interval or leave it on auto.

<p>Choose the interval used to resample incoming measurements.</p>
<ul>
  <li><b>auto</b> keeps the native spacing of the data.</li>
  <li>Preset values such as <b>1 minute</b> or <b>1 hour</b> resample the series to an even grid.</li>
  <li>Enter a custom number of seconds to match an exact cadence.</li>
</ul>
<p>Resampling can make downstream statistics and charts easier to compare.</p>


#### Statistics actions

Run the computation and store results when you are happy with them.

<p>Use these buttons to generate and persist the statistics previewed in this tab.</p>
<ul>
  <li><b>Gather statistics</b> executes the selected statistics with the current filters, mode, and preprocessing settings.</li>
  <li><b>Save to database</b> writes the last computed preview into the database so it can be reused in other tabs.</li>
  <li>Run again any time you tweak filters, periods, or selected statistics to refresh the preview.</li>
</ul>


#### Group column

Pick the categorical column that defines each group in column mode.

<p>Available options come from the dataset and only apply when <b>Group by column</b> is selected.</p>
<ul>
  <li>Choose identifiers like <b>system</b>, <b>dataset</b>, or any other categorical field.</li>
  <li>Each distinct value becomes its own row or bar in the preview.</li>
  <li>Leave the dropdown empty in time-based mode—the control is disabled automatically.</li>
</ul>


#### Statistics to compute

Pick the summary measures calculated for each time bucket or group.

<p>Select one or more statistics to include in the output. Each choice adds a column to the preview and saved results.</p>
<ul>
  <li><b>Average</b> (avg/mean): mean value of the samples.</li>
  <li><b>Minimum</b> / <b>Maximum</b>: smallest or largest observation.</li>
  <li><b>Median</b>: middle value that is robust to outliers.</li>
  <li><b>Std Dev</b>: spread of the samples around the mean.</li>
  <li><b>Sum</b>: total of all values in the bucket.</li>
  <li><b>Count</b>: number of rows contributing to the statistic.</li>
  <li><b>Interquartile Range</b> (IQR): difference between the 75th and 25th percentiles.</li>
  <li><b>Outliers (3σ %)</b>: percentage of points that land three standard deviations from the mean.</li>
  <li><b>Max derivative</b>: largest absolute change between consecutive samples.</li>
</ul>
<p>Pick only what you need to keep previews fast; you can always rerun with more metrics.</p>


#### Aggregation mode

Choose whether to aggregate over time or by a specific column.

<ul>
  <li><b>Time based</b> groups measurements into regular periods using the <b>Statistics period</b> setting.</li>
  <li><b>Group by column</b> ignores time buckets and instead aggregates by the values of the chosen <b>Group column</b>.</li>
</ul>
<p>Switching modes changes how the preview table and chart summarize your data.</p>


#### Statistics period

Define the window used for time-based aggregation.

<p>Time-based mode buckets records into regular windows before applying your chosen statistics.</p>
<ul>
  <li><b>hourly</b>, <b>daily</b>, <b>weekly</b>, <b>monthly</b>: convenient presets that translate to fixed second intervals.</li>
  <li><b>custom</b>: enter a number of seconds in <b>Custom period (s)</b> to match your own cadence.</li>
  <li>Pick a shorter window for fine-grained trends or a longer one for smoother, aggregate views.</li>
</ul>


#### Separate timeframes

Choose whether group-kind statistics are split by each saved timeframe segment.

<p>This setting applies when using <b>Group by column</b> with a database group kind (for example <b>group:som_cluster</b>).</p>
<ul>
  <li><b>Enabled</b>: each saved timeframe segment is treated as its own group bucket.</li>
  <li><b>Disabled</b>: all rows with the same group label are merged into one bucket, regardless of timeframe segments.</li>
</ul>


#### Statistics

Review derived statistics and prepared measurements.

<p>The <b>Statistics</b> tab focuses on validating the numbers produced from your data.</p>
<ul>
  <li>Preview aggregated measurements and sanity-check values before saving them.</li>
  <li>See how filters and preprocessing settings shape the resulting metrics.</li>
  <li>Confirm column names, units, and sampling windows prior to exporting.</li>
</ul>
<p>This tab highlights the statistical outputs themselves, while the preprocessing widget controls how they are generated.</p>


### Charts

#### Charts

Visualize your dataset with quick plots.

<p>The <b>Charts</b> tab provides fast visual feedback on your data.</p>
<ul>
  <li>Create common plots such as histograms, scatter plots, and correlations.</li>
  <li>Use selections and filters to focus charts on specific subsets.</li>
  <li>Pick a target feature and run <b>Find feature correlations</b> to auto-build:
    a top-20 correlation bar chart plus 3 scatter charts for the strongest matches.</li>
  <li>Compare relationships between variables before modeling.</li>
</ul>
<p>Use charts to spot trends, outliers, and data quality issues early.</p>


### SOM

#### Self-Organizing Maps (SOM)

Unsupervised neural network for data visualization and clustering.

<p><b>Self-Organizing Maps</b> (also called Kohonen maps) are a type of artificial neural network
trained using unsupervised learning to produce a low-dimensional representation of the input space.</p>
<p>Key features:</p>
<ul>
  <li><b>Dimensionality Reduction</b> - Projects high-dimensional data onto a 2D grid</li>
  <li><b>Topology Preservation</b> - Similar data points are mapped to nearby locations</li>
  <li><b>Clustering</b> - Automatically groups similar observations</li>
  <li><b>Anomaly Detection</b> - Outliers appear in distant or sparse regions</li>
</ul>
<p>SOMs are particularly useful for exploratory data analysis and pattern recognition.</p>
<p>FastData uses the <code>MiniSom</code> library for efficient SOM computation.</p>


#### Auto cluster features

Automatically run feature clustering after SOM training.

<p>When enabled, feature clustering starts automatically each time a SOM model finishes training.</p>
<ul>
  <li>Uses the current Feature clustering settings (model, Max K, Clusters, Score).</li>
  <li>Runs in the background, the same as clicking <b>Cluster features</b> manually.</li>
  <li>Disable this if you only want to train SOM maps without updating feature clusters.</li>
</ul>


#### Cluster features

Group SOM component planes into feature clusters.

<p>Runs clustering on the trained SOM component planes.</p>
<ul>
  <li>Requires a trained SOM model; train first if the button is disabled.</li>
  <li>Outputs cluster labels you can inspect on the feature map tab.</li>
  <li>Use the scoring metric to automatically select K when <b>Clusters</b> is Auto.</li>
</ul>


#### Max K & Clusters

Control the cluster search range or pin a fixed number.

<p>These two fields work together:</p>
<ul>
  <li><b>Max K</b> sets the upper bound for automatic K search (min is 2).</li>
  <li><b>Clusters</b> overrides auto-search when set; 0 or "Auto" falls back to scoring.</li>
  <li>If the dataset has few samples, the system trims the range to stay valid.</li>
</ul>


#### Feature clustering model

Pick the algorithm used to group feature planes.

<p>Choose how features are grouped based on their component planes.</p>
<ul>
  <li><b>K-Means / Mini-Batch K-Means</b> – fast centroids, works well as a default.</li>
  <li><b>Agglomerative</b> – hierarchical merges; good when you expect nested structure.</li>
  <li><b>Spectral</b> – uses graph cuts; handles complex shapes but needs a few more samples.</li>
</ul>
<p>All options support automatic K search when allowed by the method.</p>


#### Scoring metric

Metric used to pick the best K during auto-search.

<p>Scores compare candidate clusterings:</p>
<ul>
  <li><b>Silhouette</b> – balances cohesion and separation; good general-purpose metric.</li>
  <li><b>Calinski-Harabasz</b> – favors compact, well-separated clusters; fast to compute.</li>
  <li><b>Davies-Bouldin</b> – lower is better; sensitive to overlapping clusters.</li>
</ul>
<p>Only used when <b>Clusters</b> is set to Auto.</p>


#### Epochs

How many training passes to run (defaults to 100).

<p><b>Epochs</b> determines how many times the algorithm sweeps through your data.</p>
<ul>
  <li>Leave at <code>0</code> or empty to use the default of 100 epochs.</li>
  <li>More epochs improve convergence but increase training time.</li>
  <li>If training stalls, try a few extra epochs together with a smaller learning rate.</li>
</ul>


#### Learning rate

Step size used when updating SOM weights.

<p><b>Learning rate</b> controls how quickly the map adapts to each sample.</p>
<ul>
  <li>Higher values (e.g., 0.5) converge faster but can overshoot fine structure.</li>
  <li>Lower values make training steadier at the cost of more iterations.</li>
  <li>Learning rate decays over epochs to stabilize the final map.</li>
</ul>


#### Map height

Vertical size of the SOM grid (auto if left blank).

<p><b>Map height</b> controls how many neurons the map has along the Y-axis.</p>
<ul>
  <li>Leave it empty to reuse the automatic heuristic used for width.</li>
  <li>Specify a value to make rectangular maps (e.g., wider than tall) when you expect directional structure.</li>
  <li>Both width and height must be positive; the UI enforces a minimum of 2.</li>
</ul>


#### Map width

Horizontal size of the SOM grid (auto if left blank).

<p><b>Map width</b> controls how many neurons the map has along the X-axis.</p>
<ul>
  <li>Leave the field empty to let FastData pick a balanced width based on your dataset size (roughly <code>sqrt(5 * sqrt(N))</code>).</li>
  <li>Enter an integer to force a specific width. Values below 2 are clamped to a minimum grid size.</li>
  <li>Wider maps expose more local structure but require more samples to train well.</li>
</ul>


#### Normalisation

How features are scaled before training.

<p>Scaling features keeps each variable on a comparable range.</p>
<ul>
  <li><b>Z-score</b> (default) standardizes to zero mean and unit variance—best general choice.</li>
  <li><b>Min-max</b> rescales each column to [0, 1] to preserve relative spacing.</li>
  <li><b>None</b> skips scaling; use only when your inputs are already normalized.</li>
</ul>


#### Sigma

Initial neighborhood radius for SOM training.

<p><b>Sigma</b> sets how far the learning influence of a winning neuron spreads across the grid at the start of training.</p>
<ul>
  <li>Larger values encourage smoother, more global organization early on.</li>
  <li>Smaller values keep neighborhoods tight, emphasizing local differences.</li>
  <li>The value decays during training; 6.0 is a sensible default for most datasets.</li>
</ul>


#### Training mode

Choose between batch and random SOM updates.

<p>Pick the strategy used to present samples during training.</p>
<ul>
  <li><b>Batch</b> (default) updates the entire map per epoch for stable, reproducible results.</li>
  <li><b>Random</b> feeds one sample at a time in random order, which can discover sharp local structures.</li>
  <li>If results look noisy, switch back to batch mode or increase sigma.</li>
</ul>


#### Auto cluster timeline

Automatically run neuron clustering after SOM training.

<p>When enabled, neuron clustering starts automatically after each SOM training run.</p>
<ul>
  <li>Uses the current Neuron clustering settings (model, Max K, Clusters, Score).</li>
  <li>Keeps timeline cluster mode and cluster map ready without a separate manual step.</li>
  <li>Disable this if you want manual control over when timeline clusters are recalculated.</li>
</ul>


#### Cluster neurons

Partition the SOM grid into neuron clusters.

<p>Runs clustering on the neurons themselves, using their codebook vectors.</p>
<ul>
  <li>Requires a trained SOM model before it can run.</li>
  <li>Results can be visualized on the maps to highlight regional patterns.</li>
  <li>Use <b>Max K</b>, <b>Clusters</b>, and <b>Score</b> to tune how fine-grained the regions are.</li>
</ul>


#### Max K & Clusters (neurons)

Set the range or fixed count for neuron groups.

<p>Configure how many neuron groups to consider:</p>
<ul>
  <li><b>Max K</b> caps the auto-search; higher values explore finer partitions.</li>
  <li><b>Clusters</b> locks in an exact count when set to a value &gt; 0.</li>
  <li>If the SOM grid is small, the available K values are trimmed to remain valid.</li>
</ul>


#### Neuron clustering model

Pick the algorithm used to group SOM neurons.

<p>The same clustering algorithms are available for neurons as for features:</p>
<ul>
  <li><b>K-Means / Mini-Batch K-Means</b> – fast, dependable defaults.</li>
  <li><b>Agglomerative</b> – builds a hierarchy of regions on the map.</li>
  <li><b>Spectral</b> – can separate winding or non-convex neuron regions.</li>
</ul>
<p>Choose the method that matches how you expect neurons to organize across the grid.</p>


#### Scoring metric (neurons)

Metric used when auto-selecting neuron clusters.

<p>Uses the same metrics as feature clustering:</p>
<ul>
  <li><b>Silhouette</b> – balanced, works well on most maps.</li>
  <li><b>Calinski-Harabasz</b> – rewards tight neuron groups.</li>
  <li><b>Davies-Bouldin</b> – lower is better; highlights overlapping groups.</li>
</ul>


#### Cluster Timeline

Shows BMU/cluster states over time, with optional selected-feature overlays.

<p>This chart plots timeline layers selected in <b>Display</b>.</p>
<ul>
  <li><b>BMU</b> shows the raw winning neuron index over time.</li>
  <li><b>Neuron clusters</b> shows clustered neuron IDs after clustering.</li>
  <li><b>Selected features</b> overlays scaled feature traces on top of the timeline.</li>
  <li>Use this to spot regime shifts and align feature changes with BMU/cluster transitions.</li>
</ul>


#### Cluster Map

Neuron grid colored by cluster assignment.

<p>The cluster map colors each neuron by its cluster ID.</p>
<ul>
  <li>Use it to see spatial structure in the clustering.</li>
  <li>Click neurons to inspect specific cluster groups.</li>
</ul>


#### Data Table

Row-level BMU assignments used by the timeline.

<p>This table lists BMU assignments for each data row.</p>
<ul>
  <li>Select rows to highlight their neurons on the cluster map.</li>
  <li>Use it to inspect specific time ranges in detail.</li>
</ul>


#### Timeline Display Layers

Select one or more layers: BMU, neuron clusters, and selected features.

<p>Use the Display multi-select control to choose timeline layers:</p>
<ul>
  <li><b>BMU</b>: shows the raw winning neuron index over time.</li>
  <li><b>Neuron clusters</b>: shows clustered neuron IDs after clustering.</li>
  <li><b>Selected features</b>: overlays feature traces for currently selected features.</li>
</ul>


#### Save as timeframes

Store contiguous cluster runs as start/end ranges instead of single points.

<p>When saving timeline clusters, this option controls how group assignments are stored.</p>
<ul>
  <li><b>Enabled</b>: consecutive rows with the same cluster label are merged into one timeframe range.</li>
  <li><b>Disabled</b>: each timeline row is saved as an individual timestamp assignment.</li>
  <li>Timeframe mode improves readability in statistics and group-based analysis by preserving segment boundaries.</li>
</ul>


#### SOM

Explore data structure with Self-Organizing Maps.

<p>The <b>SOM</b> tab visualizes high-dimensional data on a 2D grid.</p>
<ul>
  <li>Train self-organizing maps to cluster similar observations.</li>
  <li>Inspect component planes and distance maps to spot patterns.</li>
  <li>Experiment with map sizes and training parameters.</li>
</ul>
<p>Use this view for exploratory analysis and for identifying interesting cohorts.</p>


### Regression

#### Regression Analysis

Build predictive models to estimate continuous values.

<p><b>Regression</b> is a statistical method for modeling the relationship between a dependent variable 
and one or more independent variables (features).</p>
<p>FastData supports multiple regression algorithms including:</p>
<ul>
  <li><b>Linear Regression</b> - Simple, interpretable baseline</li>
  <li><b>Ridge/Lasso</b> - Regularized linear models to prevent overfitting</li>
  <li><b>Random Forest</b> - Ensemble method for complex non-linear relationships</li>
  <li><b>Gradient Boosting</b> - Advanced ensemble technique for best performance</li>
</ul>
<p>Use regression when your target variable is continuous (e.g., price, temperature, sales volume).</p>


#### Folds

Set how many slices to use when performing K-Fold style validation.

<p>The number of <b>folds</b> controls how many train/validation rotations are executed.</p>
<ul>
  <li>Common choices are 5 or 10; higher values increase runtime but give smoother estimates.</li>
  <li>For very small datasets keep folds low so each validation set still contains enough samples.</li>
  <li>Time series splits always step forward through the ordered data regardless of the fold count.</li>
</ul>


#### Group

Choose the group label used for Group K-Fold.

<p>Select a <b>group kind</b> so all rows from the same group stay in the same fold.</p>
<ul>
  <li>Use this when data points are clustered by equipment, batch, site, or other group labels.</li>
  <li>If the selected group has too few unique labels, the run falls back to K-Fold.</li>
  <li>Only used with Group K-Fold.</li>
</ul>


#### Shuffle folds

Randomly reshuffle rows before forming non-time-series folds.

<p>Enable shuffling to randomise the order of records before K-Fold or Stratified K-Fold splits.</p>
<ul>
  <li>Recommended when data has any temporal or grouped ordering that could bias folds.</li>
  <li>Automatically disabled for time series validation to preserve chronological order.</li>
  <li>Pair with a fixed random state in model parameters when you need repeatable runs.</li>
</ul>


#### Cross-validation strategy

Pick how training/validation folds are built.

<p>Cross-validation helps estimate how well a model generalises. Choose a strategy that matches your data.</p>
<ul>
  <li><b>No cross-validation</b>: train once on the full training set.</li>
  <li><b>K-Fold</b>: split rows into <i>k</i> equal parts and rotate the holdout fold.</li>
  <li><b>Stratified K-Fold</b>: like K-Fold but keeps the class/bin distribution of the chosen stratify feature.</li>
  <li><b>Time series split</b>: preserves order by using earlier rows for training and later rows for validation.</li>
  <li><b>Group K-Fold</b>: keep all rows from the same group in the same fold to prevent leakage.</li>
</ul>
<p>Time-series splits build expanding/rolling windows in chronological order, ignore shuffling, and can use a time gap to reduce leakage.</p>


#### Stratify by

Balance folds by matching the distribution of a feature.

<p>Select a categorical feature or the target to <b>stratify</b> K-Fold splits.</p>
<ul>
  <li>Stratification keeps each fold representative when the target has few unique values or strong grouping.</li>
  <li>Works with Stratified K-Fold and test split stratification; ignored for time series validation.</li>
  <li>Leave empty to let the system choose the most suitable option or disable stratification.</li>
</ul>


#### Time gap

Skip recent observations between training and validation windows.

<p>When using the <b>Time series split</b> strategy, the gap adds a buffer between the end of the training window and the start of the validation window.</p>
<ul>
  <li>Use a positive gap to avoid leakage when measurements have autocorrelation.</li>
  <li>Set to 0 to use adjacent windows.</li>
  <li>Time series splits never shuffle; they always move forward in time.</li>
  <li>This setting is ignored for non-time-series strategies.</li>
</ul>


#### Dimensionality reduction

Optionally project features into a lower-dimensional representation before modeling.

<p>Dimensionality-reduction methods transform your selected input features before the regression model is trained.</p>
<ul>
  <li>You can select multiple methods; each selected method multiplies the total number of runs.</li>
  <li>Methods run after feature selection and before the regression model.</li>
  <li>Use this when you have many correlated or noisy inputs, or when you want a compact representation.</li>
</ul>


#### Feature selection

Optionally enable automatic selectors before training.

<p>Feature selectors reduce the input columns to the most informative subset.</p>
<ul>
  <li>Choose one or more selectors to compare, or leave all unchecked to train with every selected feature.</li>
  <li>Each selector may expose hyperparameters in the <b>Hyperparameters</b> section below.</li>
  <li>Combining selectors and models multiplies the number of experiment runs.</li>
</ul>


#### Learning rate (AdaBoost)

Contribution of each weak learner.

<p>Lower values require more estimators to reach the same performance.</p>
<p>Higher values can overemphasize errors and reduce stability.</p>


#### Number of estimators (AdaBoost)

Number of weak learners to combine.

<p>More estimators can improve accuracy but increase runtime.</p>
<p>Too many stages can overfit noisy data.</p>


#### Random state (AdaBoost)

Seed for boosting randomness.

<p>Fix this value for repeatable results.</p>
<p>Helps compare parameter tweaks consistently.</p>


#### Max depth (decision tree)

Limit the depth of the tree.

<p>Use <b>None</b> to expand until all leaves are pure or minimal.</p>
<p>Shallower trees are easier to interpret but may underfit.</p>


#### Min samples leaf (decision tree)

Minimum samples required in a leaf node.

<p>Higher values smooth predictions and improve generalization.</p>
<p>Use larger leaves for noisy or sparse datasets.</p>


#### Min samples split (decision tree)

Minimum samples required to split a node.

<p>Higher values reduce overfitting.</p>
<p>Set as a count or fraction of the training samples.</p>


#### Random state (decision tree)

Seed for randomized splits.

<p>Fix this value for repeatable trees.</p>
<p>Useful when comparing depth or split settings.</p>


#### Alpha (elastic net)

Regularization strength for elastic net.

<p>Controls the combined L1/L2 penalty magnitude.</p>
<p>Increase to shrink coefficients more aggressively.</p>


#### L1 ratio

Balance between L1 and L2 penalties.

<p>0.0 is pure ridge (L2), 1.0 is pure lasso (L1).</p>
<p>Intermediate values blend sparsity with coefficient shrinkage.</p>


#### Max iterations (elastic net)

Maximum optimizer steps before stopping.

<p>Increase if convergence is slow or unstable.</p>
<p>Higher values can improve accuracy on large feature sets.</p>


#### Random state (elastic net)

Seed for solver randomness.

<p>Fix this value to reproduce elastic net results.</p>
<p>Only relevant when the optimizer is stochastic.</p>


#### Max depth (extra trees)

Limit the depth of each tree.

<p>Use <b>None</b> to expand until all leaves are pure or minimal.</p>
<p>Lower depths reduce variance and improve generalization.</p>


#### Min samples leaf (extra trees)

Minimum samples required in a leaf node.

<p>Higher values smooth the model and reduce variance.</p>
<p>Larger leaves can help with noisy measurements.</p>


#### Min samples split (extra trees)

Minimum samples required to split a node.

<p>Higher values reduce overfitting.</p>
<p>Set as a count or fraction of the training samples.</p>


#### Number of trees (extra trees)

How many extra trees to build.

<p>More trees improve stability but increase runtime.</p>
<p>Extra Trees are more randomized, so more estimators help.</p>


#### Random state (extra trees)

Seed for randomized splits.

<p>Fix this value for repeatable results.</p>
<p>Controls the randomness of feature and split selection.</p>


#### Learning rate (gradient boosting)

Shrinkage applied to each boosting step.

<p>Smaller values require more estimators but can generalize better.</p>
<p>Larger values converge faster but risk overshooting.</p>


#### Max depth (gradient boosting)

Depth of individual regression trees.

<p>Shallow trees reduce variance but may underfit.</p>
<p>Depth 2-4 is common for stable boosting models.</p>


#### Min samples leaf (gradient boosting)

Minimum samples required in a leaf node.

<p>Higher values improve generalization on noisy data.</p>
<p>Larger leaves produce smoother predictions.</p>


#### Min samples split (gradient boosting)

Minimum samples required to split a node.

<p>Higher values reduce model variance.</p>
<p>Set as a count or fraction of the training samples.</p>


#### Boosting stages

Number of boosting stages to perform.

<p>More stages can improve accuracy but raise risk of overfitting.</p>
<p>Pair with a smaller learning rate for smoother training.</p>


#### Random state (gradient boosting)

Seed for the boosting process.

<p>Fix this value for repeatable results.</p>
<p>Ensures the same subsampling order where applicable.</p>


#### Algorithm (KNN)

Search algorithm for nearest neighbors.

<ul>
  <li>auto selects based on data size.</li>
  <li>ball_tree and kd_tree are optimized for structured data.</li>
  <li>brute checks all points directly.</li>
  <li>High-dimensional data often defaults to brute force.</li>
</ul>


#### Number of neighbors (KNN)

How many neighbors to average for predictions.

<p>Lower values fit locally; higher values smooth predictions.</p>
<p>Odd numbers can reduce tie votes in classification-like data.</p>


#### Weights (KNN)

Weighting strategy for neighbors.

<ul>
  <li>uniform treats all neighbors equally.</li>
  <li>distance gives closer neighbors more influence.</li>
  <li>Distance weighting benefits from scaled features.</li>
</ul>


#### Alpha (lasso)

Regularization strength for lasso regression.

<p>Higher values apply stronger L1 penalty and promote sparsity.</p>
<p>Too much regularization can drive useful coefficients to zero.</p>


#### Max iterations (lasso)

Maximum optimizer steps before stopping.

<p>Increase if you see convergence warnings or unstable results.</p>
<p>Larger datasets or high regularization may require more iterations.</p>


#### Random state (lasso)

Seed for solver randomness.

<p>Fix this value to make lasso results repeatable.</p>
<p>Applies when the optimizer uses random coordinate selection.</p>


#### Fit intercept

Include an intercept term in the linear regression model.

<p>When enabled, the model learns a bias term in addition to feature weights.</p>
<ul>
  <li>Disable only when your features are already centered around zero.</li>
  <li>The intercept shifts predictions up or down without changing slopes.</li>
</ul>


#### Positive coefficients

Constrain coefficients to be non-negative.

<p>Forces all learned weights to be positive, which can help with interpretability.</p>
<ul>
  <li>Use when negative contributions are not meaningful for your data.</li>
  <li>Constraint can reduce flexibility and slightly slow down fitting.</li>
</ul>


#### Activation (MLP)

Activation function used in hidden layers.

<p><b>relu</b> is a strong default; <b>tanh</b> can help with bounded data.</p>


#### Alpha (MLP)

L2 regularization strength for the network.

<p>Higher values apply stronger weight decay.</p>


#### Layers (MLP)

Sizes of the hidden layers for the neural network.

<p>Enter comma-separated sizes, e.g. <b>64,32,16</b> for three layers.</p>
<p>More layers and neurons increase capacity but can overfit.</p>


#### Learning rate schedule (MLP)

How the learning rate evolves during training.

<p><b>constant</b> uses a fixed rate.</p>
<p><b>adaptive</b> reduces the rate when progress stalls.</p>


#### Max iterations (MLP)

Maximum training epochs.

<p>Increase if the model stops before converging.</p>


#### Random state (MLP)

Seed for weight initialization.

<p>Fix this value for repeatable runs.</p>


#### Solver (MLP)

Optimizer used to train the network.

<p><b>adam</b> is robust for most datasets.</p>
<p><b>lbfgs</b> can converge faster on smaller datasets.</p>


#### Polynomial degree

Degree of polynomial features.

<p>Higher degrees allow more complex curves but can overfit.</p>
<p>Degrees 2-3 are common starting points for nonlinear trends.</p>


#### Fit intercept (polynomial)

Include a bias term in the polynomial model.

<p>Enable unless your features are centered around zero.</p>
<p>Disable if you already include a bias feature in preprocessing.</p>


#### Max depth (random forest)

Limit the depth of each tree.

<p>Use <b>None</b> to expand until all leaves are pure or minimal.</p>
<p>Shallower trees reduce variance but may underfit.</p>


#### Min samples leaf (random forest)

Minimum samples required in a leaf node.

<p>Higher values smooth the model and reduce variance.</p>
<p>Set as a count or fraction for larger datasets.</p>


#### Min samples split (random forest)

Minimum samples required to split a node.

<p>Higher values reduce overfitting.</p>
<p>Set as a count or fraction of the training samples.</p>


#### Number of trees (random forest)

How many trees to build in the forest.

<p>More trees improve stability but increase runtime.</p>
<p>Accuracy gains diminish once the forest is large enough.</p>


#### Random state (random forest)

Seed for the bootstrap and feature selection.

<p>Fix this value for repeatable forests.</p>
<p>Helps compare runs when tuning other parameters.</p>


#### Alpha (ridge)

Regularization strength for ridge regression.

<p>Higher values apply stronger L2 penalty and reduce coefficient magnitude.</p>
<ul>
  <li>Start small and increase if the model overfits.</li>
  <li>Very large values can underfit by shrinking weights too much.</li>
</ul>


#### Random state (ridge)

Seed for solver randomness where applicable.

<p>Set a fixed value to make the ridge solution reproducible.</p>
<p>Only affects solvers that rely on stochastic optimization.</p>


#### Ridge solver

Numerical method used to fit ridge regression.

<p>Select a solver to match dataset size and stability requirements.</p>
<ul>
  <li>auto chooses a sensible default.</li>
  <li>svd and cholesky are good for dense data.</li>
  <li>lsqr, sparse_cg, sag, and saga handle large datasets.</li>
  <li>lbfgs supports constrained or sparse scenarios.</li>
  <li>sag and saga typically benefit from standardized features.</li>
</ul>


#### Regularization (C)

Penalty for errors in SVR.

<p>Higher values fit the training data more closely.</p>
<p>Too high can overfit; too low can underfit.</p>


#### Epsilon (SVR)

Margin of tolerance in the loss function.

<p>Higher values ignore small errors and create a smoother fit.</p>
<p>Larger epsilon usually yields fewer support vectors.</p>


#### Kernel (SVR)

Kernel function used by support vector regression.

<p>Choose the kernel that best matches the data shape.</p>
<ul>
  <li>rbf for nonlinear relationships.</li>
  <li>linear for linear trends.</li>
  <li>poly for polynomial curves.</li>
  <li>sigmoid for neural-like boundaries.</li>
  <li>Scale features when using rbf, poly, or sigmoid kernels.</li>
</ul>


#### Iterated power (FactorAnalysis)

Power iterations used in randomized SVD.

<p>Higher values can improve approximation quality.</p>


#### Components (FactorAnalysis)

Number of latent factors.

<p>Use <b>None</b> to infer from input dimensionality.</p>


#### Random state (FactorAnalysis)

Seed for randomized solver components.

<p>Set for reproducible randomized runs.</p>


#### Rotation (FactorAnalysis)

Optional factor rotation for interpretability.

<p>Use <b>None</b> for unrotated factors, or <b>varimax/quartimax</b> for rotated solutions.</p>


#### SVD method (FactorAnalysis)

Backend used during factor estimation.

<p><b>randomized</b> is faster on large data; <b>lapack</b> is deterministic.</p>


#### Tolerance (FactorAnalysis)

Convergence tolerance.

<p>Lower values require stricter convergence.</p>


#### Algorithm (FastICA)

Parallel or deflation update strategy.

<p><b>parallel</b> estimates components together; <b>deflation</b> extracts one by one.</p>


#### Contrast function (FastICA)

Nonlinearity used to estimate non-Gaussian components.

<p><b>logcosh</b> is usually stable; try others when decomposition quality is poor.</p>


#### Max iterations (FastICA)

Maximum iterations before stopping.

<p>Increase if ICA fails to converge.</p>


#### Components (FastICA)

Number of independent components to estimate.

<p>Use <b>None</b> to infer based on feature count.</p>


#### Random state (FastICA)

Seed for ICA initialization.

<p>Fix this value for reproducible components.</p>


#### Tolerance (FastICA)

Stopping tolerance for ICA updates.

<p>Lower values enforce stricter convergence.</p>


#### Whiten mode (FastICA)

Whitening behavior before ICA optimization.

<p><b>unit-variance</b> is a robust default for regression pipelines.</p>


#### PCA components

How many principal components to keep.

<p>Use <b>None</b> to keep all components.</p>
<p>Lower values reduce dimensionality more aggressively.</p>


#### Random state (PCA)

Seed for randomized PCA behavior.

<p>Fix this value when using the randomized solver to reproduce results.</p>


#### PCA solver

Algorithm used to compute principal components.

<p><b>auto</b> picks a reasonable solver based on data shape.</p>
<p><b>full</b> uses a deterministic full SVD.</p>
<p><b>arpack</b> computes a truncated decomposition.</p>
<p><b>randomized</b> is faster for large datasets but stochastic.</p>


#### Whiten components

Scale components to unit variance.

<p>Whitening can help some models but may amplify noise.</p>
<p>Keep it off unless you have a specific reason to enable it.</p>


#### Max iterations (PLSRegression)

Maximum iterations for the NIPALS solver.

<p>Increase if convergence warnings occur.</p>


#### Components (PLSRegression)

Number of latent variables to keep.

<p>Higher values capture more signal but increase model complexity.</p>
<p>Keep this below the effective rank of your input data.</p>


#### Scale data (PLSRegression)

Standardize X and y inside the PLS step.

<p>Enable in most cases unless inputs are already consistently scaled.</p>


#### Tolerance (PLSRegression)

Convergence tolerance for iterative updates.

<p>Lower tolerance can improve precision but may require more iterations.</p>


#### Algorithm (TruncatedSVD)

Solver for truncated decomposition.

<p><b>randomized</b> is efficient for large matrices; <b>arpack</b> can be more precise.</p>


#### Components (TruncatedSVD)

Number of singular vectors to keep.

<p>Higher values preserve more information but reduce compression.</p>


#### Power iterations (TruncatedSVD)

Additional iterations for randomized solver accuracy.

<p>Increase when singular value gaps are small.</p>


#### Random state (TruncatedSVD)

Seed for randomized solver.

<p>Fix to make randomized decompositions reproducible.</p>


#### Tolerance (TruncatedSVD)

Convergence tolerance for ARPACK solver.

<p>Mostly relevant when using <b>arpack</b>.</p>


#### Number of trees (Extra Trees importance)

How many trees to build for importance scores.

<p>More trees give more stable importance estimates.</p>
<p>Extra Trees are noisy, so more estimators help stability.</p>


#### Random state (Extra Trees importance)

Seed for the importance estimator.

<p>Fix this value for repeatable importances.</p>
<p>Helps compare threshold values consistently.</p>


#### Importance threshold (Extra Trees)

Minimum importance required to keep a feature.

<p>Use <b>median</b>, <b>mean</b>, or a numeric value.</p>
<p>Higher thresholds keep only the most influential features.</p>


#### Number of estimators (GB importance)

How many estimators to build for importance scores.

<p>More estimators give more stable importance estimates.</p>
<p>Pair with smaller learning rates for stability.</p>


#### Random state (GB importance)

Seed for the importance estimator.

<p>Fix this value for repeatable importances.</p>
<p>Use when comparing thresholds across runs.</p>


#### Importance threshold (Gradient Boosting)

Minimum importance required to keep a feature.

<p>Use <b>median</b>, <b>mean</b>, or a numeric value.</p>
<p>Higher values prune more aggressive feature sets.</p>


#### Number of features (Mutual Info)

Pick the top K features by mutual information.

<p>Choose <b>All</b> to keep every feature.</p>
<p>Use smaller K to focus on the strongest nonlinear signals.</p>


#### Random state (Mutual Info)

Seed for mutual information estimation.

<p>Fix this value for repeatable scores.</p>
<p>Important when comparing ranking stability.</p>


#### Number of trees (RF importance)

How many trees to build for importance scores.

<p>More trees give more stable importance estimates.</p>
<p>Smaller values run faster but increase variance.</p>


#### Random state (RF importance)

Seed for the importance estimator.

<p>Fix this value for repeatable importances.</p>
<p>Useful when comparing thresholds.</p>


#### Importance threshold (Random Forest)

Minimum importance required to keep a feature.

<p>Use <b>median</b>, <b>mean</b>, or a numeric value.</p>
<p>Features below the threshold are removed from the dataset.</p>


#### Features to select (RFE)

Target number of features to keep.

<p>Use <b>None</b> to keep half the features by default.</p>
<p>Smaller targets yield more aggressive reduction.</p>


#### Random state (RFE)

Seed for randomized estimator behavior.

<p>Fix this value for repeatable selection.</p>
<p>Helps verify if rankings are stable across runs.</p>


#### Features removed per step (RFE)

How many features to eliminate each iteration.

<p>Smaller steps are more precise but take longer.</p>
<p>Use larger steps for faster but coarser selection.</p>


#### Number of features (Select K Best)

Pick the top K features by score.

<p>Choose <b>All</b> to keep every feature.</p>
<p>Lower values enforce more aggressive feature selection.</p>


#### Variance threshold

Remove features with variance below this threshold.

<p>Use higher values to drop low-variance features.</p>
<p>A threshold of 0 removes only constant features.</p>


#### Regression models

Pick the algorithms to evaluate for this experiment.

<p>Select one or more models to train on the same dataset and compare their results.</p>
<ul>
  <li>Toggle multiple options to run them in parallel; results appear in the runs table, predictions table, and charts.</li>
  <li>Model-specific settings are available under <b>Hyperparameters → Models</b>.</li>
  <li>For quick baselines keep a simple model (e.g., Linear Regression) alongside more complex methods.</li>
</ul>


#### FactorAnalysis

Latent-factor model that explains observed variables through shared factors plus noise.

<p><b>FactorAnalysis</b> models each feature as a combination of latent factors and feature-specific noise.</p>
<ul>
  <li>Useful for denoising and identifying compact latent structure.</li>
  <li>Common in domains where measurement noise is significant.</li>
</ul>


#### FastICA

Independent component analysis for separating non-Gaussian latent sources.

<p><b>FastICA</b> estimates statistically independent components instead of variance-maximizing ones.</p>
<ul>
  <li>Can help when signal sources are mixed and non-Gaussian.</li>
  <li>More sensitive to scaling and solver settings than PCA.</li>
</ul>


#### PCA

Linear projection that keeps directions with the highest variance.

<p><b>PCA</b> transforms correlated features into orthogonal components that preserve as much variance as possible.</p>
<ul>
  <li>Good baseline reducer for high-dimensional, correlated input sets.</li>
  <li>Improves numerical stability for some regressors.</li>
  <li>Component axes are not directly interpretable as original features.</li>
</ul>


#### PLSRegression

Supervised projection that uses both inputs and target to build latent variables.

<p><b>PLSRegression</b> finds latent components that maximize covariance between features and the target.</p>
<ul>
  <li>Useful when predictors are highly collinear and target signal is weak.</li>
  <li>Unlike PCA, it is supervised and can focus on prediction-relevant structure.</li>
</ul>


#### TruncatedSVD

Low-rank SVD projection that works well with sparse or large matrices.

<p><b>TruncatedSVD</b> projects features onto a smaller number of singular vectors without centering.</p>
<ul>
  <li>Efficient for large feature spaces and sparse data.</li>
  <li>Often used as a scalable alternative to PCA in high dimensions.</li>
</ul>


#### Target feature

Choose the numeric column you want the models to predict.

<p>Select exactly one <b>target feature</b>. The algorithms will try to predict this value from the input features.</p>
<ul>
  <li>Only one target can be active at a time; use the dropdown to switch quickly.</li>
  <li>Pick a continuous column (price, temperature, throughput) for best results.</li>
  <li>Changing the target automatically updates default stratification choices where possible.</li>
</ul>


#### Stratify bins

Set how many buckets to create when stratifying continuous values.

<p>When the stratify field contains continuous numbers, the values are bucketed into <b>bins</b> to enable stratified sampling.</p>
<ul>
  <li>Higher bin counts capture more detail but require larger datasets to keep buckets populated.</li>
  <li>Start with 5 bins for most targets; increase if you have plenty of records.</li>
  <li>Has no effect when stratifying by categorical features.</li>
</ul>


#### Hold-out test set

Reserve part of the data for final evaluation.

<p>Enable the <b>test set</b> to keep a portion of data untouched during training and cross-validation.</p>
<ul>
  <li>Disable when you want to use every row for cross-validation only.</li>
  <li>Keep it enabled to report unbiased metrics on unseen data.</li>
  <li>The reserved rows are excluded from model fitting.</li>
</ul>


#### Test size

Choose the fraction or fixed number of records set aside for testing.

<p>The <b>test size</b> sets how much of the dataset is held out.</p>
<ul>
  <li>Use fractions like 0.2 to reserve 20% of rows.</li>
  <li>Use integers like 200 to reserve a fixed number of rows.</li>
  <li>Typical fractional values range from 0.2 to 0.3 for balanced datasets.</li>
  <li>Use a smaller fraction when data is scarce, or larger when you expect heavy model tuning.</li>
  <li>Values outside the slider range can be typed directly when necessary.</li>
</ul>


#### Test split strategy

Control how rows are separated into train and test sets.

<p>Pick a strategy that matches the nature of your data.</p>
<ul>
  <li><b>Random</b>: shuffle rows before splitting into train/test according to the chosen size.</li>
  <li><b>Time ordered</b>: keep chronological order by placing the earliest rows in training and later rows in testing.</li>
  <li><b>Stratified</b>: preserve the distribution of the selected stratify feature in both sets.</li>
</ul>
<p>Strategies that rely on stratification will use the selected feature and binning settings below.</p>


#### Test stratification

Balance the test split using a target or helper feature.

<p>Select a feature to <b>stratify</b> the test split when using the Stratified strategy.</p>
<ul>
  <li>Use the target for imbalanced regression targets or choose another categorical field.</li>
  <li>Continuous targets are discretised into bins using the setting below before splitting.</li>
  <li>Leave empty to disable stratification even if the strategy allows it.</li>
</ul>


#### Regression

Build and evaluate regression models.

<p>The <b>Regression</b> tab guides you through training models for continuous targets.</p>
<ul>
  <li>Select algorithms, hyperparameters, and cross-validation settings.</li>
  <li>Train models on the current selection and preprocessing pipeline.</li>
  <li>Review metrics and compare runs to pick the best performer.</li>
</ul>
<p>Use this tab when predicting numeric outcomes such as prices, temperatures, or measurements.</p>


### Forecasting

#### Time Series Forecasting

Predict future values based on historical time series data.

<p><b>Forecasting</b> analyzes time-ordered data to predict future values.</p>
<p>Key concepts in time series forecasting:</p>
<ul>
  <li><b>Trend</b> - Long-term increase or decrease in the data</li>
  <li><b>Seasonality</b> - Regular patterns that repeat over fixed periods</li>
  <li><b>Lag Features</b> - Past values used to predict future ones</li>
  <li><b>Rolling Statistics</b> - Moving averages and other window-based features</li>
</ul>
<p>FastData uses <code>scikit-learn</code> models for forecasting, with manual time-series feature engineering and split strategies.</p>
<p>The previous <code>sktime</code> version is kept as a non-active reference in <code>backend/services/legacy_forecasting/forecasting_service_sktime.py</code>.</p>
<p>Common use cases: sales forecasting, demand prediction, financial projections.</p>


#### Forecast horizon

How many future time steps to predict in each run.

<p>The <b>horizon</b> sets how far ahead each model predicts.</p>
<ul>
  <li>Use a small horizon (e.g., 1–24 steps) for near-term monitoring.</li>
  <li>Larger horizons extend the forecast further but usually increase uncertainty.</li>
  <li>Match the value to your reporting cadence (steps follow the dataset's sampling).</li>
</ul>
<p>Models will generate this many points for every feature you selected.</p>


#### Use Box-Cox (BATS)

Apply a Box-Cox transform to stabilize variance.

<p>Helps when variability grows with the level of the series.</p>


#### Use damped trend (BATS)

Dampen the trend so it flattens over time.

<p>Useful when trends should level off rather than grow indefinitely.</p>


#### Use trend (BATS)

Include a trend component in the BATS model.

<p>Enable to capture long-term upward or downward movement.</p>


#### Smoothing (Croston)

Smoothing factor for intermittent demand forecasting.

<p>Controls how quickly the model reacts to new observations.</p>
<ul>
  <li>Lower values smooth more, higher values respond faster.</li>
</ul>


#### Seasonal (exponential smoothing)

Seasonality mode for the exponential smoothing model.

<p>Select how seasonal patterns are modeled.</p>
<ul>
  <li><b>add</b>: seasonality adds to the level.</li>
  <li><b>mul</b>: seasonality multiplies the level.</li>
  <li><b>None</b>: no seasonal component.</li>
</ul>


#### Seasonal period (exponential smoothing)

Number of steps in one seasonal cycle.

<p>Use the length of your repeating pattern to get better seasonal fits.</p>


#### Trend (exponential smoothing)

Type of trend component to include.

<p>Controls how the model extrapolates the long-term movement.</p>
<ul>
  <li><b>add</b>: linear trend (adds the trend each step).</li>
  <li><b>mul</b>: multiplicative trend (scales with level).</li>
  <li><b>None</b>: no explicit trend component.</li>
</ul>


#### Seasonal period (naive)

Length of the seasonality cycle used by the naive model.

<p>Defines how many time steps make up one season.</p>
<ul>
  <li>Use 7 for weekly seasonality on daily data, 12 for monthly seasonality on monthly data, etc.</li>
</ul>


#### Strategy (naive)

How the naive forecaster projects future values.

<p>Select the baseline rule used to forecast.</p>
<ul>
  <li><b>last</b>: repeats the last observed value.</li>
  <li><b>mean</b>: forecasts the historical mean.</li>
  <li><b>drift</b>: extends the overall linear trend.</li>
</ul>


#### Polynomial degree

Complexity of the polynomial trend.

<p>Higher degrees allow more curvature in the trend.</p>
<ul>
  <li>Start with a low degree to avoid overfitting.</li>
</ul>


#### Use Box-Cox (TBATS)

Apply a Box-Cox transform to stabilize variance.

<p>Improves modeling when variance changes with the level.</p>


#### Use damped trend (TBATS)

Dampen the trend component over time.

<p>Prevents overly aggressive trend extrapolation.</p>


#### Use trend (TBATS)

Include a trend component in the TBATS model.

<p>Enable to track long-term movement in the series.</p>


#### Deseasonalize (theta)

Remove seasonality before fitting the theta model.

<p>Enable when your series has strong seasonal patterns.</p>
<ul>
  <li>If disabled, the model fits directly on the original data.</li>
</ul>


#### Seasonal period (theta)

Season length used by the theta forecaster.

<p>Helps the model separate seasonality from trend when present.</p>
<ul>
  <li>Set to the number of observations per season.</li>
</ul>


#### Learning rate (time series gradient boosting)

Shrinkage applied to each boosting step.

<p>Smaller values require more estimators but often generalize better.</p>


#### Max depth (time series gradient boosting)

Depth of individual regression trees.

<p>Shallower trees generalize better on noisy data.</p>


#### Estimators (time series gradient boosting)

Number of boosting stages.

<p>More stages can improve accuracy but increase risk of overfitting.</p>


#### Random state (time series gradient boosting)

Seed for model training randomness.

<p>Set to ensure reproducible results across runs.</p>


#### Strategy (time series gradient boosting)

Multi-step forecasting method for gradient boosting.

<p>Choose between recursive, direct, or multioutput forecasting.</p>


#### Window length (time series gradient boosting)

Number of lagged steps used as input features.

<p>Increase to capture longer temporal patterns.</p>


#### Alpha (time series lasso)

Regularization strength for lasso regression.

<p>Higher alpha encourages sparse coefficients.</p>


#### Strategy (time series lasso)

Approach for multi-step forecasting.

<p>Recursive, direct, or multioutput behavior for generating horizons.</p>


#### Window length (time series lasso)

Number of lagged steps used as input features.

<p>Increase to capture longer temporal dependencies.</p>


#### Activation (time series MLP)

Activation function in hidden layers.

<p><b>relu</b> is a good default; <b>tanh</b> can smooth outputs.</p>


#### Alpha (time series MLP)

L2 regularization strength.

<p>Higher values apply more weight decay to reduce overfitting.</p>


#### Layers (time series MLP)

Hidden layer sizes for the MLP regressor.

<p>Enter comma-separated sizes, e.g. <b>64,32,16</b>.</p>
<p>More layers increase capacity but also training time.</p>


#### Learning rate (time series MLP)

Learning rate schedule.

<p><b>constant</b> keeps a fixed rate.</p>
<p><b>adaptive</b> reduces the rate when learning stalls.</p>


#### Max iterations (time series MLP)

Maximum training epochs.

<p>Increase if the model fails to converge.</p>


#### Random state (time series MLP)

Seed for weight initialization.

<p>Set to ensure reproducible results.</p>


#### Solver (time series MLP)

Optimizer for MLP training.

<p><b>adam</b> works well for most datasets.</p>
<p><b>lbfgs</b> can be faster on smaller datasets.</p>


#### Strategy (time series MLP)

How the MLP produces multi-step forecasts.

<p>Recursive, direct, or multioutput forecasting.</p>


#### Window length (time series MLP)

Number of lagged steps used as features.

<p>Longer windows capture more history but increase model size.</p>


#### Max depth (time series random forest)

Maximum depth of each tree.

<p>Use <b>None</b> to allow trees to expand fully.</p>


#### Estimators (time series random forest)

Number of trees in the forest.

<p>More trees improve stability but increase runtime.</p>


#### Random state (time series random forest)

Seed for the random forest training process.

<p>Set a fixed seed to make runs reproducible.</p>


#### Strategy (time series random forest)

How the model produces multi-step forecasts.

<p>Select recursive, direct, or multioutput forecasting.</p>


#### Window length (time series random forest)

Number of historical steps turned into features.

<p>Larger windows capture more history but can add noise.</p>


#### Alpha (time series ridge)

Regularization strength for ridge regression in time series.

<p>Higher alpha shrinks coefficients and reduces overfitting.</p>


#### Strategy (time series ridge)

How multi-step forecasts are generated.

<p>Choose between recursive, direct, or multioutput strategies.</p>
<ul>
  <li><b>recursive</b>: one-step model iterated forward.</li>
  <li><b>direct</b>: separate model per horizon step.</li>
  <li><b>multioutput</b>: single model predicts all steps.</li>
</ul>


#### Window length (time series ridge)

Number of past steps used as features.

<p>Longer windows capture more history but increase model size.</p>


#### Initial window size

Length of the first training window (auto when set to Auto).

<p>Controls how many observations are used before the first forecast window.</p>
<ul>
  <li><b>Auto</b> picks a reasonable size based on the horizon and model defaults.</li>
  <li>Provide a positive integer to force a specific training length.</li>
  <li>Use smaller windows for rapidly changing signals; larger windows for long-term patterns.</li>
</ul>
<p>This applies to sliding and expanding strategies; the single split uses the full training span.</p>


#### Forecasting models

Pick one or more algorithms to run for each selected feature.

<p>Select the models to include in the experiment. Each option offers different strengths:</p>
<ul>
  <li><b>Naive</b> – simple baselines using the last value, mean, or drift trend.</li>
  <li><b>Theta</b> – decomposes the series and extrapolates a flexible trend component.</li>
  <li><b>Exponential Smoothing</b> – configurable trend/seasonality smoothing (additive or multiplicative).</li>
  <li><b>Polynomial Trend</b> – fits a low-degree polynomial to capture smooth long-term movements.</li>
  <li><b>BATS</b> – Box-Cox transform with ARMA errors, trend, and seasonality support.</li>
  <li><b>TBATS</b> – BATS extended with multiple seasonal periods and trigonometric components.</li>
  <li><b>Croston</b> – designed for intermittent demand with sporadic non-zero observations.</li>
  <li><b>Time Series Ridge/Lasso Regression</b> – reduction to tabular regression with sliding windows.</li>
  <li><b>Time Series Random Forest</b> – ensemble of decision trees for non-linear relationships.</li>
  <li><b>Time Series Gradient Boosting</b> – boosted trees for strong accuracy on complex patterns.</li>
  <li><b>Time Series MLP</b> – neural network regressor for nonlinear temporal patterns.</li>
</ul>
<p>Run several models together to compare metrics and pick the best-performing approach.</p>


#### Target feature

Optional exogenous target for time series regression models.

<p>Select a target feature when using regression-based forecasters (ridge, lasso, random forest, gradient boosting).</p>
<ul>
  <li>Leave blank for univariate forecasts where each feature predicts itself.</li>
  <li>Choose a single target to train models that learn relationships from multiple inputs to one output.</li>
  <li>The target is included in preprocessing so scaling or filtering stays consistent.</li>
</ul>
<p>Only one target can be active at a time; deselect it to return to univariate mode.</p>


#### Window strategy

Pick how training and validation windows move through the series.

<p>Choose the cross-validation style used while fitting time-series models.</p>
<ul>
  <li><b>Single Train/Test Split</b> – one hold-out split; fastest baseline.</li>
  <li><b>Sliding Window</b> – fixed-size training window that slides forward to mimic rolling forecasts.</li>
  <li><b>Expanding Window</b> – training window grows over time to reuse all past observations.</li>
</ul>
<p>Sliding windows are good when concept drift is likely; expanding windows favor stability.</p>


#### Forecasting

Create time-series forecasts from historical data.

<p>The <b>Forecasting</b> tab focuses on time-aware modeling.</p>
<ul>
  <li>Configure horizons, frequency settings, and train/test splits for temporal data.</li>
  <li>Train forecasting models powered by scikit-learn and compare error metrics.</li>
  <li>Visualize predicted vs. actual values to validate performance.</li>
</ul>
<p>Choose this tab for scenarios like demand planning, capacity prediction, or financial projections.</p>


### Chat

#### Overview

Ask the built-in assistant and keep a record of conversations.

<p>The chat window lets you send questions to the assistant and keeps a history in the log database.</p>
<ul>
  <li>Chat messages are saved automatically with the log database.</li>
  <li>Clearing chat history removes conversations but keeps other log entries.</li>
  <li>Resetting the log database clears chat history too.</li>
</ul>


#### Clear history

Remove chat transcripts without deleting logs.

<p>Use the chat window to clear the current chat history.</p>
<ul>
  <li>This only removes chat messages; other logs remain.</li>
  <li>Resetting the log database clears both logs and chats.</li>
</ul>


#### OpenAI API key

Add your OpenAI key for ChatGPT/OpenAI access.

<p>This key is only required for the ChatGPT (OpenAI) provider.</p>
<p>Create or manage keys at <a href="https://platform.openai.com/api-keys">OpenAI API keys</a>.</p>
<p>If you do not have a key yet, sign in and generate one, then paste it here.</p>
<p>Use <b>Ask from AI</b> below if you want step-by-step guidance.</p>


#### Model name

Enter the model identifier for the selected provider.

<p>Provide the exact model name your provider expects.</p>
<ul>
  <li>OpenAI examples: <code>gpt-4o-mini</code>, <code>gpt-4o</code>, <code>gpt-3.5-turbo</code></li>
  <li>Ollama examples: <code>llama3.2</code>, <code>mistral</code>, <code>codellama</code></li>
</ul>
<p>Browse model lists:</p>
<ul>
  <li><a href="https://platform.openai.com/docs/models">OpenAI model catalog</a></li>
  <li><a href="https://ollama.com/library">Ollama model library</a></li>
</ul>
<p>You can click <b>Ask from AI</b> below for help choosing a model.</p>


#### LLM provider

Choose the backend that powers chat replies.

<p>Select which provider handles the chat requests sent from the chat window.</p>
<ul>
  <li><b>ChatGPT (OpenAI)</b> uses OpenAI-hosted models and requires an API key.</li>
  <li><b>Ollama (Local)</b> runs models on your machine without an API key.</li>
</ul>
<p>Helpful links:</p>
<ul>
  <li><a href="https://chat.openai.com/">ChatGPT</a> and <a href="https://platform.openai.com/">OpenAI API</a></li>
  <li><a href="https://ollama.com/">Ollama</a> and <a href="https://ollama.com/download">install instructions</a></li>
</ul>
<p>If you want more guidance, use <b>Ask from AI</b> below to send this topic to the chat window.</p>


