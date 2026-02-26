CREATE INDEX IF NOT EXISTS idx_selection_settings_name ON selection_settings(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_selection_settings_active ON selection_settings(is_active);
