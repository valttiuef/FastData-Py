CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    level INTEGER NOT NULL,
    logger_name TEXT NOT NULL,
    origin TEXT NOT NULL,
    message TEXT NOT NULL,
    formatted TEXT NOT NULL
);
