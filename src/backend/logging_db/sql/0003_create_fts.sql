CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
    message,
    formatted,
    logger_name,
    origin,
    content=''
);
