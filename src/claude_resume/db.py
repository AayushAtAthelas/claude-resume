"""SQLite database for conversation index."""

import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".claude" / "resume_index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,
    slug TEXT,
    project_path TEXT,
    first_user_message TEXT,
    ai_title TEXT,
    ai_tags TEXT,  -- JSON array of tags
    message_count INTEGER,
    user_message_count INTEGER,
    tools_used TEXT,  -- JSON array
    files_touched TEXT,  -- JSON array
    first_timestamp TEXT,
    last_timestamp TEXT,
    file_size INTEGER,
    file_mtime REAL,
    embedding BLOB  -- numpy array bytes
);

CREATE INDEX IF NOT EXISTS idx_project ON conversations(project_path);
CREATE INDEX IF NOT EXISTS idx_mtime ON conversations(file_mtime DESC);
CREATE INDEX IF NOT EXISTS idx_title ON conversations(ai_title);
"""


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def needs_reindex(db: sqlite3.Connection, session_id: str, file_mtime: float) -> bool:
    """Check if a conversation needs re-indexing based on file modification time."""
    row = db.execute(
        "SELECT file_mtime FROM conversations WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return True
    return file_mtime > row["file_mtime"]
