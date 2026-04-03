"""Text-based search across the conversation index."""

import re

from claude_resume.db import get_db


def text_search(
    query: str,
    project_filter: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Search conversations by text matching on title, slug, first message, tags, and files."""
    db = get_db()
    terms = query.lower().split()

    sql = """
        SELECT *, 0 as score FROM conversations
        WHERE 1=1
    """
    params = []
    if project_filter:
        sql += " AND project_path = ?"
        params.append(project_filter)

    sql += " ORDER BY file_mtime DESC"
    rows = db.execute(sql, params).fetchall()
    db.close()

    scored = []
    for row in rows:
        score = _score_match(row, terms)
        if score > 0:
            scored.append({
                "session_id": row["session_id"],
                "slug": row["slug"],
                "ai_title": row["ai_title"],
                "ai_tags": row["ai_tags"],
                "first_user_message": row["first_user_message"],
                "first_timestamp": row["first_timestamp"],
                "last_timestamp": row["last_timestamp"],
                "message_count": row["message_count"],
                "file_mtime": row["file_mtime"],
                "score": score,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def _score_match(row, terms: list[str]) -> float:
    """Score how well a conversation matches the search terms."""
    score = 0.0
    fields = {
        "ai_title": 5.0,
        "slug": 2.0,
        "first_user_message": 3.0,
        "ai_tags": 4.0,
        "files_touched": 1.5,
        "tools_used": 1.0,
    }

    for field, weight in fields.items():
        value = row[field]
        if not value:
            continue
        value_lower = value.lower()

        for term in terms:
            if term in value_lower:
                score += weight
                if re.search(rf"\b{re.escape(term)}\b", value_lower):
                    score += weight * 0.5

    combined = " ".join(str(row[f] or "") for f in fields).lower()
    if not all(term in combined for term in terms):
        return 0.0

    return score


def list_recent(
    project_filter: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List most recent conversations."""
    db = get_db()
    sql = "SELECT * FROM conversations WHERE 1=1"
    params = []
    if project_filter:
        sql += " AND project_path = ?"
        params.append(project_filter)
    sql += " ORDER BY file_mtime DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    db.close()

    return [dict(row) for row in rows]
