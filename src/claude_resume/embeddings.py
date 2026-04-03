"""Semantic search using TF-IDF vectors — no API keys needed.

Instead of calling an external embedding API, we build a TF-IDF matrix from
conversation summaries and use cosine similarity for search. This is fast,
free, and works offline.
"""

import json
import math
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from claude_resume.db import get_db

EMBEDDING_DIM = 256


def _serialize_embedding(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _deserialize_embedding(data: bytes) -> np.ndarray:
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "each", "few", "more", "most", "other", "some", "such", "no",
        "only", "own", "same", "than", "too", "very", "just", "because",
        "if", "when", "while", "that", "this", "these", "those", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their", "what", "which", "who",
        "whom", "how", "all", "any", "also", "about", "up", "there", "here",
        "type", "text", "command", "message", "name", "args",
    }
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in stopwords]


def _build_summary_text(row) -> str:
    """Build a text summary of a conversation for embedding."""
    parts = []
    if row["ai_title"]:
        parts.append(row["ai_title"])
        parts.append(row["ai_title"])
    if row["ai_tags"]:
        try:
            tags = json.loads(row["ai_tags"])
            parts.append(" ".join(tags))
            parts.append(" ".join(tags))
        except json.JSONDecodeError:
            pass
    if row["first_user_message"]:
        parts.append(row["first_user_message"][:500])
    if row["files_touched"]:
        try:
            files = json.loads(row["files_touched"])[:15]
            basenames = [Path(f).stem for f in files]
            parts.append(" ".join(basenames))
        except json.JSONDecodeError:
            pass
    if row["tools_used"]:
        try:
            tools = json.loads(row["tools_used"])[:10]
            parts.append(" ".join(tools))
        except json.JSONDecodeError:
            pass
    return " ".join(parts)


def _hash_token(token: str) -> int:
    """Deterministic hash of a token into embedding dimension range."""
    h = 0
    for c in token:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h % EMBEDDING_DIM


def _text_to_embedding(text: str) -> np.ndarray:
    """Convert text to a dense vector using hashed TF-IDF-like weighting."""
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    tf = Counter(tokens)
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    for token, count in tf.items():
        idx = _hash_token(token)
        weight = 1.0 + math.log(count) if count > 0 else 0
        sign = 1 if (hash(token) & 1) == 0 else -1
        vec[idx] += sign * weight

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


def generate_embeddings(project_filter: str | None = None, batch_size: int = 500, **_kwargs) -> int:
    """Generate TF-IDF embeddings for conversations that need them."""
    db = get_db()

    query = "SELECT session_id, project_path, ai_title, ai_tags, first_user_message, files_touched, tools_used FROM conversations WHERE embedding IS NULL"
    params = []
    if project_filter:
        query += " AND project_path = ?"
        params.append(project_filter)
    query += " ORDER BY file_mtime DESC LIMIT ?"
    params.append(batch_size)

    rows = db.execute(query, params).fetchall()
    if not rows:
        return 0

    generated = 0
    for row in rows:
        text = _build_summary_text(row)
        if not text:
            continue

        vec = _text_to_embedding(text)
        blob = _serialize_embedding(vec)
        db.execute(
            "UPDATE conversations SET embedding = ? WHERE session_id = ?",
            (blob, row["session_id"]),
        )
        generated += 1

    db.commit()
    db.close()
    return generated


def semantic_search(
    query: str,
    project_filter: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search conversations by vector similarity."""
    db = get_db()
    query_vec = _text_to_embedding(query)

    if np.linalg.norm(query_vec) == 0:
        db.close()
        return []

    sql = "SELECT * FROM conversations WHERE embedding IS NOT NULL"
    params = []
    if project_filter:
        sql += " AND project_path = ?"
        params.append(project_filter)

    rows = db.execute(sql, params).fetchall()
    db.close()

    if not rows:
        return []

    scored = []
    for row in rows:
        emb = _deserialize_embedding(row["embedding"])
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        similarity = float(np.dot(query_vec, emb))
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
            "similarity": similarity,
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]
