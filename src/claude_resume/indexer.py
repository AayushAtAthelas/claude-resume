"""Scan .jsonl conversation files and build the metadata index."""

import json
from collections import Counter
from pathlib import Path

from claude_resume.db import get_db, needs_reindex

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"


def discover_projects() -> list[Path]:
    """Find all project directories under ~/.claude/projects/."""
    if not PROJECTS_DIR.exists():
        return []
    return [p for p in PROJECTS_DIR.iterdir() if p.is_dir()]


def discover_conversations(project_dir: Path) -> list[Path]:
    """Find all .jsonl conversation files in a project directory."""
    return list(project_dir.glob("*.jsonl"))


def parse_conversation(jsonl_path: Path) -> dict:
    """Extract metadata from a conversation .jsonl file."""
    session_id = jsonl_path.stem
    slug = None
    first_user_message = None
    first_timestamp = None
    last_timestamp = None
    message_count = 0
    user_message_count = 0
    tools = Counter()
    files_touched = set()
    user_messages_for_summary = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type")
            timestamp = obj.get("timestamp")

            if timestamp:
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp

            if "slug" in obj and obj["slug"]:
                slug = obj["slug"]

            if msg_type in ("user", "assistant"):
                message_count += 1

            if msg_type == "user":
                user_message_count += 1
                message = obj.get("message", {})
                content = message.get("content", "")
                text = _extract_text(content)

                if not first_user_message and text:
                    if not obj.get("toolUseResult") and not obj.get("isMeta"):
                        first_user_message = text[:500]

                if text and len(user_messages_for_summary) < 10:
                    if not obj.get("toolUseResult") and not obj.get("isMeta"):
                        user_messages_for_summary.append(text[:300])

            if msg_type == "assistant":
                message = obj.get("message", {})
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            tools[tool_name] += 1
                            tool_input = block.get("input", {})
                            for key in ("file_path", "path", "file"):
                                if key in tool_input and isinstance(tool_input[key], str):
                                    files_touched.add(tool_input[key])

    top_tools = [t for t, _ in tools.most_common(20)]
    project_path = jsonl_path.parent.name

    return {
        "session_id": session_id,
        "slug": slug,
        "project_path": project_path,
        "first_user_message": first_user_message,
        "message_count": message_count,
        "user_message_count": user_message_count,
        "tools_used": json.dumps(top_tools),
        "files_touched": json.dumps(sorted(files_touched)[:50]),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "file_size": jsonl_path.stat().st_size,
        "file_mtime": jsonl_path.stat().st_mtime,
        "user_messages_for_summary": user_messages_for_summary,
    }


def _extract_text(content) -> str:
    """Extract plain text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def index_project(project_dir: Path, force: bool = False) -> tuple[int, int]:
    """Index all conversations in a project. Returns (indexed, skipped) counts."""
    db = get_db()
    conversations = discover_conversations(project_dir)
    indexed = 0
    skipped = 0

    for jsonl_path in conversations:
        session_id = jsonl_path.stem
        file_mtime = jsonl_path.stat().st_mtime

        if not force and not needs_reindex(db, session_id, file_mtime):
            skipped += 1
            continue

        try:
            meta = parse_conversation(jsonl_path)
        except Exception as e:
            print(f"  Error parsing {jsonl_path.name}: {e}")
            continue

        db.execute(
            """
            INSERT OR REPLACE INTO conversations
            (session_id, slug, project_path, first_user_message,
             message_count, user_message_count, tools_used, files_touched,
             first_timestamp, last_timestamp, file_size, file_mtime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta["session_id"],
                meta["slug"],
                meta["project_path"],
                meta["first_user_message"],
                meta["message_count"],
                meta["user_message_count"],
                meta["tools_used"],
                meta["files_touched"],
                meta["first_timestamp"],
                meta["last_timestamp"],
                meta["file_size"],
                meta["file_mtime"],
            ),
        )
        indexed += 1

    db.commit()
    db.close()
    return indexed, skipped


def index_all(force: bool = False) -> dict:
    """Index all projects. Returns stats per project."""
    stats = {}
    for project_dir in discover_projects():
        indexed, skipped = index_project(project_dir, force=force)
        if indexed > 0 or skipped > 0:
            stats[project_dir.name] = {"indexed": indexed, "skipped": skipped}
    return stats
