"""Generate AI titles for conversations using Claude Code as the AI backend."""

import json
import subprocess

from claude_resume.db import get_db
from claude_resume.indexer import parse_conversation, PROJECTS_DIR


def _ask_claude(prompt: str) -> str | None:
    """Use the `claude` CLI to generate a response (no API key needed)."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "claude-haiku-4-5-20251001", prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def generate_titles(project_filter: str | None = None, batch_size: int = 20) -> int:
    """Generate AI titles for conversations that don't have one.

    Uses Claude Code CLI as the backend — no API key needed.
    Returns number of titles generated.
    """
    db = get_db()

    query = "SELECT session_id, project_path, first_user_message FROM conversations WHERE (ai_title IS NULL OR ai_title = '')"
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
    batch = []
    for row in rows:
        session_id = row["session_id"]
        first_msg = row["first_user_message"] or ""

        project_dir = PROJECTS_DIR / row["project_path"]
        jsonl_path = project_dir / f"{session_id}.jsonl"

        user_messages = []
        if jsonl_path.exists():
            try:
                meta = parse_conversation(jsonl_path)
                user_messages = meta.get("user_messages_for_summary", [])
            except Exception:
                pass

        if not user_messages and first_msg:
            user_messages = [first_msg]

        if not user_messages:
            continue

        batch.append({
            "session_id": session_id,
            "messages": user_messages,
        })

    total_batches = (len(batch) + 9) // 10
    for batch_num, i in enumerate(range(0, len(batch), 10)):
        sub_batch = batch[i : i + 10]
        print(f"  Batch {batch_num + 1}/{total_batches} ({len(sub_batch)} conversations)...", flush=True)
        conversations_text = ""
        for idx, item in enumerate(sub_batch):
            msgs = "\n".join(f"  - {m[:200]}" for m in item["messages"][:5])
            conversations_text += f"\n[{idx}] Session {item['session_id'][:8]}:\n{msgs}\n"

        prompt = f"""Generate a short, descriptive title (5-10 words) for each coding conversation below. Also generate 1-3 topic tags per conversation.

Reply ONLY with a JSON array where each element is: {{"index": <number>, "title": "<title>", "tags": ["tag1", "tag2"]}}

No markdown, no explanation, just the JSON array.

Conversations:
{conversations_text}"""

        response = _ask_claude(prompt)
        if not response:
            print(f"  Failed to get response for batch starting at {i}")
            continue

        try:
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            results = json.loads(text)

            for result in results:
                idx = result["index"]
                if 0 <= idx < len(sub_batch):
                    session_id = sub_batch[idx]["session_id"]
                    title = result.get("title", "")
                    tags = json.dumps(result.get("tags", []))
                    db.execute(
                        "UPDATE conversations SET ai_title = ?, ai_tags = ? WHERE session_id = ?",
                        (title, tags, session_id),
                    )
                    generated += 1

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  Failed to parse response: {e}")
            for item in sub_batch:
                title = _generate_single_title(item["messages"])
                if title:
                    db.execute(
                        "UPDATE conversations SET ai_title = ? WHERE session_id = ?",
                        (title, item["session_id"]),
                    )
                    generated += 1

    db.commit()
    db.close()
    return generated


def _generate_single_title(messages: list[str]) -> str | None:
    """Fallback: generate title for a single conversation."""
    msgs_text = "\n".join(f"- {m[:200]}" for m in messages[:5])
    prompt = f"Generate a short title (5-10 words) for this coding conversation. Reply with ONLY the title, no quotes, no explanation.\n\nUser messages:\n{msgs_text}"
    response = _ask_claude(prompt)
    if response:
        return response.strip().strip('"').strip("'")
    return None
