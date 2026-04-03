# claude-resume

AI-powered conversation search for [Claude Code](https://github.com/anthropics/claude-code).

Claude Code stores your conversations as `.jsonl` files in `~/.claude/projects/`, but the built-in `/resume` only lets you search by random slugs and raw first messages. **claude-resume** makes them searchable with AI-generated titles, topic tags, and semantic search.

## Install

```bash
# From GitHub
pipx install git+https://github.com/aayushgandhi/claude-resume.git

# Or from source
git clone https://github.com/aayushgandhi/claude-resume.git
cd claude-resume
pip install .
```

## Quick Start

```bash
# One-time setup: index conversations + generate titles + build embeddings
cresume bootstrap --all-projects

# Search your conversations
cresume search "auth middleware bug"
cresume search -s "that session where I fixed the 500 errors"  # semantic search
cresume list                                                    # recent conversations
cresume resume "auth"                                           # search + pick + resume
```

## Prerequisites

- **Python 3.11+**
- **Claude Code** installed and authenticated (used for AI title generation via `claude -p`)

No API keys needed — titles are generated through your existing Claude Code auth, embeddings are computed locally with TF-IDF.

## Commands

| Command | Description |
|---------|-------------|
| `cresume bootstrap` | Full setup: index + titles + embeddings |
| `cresume index` | Scan and index conversation files |
| `cresume titles` | Generate AI titles via Claude Code |
| `cresume embed` | Generate local TF-IDF embeddings |
| `cresume search <query>` | Text search across titles, tags, messages, files |
| `cresume search -s <query>` | Semantic similarity search |
| `cresume list` | List recent conversations |
| `cresume resume [query]` | Interactive search + resume |
| `cresume stats` | Show index statistics |

## Options

Most commands support:
- `--all-projects` — Search/index all projects (default: auto-detects current project from cwd)
- `-p <project>` — Target a specific project
- `-n <limit>` — Limit results

## How It Works

1. **Indexer** scans `~/.claude/projects/*/*.jsonl` files and extracts metadata (slug, messages, tools used, files touched) into a SQLite index at `~/.claude/resume_index.db`
2. **Title generator** batches conversations and calls `claude -p --model claude-haiku-4-5-20251001` to generate 5-10 word titles and topic tags
3. **Embeddings** use hashed TF-IDF vectors (no external API) for instant semantic similarity search
4. **Search** combines weighted text matching across titles, tags, messages, and file paths
