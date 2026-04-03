#!/usr/bin/env python3
"""claude-resume: AI-powered conversation search for Claude Code."""

import json
import os
import subprocess
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from claude_resume.db import get_db
from claude_resume.indexer import index_all, index_project, PROJECTS_DIR
from claude_resume.search import text_search, list_recent

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_project_path() -> str | None:
    """Derive the Claude Code project path key for the current working directory."""
    cwd = os.getcwd()
    return cwd.replace("/", "-").lstrip("-")


def _format_time(ts: str | float | None) -> str:
    if ts is None:
        return ""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts)
    else:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return str(ts)[:16]
    now = datetime.now(dt.tzinfo)
    delta = now - dt
    if delta.days == 0:
        hours = delta.seconds // 3600
        if hours == 0:
            mins = delta.seconds // 60
            return f"{mins}m ago" if mins > 0 else "just now"
        return f"{hours}h ago"
    if delta.days == 1:
        return "yesterday"
    if delta.days < 7:
        return f"{delta.days}d ago"
    return dt.strftime("%b %d")


def _format_tags(tags_json: str | None) -> str:
    if not tags_json:
        return ""
    try:
        tags = json.loads(tags_json)
        return " ".join(f"[dim]#{t}[/dim]" for t in tags)
    except (json.JSONDecodeError, TypeError):
        return ""


def _render_results(results: list[dict], show_score: bool = False):
    """Render conversation results as a rich table."""
    if not results:
        console.print("[yellow]No conversations found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, show_lines=True, expand=True)
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Title / First Message", ratio=4)
    table.add_column("Slug", style="cyan", ratio=1)
    table.add_column("When", style="green", width=10)
    table.add_column("Msgs", style="yellow", width=5, justify="right")
    if show_score:
        table.add_column("Score", style="magenta", width=6, justify="right")

    for i, r in enumerate(results):
        title = r.get("ai_title") or ""
        first_msg = r.get("first_user_message") or ""

        if title:
            title_text = f"[bold]{title}[/bold]"
            if first_msg:
                truncated = first_msg[:80] + ("..." if len(first_msg) > 80 else "")
                title_text += f"\n[dim]{truncated}[/dim]"
        else:
            truncated = first_msg[:120] + ("..." if len(first_msg) > 120 else "")
            title_text = truncated or "[dim]<empty>[/dim]"

        tags = _format_tags(r.get("ai_tags"))
        if tags:
            title_text += f"\n{tags}"

        row = [
            str(i + 1),
            title_text,
            r.get("slug") or "",
            _format_time(r.get("last_timestamp") or r.get("file_mtime")),
            str(r.get("message_count") or 0),
        ]
        if show_score:
            score_val = r.get("score") or r.get("similarity") or 0
            row.append(f"{score_val:.2f}")

        table.add_row(*row)

    console.print(table)
    console.print()
    console.print("[dim]Resume a conversation:[/dim]")
    console.print("[bold]  claude --resume <slug>[/bold]")
    console.print("[dim]  or: claude --resume <session_id>[/dim]")


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """AI-powered conversation search for Claude Code."""
    pass


@cli.command()
@click.option("--force", is_flag=True, help="Re-index all conversations even if unchanged")
@click.option("--project", "-p", default=None, help="Only index a specific project")
def index(force: bool, project: str | None):
    """Build/update the conversation index."""
    console.print("[bold]Indexing conversations...[/bold]")

    if project:
        project_dir = PROJECTS_DIR / project
        if not project_dir.exists():
            console.print(f"[red]Project directory not found: {project_dir}[/red]")
            return
        indexed, skipped = index_project(project_dir, force=force)
        console.print(f"  {project}: [green]{indexed} indexed[/green], [dim]{skipped} unchanged[/dim]")
    else:
        stats = index_all(force=force)
        total_indexed = 0
        total_skipped = 0
        for _proj, s in stats.items():
            total_indexed += s["indexed"]
            total_skipped += s["skipped"]
        console.print(f"[green]{total_indexed} conversations indexed[/green], [dim]{total_skipped} unchanged[/dim] across [cyan]{len(stats)} projects[/cyan]")


@cli.command()
@click.option("--project", "-p", default=None, help="Only generate for a specific project (auto-detects current)")
@click.option("--batch-size", "-n", default=50, help="Max conversations to process")
@click.option("--all-projects", is_flag=True, help="Generate titles for all projects, not just current")
def titles(project: str | None, batch_size: int, all_projects: bool):
    """Generate AI titles using Claude Code CLI (no API key needed)."""
    from claude_resume.titles import generate_titles

    if not project and not all_projects:
        project = _current_project_path()
        if project:
            console.print(f"[dim]Auto-detected project: {project}[/dim]")

    console.print("[bold]Generating AI titles via Claude Code...[/bold]")
    count = generate_titles(project_filter=project, batch_size=batch_size)
    console.print(f"[green]{count} titles generated[/green]")


@cli.command()
@click.option("--project", "-p", default=None, help="Only generate for a specific project")
@click.option("--batch-size", "-n", default=500, help="Max conversations to process")
@click.option("--all-projects", is_flag=True, help="Generate embeddings for all projects")
def embed(project: str | None, batch_size: int, all_projects: bool):
    """Generate TF-IDF embeddings for semantic search (local, no API key needed)."""
    from claude_resume.embeddings import generate_embeddings

    if not project and not all_projects:
        project = _current_project_path()
        if project:
            console.print(f"[dim]Auto-detected project: {project}[/dim]")

    console.print("[bold]Generating embeddings...[/bold]")
    count = generate_embeddings(project_filter=project, batch_size=batch_size)
    console.print(f"[green]{count} embeddings generated[/green]")


@cli.command(name="search")
@click.argument("query", nargs=-1, required=True)
@click.option("--semantic", "-s", is_flag=True, help="Use semantic (AI) search instead of text search")
@click.option("--project", "-p", default=None, help="Limit to specific project")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--all-projects", is_flag=True, help="Search all projects, not just current")
def search_cmd(query: tuple, semantic: bool, project: str | None, limit: int, all_projects: bool):
    """Search conversations by keyword or semantic similarity."""
    query_str = " ".join(query)

    if not project and not all_projects:
        project = _current_project_path()

    if semantic:
        from claude_resume.embeddings import semantic_search
        console.print(f"[bold]Semantic search:[/bold] [cyan]{query_str}[/cyan]")
        results = semantic_search(query_str, project_filter=project, limit=limit)
    else:
        console.print(f"[bold]Text search:[/bold] [cyan]{query_str}[/cyan]")
        results = text_search(query_str, project_filter=project, limit=limit)

    _render_results(results, show_score=True)


@cli.command(name="list")
@click.option("--project", "-p", default=None, help="Limit to specific project")
@click.option("--limit", "-n", default=20, help="Max results")
@click.option("--all-projects", is_flag=True, help="List from all projects")
def list_cmd(project: str | None, limit: int, all_projects: bool):
    """List recent conversations."""
    if not project and not all_projects:
        project = _current_project_path()

    results = list_recent(project_filter=project, limit=limit)
    _render_results(results)


@cli.command()
@click.option("--project", "-p", default=None, help="Limit to specific project")
@click.option("--all-projects", is_flag=True, help="Show stats for all projects")
def stats(project: str | None, all_projects: bool):
    """Show index statistics."""
    db = get_db()

    if not project and not all_projects:
        project = _current_project_path()

    where = ""
    params = []
    if project:
        where = "WHERE project_path = ?"
        params = [project]

    row = db.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN ai_title IS NOT NULL AND ai_title != '' THEN 1 ELSE 0 END) as titled,
            SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded,
            SUM(message_count) as total_messages,
            SUM(file_size) as total_size,
            MIN(first_timestamp) as earliest,
            MAX(last_timestamp) as latest
        FROM conversations {where}
    """, params).fetchone()

    if not row or row["total"] == 0:
        console.print("[yellow]No indexed conversations found. Run 'index' first.[/yellow]")
        db.close()
        return

    panel_text = f"""[bold]Total conversations:[/bold] {row['total']}
[bold]With AI titles:[/bold] {row['titled']} ({100 * (row['titled'] or 0) / row['total']:.0f}%)
[bold]With embeddings:[/bold] {row['embedded']} ({100 * (row['embedded'] or 0) / row['total']:.0f}%)
[bold]Total messages:[/bold] {row['total_messages']}
[bold]Total storage:[/bold] {(row['total_size'] or 0) / 1024 / 1024:.1f} MB
[bold]Date range:[/bold] {(row['earliest'] or 'N/A')[:10]} to {(row['latest'] or 'N/A')[:10]}"""

    if project:
        panel_text = f"[dim]Project: {project}[/dim]\n\n" + panel_text

    console.print(Panel(panel_text, title="Index Statistics", border_style="blue"))
    db.close()


@cli.command()
@click.option("--project", "-p", default=None, help="Limit to specific project")
@click.option("--batch-size", "-n", default=50, help="Max conversations per step")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation")
@click.option("--all-projects", is_flag=True, help="Bootstrap all projects")
@click.option("--skip-titles", is_flag=True, help="Skip AI title generation")
def bootstrap(project: str | None, batch_size: int, skip_embeddings: bool, all_projects: bool, skip_titles: bool):
    """Full setup: index + generate titles + generate embeddings. No API keys needed."""
    if not project and not all_projects:
        project = _current_project_path()
        if project:
            console.print(f"[dim]Auto-detected project: {project}[/dim]")

    # Step 1: Index
    console.print("\n[bold]Step 1/3: Indexing conversations...[/bold]")
    if project:
        project_dir = PROJECTS_DIR / project
        if not project_dir.exists():
            console.print(f"[red]Project directory not found: {project_dir}[/red]")
            return
        indexed, skipped = index_project(project_dir)
        console.print(f"  [green]{indexed} indexed[/green], [dim]{skipped} unchanged[/dim]")
    else:
        stats_data = index_all()
        total = sum(s["indexed"] for s in stats_data.values())
        console.print(f"  [green]{total} conversations indexed[/green]")

    # Step 2: Titles
    if skip_titles:
        console.print("\n[bold]Step 2/3: Skipping titles (--skip-titles)[/bold]")
    else:
        console.print("\n[bold]Step 2/3: Generating AI titles via Claude Code...[/bold]")
        from claude_resume.titles import generate_titles
        count = generate_titles(project_filter=project, batch_size=batch_size)
        console.print(f"  [green]{count} titles generated[/green]")

    # Step 3: Embeddings
    if skip_embeddings:
        console.print("\n[bold]Step 3/3: Skipping embeddings (--skip-embeddings)[/bold]")
    else:
        console.print("\n[bold]Step 3/3: Generating local embeddings...[/bold]")
        from claude_resume.embeddings import generate_embeddings
        count = generate_embeddings(project_filter=project, batch_size=batch_size)
        console.print(f"  [green]{count} embeddings generated[/green]")

    console.print("\n[bold green]Bootstrap complete![/bold green]")
    console.print("[dim]Run 'cresume search' or 'cresume list' to explore your conversations.[/dim]")


@cli.command(name="resume")
@click.argument("query", nargs=-1)
@click.option("--semantic", "-s", is_flag=True, help="Use semantic search")
@click.option("--project", "-p", default=None, help="Limit to specific project")
@click.option("--all-projects", is_flag=True, help="Search all projects")
def resume_cmd(query: tuple, semantic: bool, project: str | None, all_projects: bool):
    """Search and directly resume a conversation in Claude Code.

    Without a query, shows recent conversations. With a query, searches and
    lets you pick one to resume.
    """
    if not project and not all_projects:
        project = _current_project_path()

    if query:
        query_str = " ".join(query)
        if semantic:
            from claude_resume.embeddings import semantic_search
            results = semantic_search(query_str, project_filter=project, limit=10)
        else:
            results = text_search(query_str, project_filter=project, limit=10)
    else:
        results = list_recent(project_filter=project, limit=15)

    if not results:
        console.print("[yellow]No conversations found.[/yellow]")
        return

    _render_results(results, show_score=bool(query))

    console.print()
    choice = click.prompt("Enter number to resume (or 'q' to quit)", default="q")
    if choice.lower() == "q":
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            session_id = results[idx]["session_id"]
            slug = results[idx].get("slug")
            resume_key = slug or session_id
            console.print(f"\n[bold]Resuming:[/bold] {results[idx].get('ai_title') or resume_key}")
            subprocess.run(["claude", "--resume", resume_key], check=False)
        else:
            console.print("[red]Invalid selection.[/red]")
    except ValueError:
        console.print("[red]Invalid input.[/red]")


if __name__ == "__main__":
    cli()
