"""
PocketRAG - Production CLI Application
"""
import typer
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from pocketrag.core.indexer import Indexer
from pocketrag.core.search import Searcher
from pocketrag.core.chat import ChatEngine
from pocketrag.config import config

app = typer.Typer(
    name="pocketrag",
    help="⚡ PocketRAG: Lightning fast, local AI document engine.",
    add_completion=False,
    pretty_exceptions_enable=False,
)

console = Console()


@app.command()
def init():
    """Initialize the PocketRAG database."""
    console.print("[bold blue]🚀 Initializing PocketRAG...[/bold blue]")
    try:
        config.ensure_db_dir()
        indexer = Indexer()
        console.print(f"[green]✅ Database initialized at: {config.db_path}[/green]")
        console.print("[dim]💡 Next step: Run 'pocketrag add <directory>' to index documents[/dim]")
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def add(
    path: Path = typer.Argument(..., help="Path to the directory containing your documents."),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Search subdirectories recursively."),
):
    """Add a directory of documents to the index."""
    if not path.exists():
        console.print(f"[red]❌ Directory not found: {path}[/red]")
        raise typer.Exit(1)
    
    if not path.is_dir():
        console.print(f"[red]❌ Not a directory: {path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold green]📚 Indexing documents from: {path}[/bold green]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Indexing...", total=None)  # Indeterminate spinner
            
            indexer = Indexer()
            stats = indexer.index_directory(str(path), recursive=recursive)
        
        console.print("\n[bold green]✅ Indexing complete![/bold green]")
        console.print(f"   • Files processed: {stats['files_processed']}")
        console.print(f"   • Files skipped: {stats['files_skipped']}")
        console.print(f"   • Chunks created: {stats['chunks_created']}")
        if "chunks_replaced" in stats:
            console.print(f"   • Chunks replaced: {stats['chunks_replaced']}")
        console.print(f"   • Errors: {stats['errors']}")
        
    except Exception as e:
        console.print(f"[red]❌ Indexing failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    model: str = typer.Option(config.default_model, "--model", "-m", help="The Ollama model to use."),
    top_k: int = typer.Option(config.default_top_k, "--top-k", "-k", help="Number of documents to retrieve."),
):
    """Start an interactive chat session with your documents."""
    try:
        engine = ChatEngine(model_name=model, top_k=top_k)
        
        # Check if documents are indexed
        if not engine.is_ready():
            console.print("[yellow]⚠️  No documents indexed yet.[/yellow]")
            console.print("[dim]Run 'pocketrag add <directory>' first to index documents.[/dim]")
            console.print()
        
        console.print(f"[bold cyan]🤖 Chatting with {model}[/bold cyan]")
        console.print("[dim]Type 'quit' or 'exit' to end the session[/dim]")
        console.print()
        
        while True:
            try:
                query = typer.prompt("You")
                
                if query.lower() in ["quit", "exit"]:
                    console.print("[blue]👋 Goodbye![/blue]")
                    break
                
                console.print()
                console.print("[bold green]Assistant:[/bold green]")
                
                # Stream the response
                for chunk in engine.stream_chat(query):
                    console.print(chunk, end="", markup=False)
                
                console.print("\n")
                
            except typer.Abort:
                console.print("\n[blue]👋 Goodbye![/blue]")
                break
            except KeyboardInterrupt:
                console.print("\n[blue]👋 Goodbye![/blue]")
                break
            except Exception as e:
                console.print(f"\n[red]❌ Error: {e}[/red]\n")
                
    except Exception as e:
        console.print(f"[red]❌ Failed to start chat: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(config.default_top_k, "--top-k", "-k", help="Number of results to return."),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Search mode (vector, fts, hybrid)."),
):
    """Search your indexed documents."""
    try:
        searcher = Searcher()
        
        if not searcher.is_indexed():
            console.print("[yellow]⚠️  No documents indexed yet.[/yellow]")
            console.print("[dim]Run 'pocketrag add <directory>' first.[/dim]")
            raise typer.Exit(1)
        
        console.print(f"[bold green]🔍 Searching for: {query}[/bold green]\n")
        
        results = searcher.search(query, top_k=top_k, mode=mode)
        
        if not results:
            console.print("[yellow]No relevant documents found.[/yellow]")
            return
        
        for i, result in enumerate(results, 1):
            console.print(f"[bold cyan]Result {i}[/bold cyan]")
            console.print(f"[dim]Source: {result.source}[/dim]")
            console.print(f"[dim]Score: {result.score:.4f}[/dim]")
            console.print(f"[white]{result.text[:500]}{'...' if len(result.text) > 500 else ''}[/white]")
            console.print()
            
    except Exception as e:
        console.print(f"[red]❌ Search failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show the current status of the PocketRAG database."""
    console.print("[bold blue]📊 PocketRAG Status[/bold blue]\n")
    
    try:
        searcher = Searcher()
        
        if searcher.is_indexed():
            count = searcher.count()
            console.print(f"[green]✅ Database is ready[/green]")
            console.print(f"   • Indexed chunks: {count}")
            console.print(f"   • Database path: {config.db_path}")
        else:
            console.print("[yellow]⚠️  No documents indexed yet[/yellow]")
            console.print(f"   • Database path: {config.db_path}")
            console.print("\n[dim]Run 'pocketrag add <directory>' to index documents.[/dim]")
            
    except Exception as e:
        console.print(f"[red]❌ Status check failed: {e}[/red]")


@app.command()
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="The config key to get or set."),
    value: Optional[str] = typer.Argument(None, help="The value to set."),
):
    """Get or set configuration values."""
    persisted_keys = set(config.to_dict().keys())

    if key is None:
        console.print("[bold blue]🛠️  PocketRAG Configuration[/bold blue]")
        for k, v in config.to_dict().items():
            console.print(f"   • {k}: [green]{v}[/green]")
        return

    if not hasattr(config, key) or isinstance(getattr(type(config), key, None), property):
        console.print(f"[red]❌ Invalid or read-only config key: {key}[/red]")
        return

    if key not in persisted_keys:
        console.print(
            f"[yellow]⚠️  '{key}' is runtime-only and is not persisted to config.json.[/yellow]"
        )

    if value is None:
        console.print(f"{key}: [green]{getattr(config, key)}[/green]")
    else:
        # Try to cast value to the correct type
        current_val = getattr(config, key)
        try:
            if isinstance(current_val, bool):
                casted_value = value.lower() in ["true", "1", "yes", "on", "enable"]
            elif isinstance(current_val, int):
                casted_value = int(value)
            elif isinstance(current_val, float):
                casted_value = float(value)
            elif isinstance(current_val, tuple):
                casted_value = tuple(part.strip() for part in value.split(",") if part.strip())
            else:
                casted_value = value

            setattr(config, key, casted_value)
            if key in persisted_keys:
                config.save()
            console.print(f"[green]✅ Set {key} to {casted_value}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to set {key}: {e}[/red]")


@app.command()
def clear(confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation.")):
    """Clear all indexed documents."""
    if not confirm:
        typer.confirm("Are you sure you want to clear all indexed documents?", abort=True)
    
    try:
        indexer = Indexer()
        indexer.clear()
        console.print("[green]✅ All documents cleared.[/green]")
    except Exception as e:
        console.print(f"[red]❌ Clear failed: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    config.load()
    app()


if __name__ == "__main__":
    main()
