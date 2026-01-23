"""Command-line interface for DCTT.

This module provides the main CLI entry point for running DCTT
experiments and utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="dctt",
    help="Discrete-to-Continuous Transition Testing for LLM Embedding Geometry",
    add_completion=False,
)


@app.command()
def census(
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B",
        "--model", "-m",
        help="HuggingFace model name",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/census"),
        "--output", "-o",
        help="Output directory",
    ),
    k: int = typer.Option(
        50,
        "--k",
        help="Number of neighbors for kNN",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample",
        help="Sample size (None = full vocab)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
) -> None:
    """Run diagnostic census on model embeddings."""
    console.print(f"[bold blue]DCTT Census[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Output: {output_dir}")

    # Import here to avoid slow startup
    from dctt.tracking.reproducibility import set_all_seeds
    set_all_seeds(seed)

    console.print("\n[yellow]Census experiment not yet implemented in CLI.[/yellow]")
    console.print("Run: python experiments/run_census.py")


@app.command()
def repair(
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B",
        "--model", "-m",
        help="HuggingFace model name",
    ),
    diagnostics_path: Path = typer.Option(
        ...,
        "--diagnostics", "-d",
        help="Path to diagnostic results",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/repair"),
        "--output", "-o",
        help="Output directory",
    ),
    top_n: int = typer.Option(
        100,
        "--top-n",
        help="Number of tokens to repair",
    ),
) -> None:
    """Repair high-severity token embeddings."""
    console.print(f"[bold blue]DCTT Repair[/bold blue]")
    console.print(f"Diagnostics: {diagnostics_path}")
    console.print(f"Top N: {top_n}")

    console.print("\n[yellow]Repair experiment not yet implemented in CLI.[/yellow]")
    console.print("Run: python experiments/run_causal_repair.py")


@app.command()
def extract(
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B",
        "--model", "-m",
        help="HuggingFace model name",
    ),
    output_path: Path = typer.Option(
        Path("outputs/embeddings/embeddings.npy"),
        "--output", "-o",
        help="Output file path",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device to use (cpu, mps, cuda)",
    ),
) -> None:
    """Extract embeddings from a model."""
    console.print(f"[bold blue]DCTT Extract Embeddings[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Output: {output_path}")

    from dctt.embeddings import extract_embeddings, get_embedding_info
    from dctt.embeddings.normalize import normalize_embeddings
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("\n[cyan]Extracting embeddings...[/cyan]")

    try:
        embeddings, tokenizer = extract_embeddings(
            model_name=model,
            device=device,
        )

        console.print(f"Raw embeddings shape: {embeddings.shape}")

        # Normalize
        embeddings_norm, norms = normalize_embeddings(embeddings, return_norms=True)

        # Save
        np.save(output_path, embeddings_norm)
        np.save(output_path.with_suffix(".norms.npy"), norms)

        # Get info
        info = get_embedding_info(embeddings_norm, model)
        console.print(f"\n[green]Success![/green]")
        console.print(f"Vocab size: {info.vocab_size}")
        console.print(f"Embedding dim: {info.embedding_dim}")
        console.print(f"Hash: {info.hash}")
        console.print(f"Saved to: {output_path}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info() -> None:
    """Show DCTT version and environment info."""
    from dctt import __version__
    from dctt.tracking.reproducibility import get_package_versions

    console.print(f"[bold]DCTT v{__version__}[/bold]")
    console.print("\n[cyan]Package versions:[/cyan]")

    versions = get_package_versions()
    for pkg, version in versions.items():
        console.print(f"  {pkg}: {version}")


if __name__ == "__main__":
    app()
