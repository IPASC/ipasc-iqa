"""ipasc_iqa CLI — Typer application."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="ipasc_iqa",
    help="Benchmarking framework for photoacoustic Image Quality Assessment.",
    no_args_is_help=True,
)

# Sub-command groups
list_app = typer.Typer(help="List available measures or datasets.", no_args_is_help=True)
manifest_app = typer.Typer(help="Manifest utilities.", no_args_is_help=True)

app.add_typer(list_app, name="list")
app.add_typer(manifest_app, name="manifest")

# ---------------------------------------------------------------------------
# list measures / datasets
# ---------------------------------------------------------------------------


@list_app.command("measures")
def list_measures(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """List all registered IQA measures."""
    from ipasc_iqa.core.registry import registry

    names = registry.list_measures()
    if json_out:
        import json

        rows = []
        for n in names:
            cls = registry.get(n)
            rows.append(
                {
                    "name": n,
                    "version": cls.metadata.version,
                    "full_reference": cls.metadata.full_reference,
                    "distribution": registry.distribution_for(n),
                }
            )
        typer.echo(json.dumps(rows, indent=2))
    else:
        typer.echo(f"{'Name':<20} {'Version':<10} {'Full-ref':<10} Distribution")
        typer.echo("-" * 60)
        for n in names:
            cls = registry.get(n)
            typer.echo(
                f"{n:<20} {cls.metadata.version:<10} "
                f"{'yes' if cls.metadata.full_reference else 'no':<10} "
                f"{registry.distribution_for(n)}"
            )


@list_app.command("datasets")
def list_datasets(
    root: str = typer.Option(".", "--root", help="Root directory to scan."),
) -> None:
    """List local dataset directories (those containing a manifest.json)."""
    from pathlib import Path

    p = Path(root)
    found = list(p.rglob("manifest.json"))
    if not found:
        typer.echo("No manifests found.")
        raise typer.Exit(1)
    for mf in found:
        import json

        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            name = data.get("dataset_name", "?")
            version = data.get("dataset_version", "?")
            n = len(data.get("images", []))
            typer.echo(f"{mf.parent}  [{name} v{version}, {n} images]")
        except Exception:
            typer.echo(f"{mf.parent}  [unreadable manifest]")


# ---------------------------------------------------------------------------
# manifest generate
# ---------------------------------------------------------------------------


@manifest_app.command("generate")
def manifest_generate(
    folder: str = typer.Argument(..., help="Dataset folder containing images/."),
    output: str = typer.Option("", "--output", help="Output path (default: <folder>/manifest.json)."),
    dataset_name: str = typer.Option("", "--name", help="Dataset name."),
    dataset_version: str = typer.Option("1.0.0", "--version", help="Dataset version string."),
) -> None:
    """Generate a manifest.json by scanning a dataset folder."""
    from pathlib import Path

    from ipasc_iqa.io.manifest import generate_manifest

    out = Path(output) if output else None
    generate_manifest(
        Path(folder),
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        output=out,
    )
    typer.echo("Manifest generated.")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@app.command()
def verify(
    dataset_path: str = typer.Argument(..., help="Path to a dataset folder."),
) -> None:
    """Verify dataset file integrity against its manifest."""
    from pathlib import Path

    from ipasc_iqa.datasets.local import LocalDataset

    ds = LocalDataset(Path(dataset_path))
    report = ds.verify()
    typer.echo(str(report))
    if not report.ok:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    dataset: list[str] = typer.Option(..., "--dataset", help="Dataset folder(s)."),
    measure: list[str] = typer.Option(..., "--measure", help="Measure name(s)."),
    output: str = typer.Option("results/run.json", "--output", "-o", help="Output JSON path."),
    reference_preference: str = typer.Option(
        "ground_truth,pseudo_reference",
        "--reference-preference",
        help="Comma-separated reference type preference order.",
    ),
    device: str = typer.Option("auto", "--device", help="Device: auto|cpu|cuda|mps."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Suppress progress bar."),
) -> None:
    """Run IQA evaluation on one or more datasets."""
    from pathlib import Path

    from ipasc_iqa.core.registry import registry
    from ipasc_iqa.core.runner import run_evaluation
    from ipasc_iqa.datasets.local import LocalDataset

    datasets = [LocalDataset(Path(d)) for d in dataset]
    measures = [registry.instantiate(m, device=device) for m in measure]
    ref_pref = [r.strip() for r in reference_preference.split(",") if r.strip()]

    run_evaluation(
        datasets,
        measures,
        reference_preference=ref_pref,
        output_path=Path(output),
        progress=not no_progress,
    )
    typer.echo(f"Results written to {output}")


# ---------------------------------------------------------------------------
# compare (stub for v0.1)
# ---------------------------------------------------------------------------


@app.command()
def compare(
    result_files: list[str] = typer.Argument(..., help="Result JSON files to compare."),
) -> None:
    """Compare two or more result JSON files (summary statistics)."""
    from pathlib import Path

    import numpy as np

    from ipasc_iqa.io.results import load_results

    all_runs = [load_results(Path(f)) for f in result_files]

    for run_res in all_runs:
        typer.echo(f"\n=== {run_res.run_id} ({run_res.finished}) ===")
        by_measure: dict[str, list[float]] = {}
        for rec in run_res.results:
            if rec.value is not None:
                by_measure.setdefault(rec.measure, []).append(rec.value)
            if rec.sub_scores:
                for k, v in rec.sub_scores.items():
                    by_measure.setdefault(f"{rec.measure}.{k}", []).append(v)

        typer.echo(f"  {'Measure':<25} {'N':>6} {'Mean':>10} {'Std':>10} {'Median':>10}")
        typer.echo("  " + "-" * 65)
        for name, vals in sorted(by_measure.items()):
            arr = np.array(vals)
            typer.echo(
                f"  {name:<25} {len(arr):>6} "
                f"{arr.mean():>10.4f} {arr.std():>10.4f} {np.median(arr):>10.4f}"
            )
        typer.echo(f"  Skipped: {len(run_res.skipped)}")
