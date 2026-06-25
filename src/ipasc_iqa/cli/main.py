"""ipasc_iqa CLI — Typer application."""

from __future__ import annotations

import typer

from ipasc_iqa.core.measure import MeasureType

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
                    "measure_type": cls.metadata.measure_type,
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
                f"{'yes' if cls.metadata.measure_type == MeasureType.FULL_REFERENCE else 'no':<10} "
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
    output: str = typer.Option(
        "", "--output", help="Output path (default: <folder>/manifest.json)."
    ),
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
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON."),
    require_gui: bool = typer.Option(
        False, "--gui", help="Treat missing GUI dependencies as errors."
    ),
    require_deep: bool = typer.Option(
        False, "--deep", help="Treat missing deep-learning dependencies as errors."
    ),
) -> None:
    """Check whether the current environment can run ipasc_iqa."""
    import json

    from ipasc_iqa.core.doctor import has_errors, run_doctor

    results = run_doctor(require_gui=require_gui, require_deep=require_deep)

    if json_out:
        typer.echo(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        typer.echo(f"{'Group':<8} {'Check':<24} {'Status':<8} {'Version':<32} Message")
        typer.echo("-" * 100)
        for result in results:
            version = result.version or "-"
            typer.echo(
                f"{result.group:<8} {result.name:<24} {result.status.upper():<8} "
                f"{version:<32} {result.message}"
            )
            if result.suggestion and result.status != "ok":
                typer.echo(f"{'':<8} {'':<24} {'':<8} {'':<32} Suggestion: {result.suggestion}")

    if has_errors(results):
        raise typer.Exit(1)


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

@app.command()
def show(
    result_file: str = typer.Argument(
        ..., help="Result JSON file to inspect."
    ),
    top: int = typer.Option(
        5,
        "--top",
        "-t",
        help="Number of best/worst examples to show per measure.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Restrict the summary to one dataset.",
    ),
) -> None:
    """Inspect a single result JSON file."""

    from pathlib import Path

    from ipasc_iqa.analysis.aggregate import (
        measure_directions,
        rich_measure_summary,
    )
    from ipasc_iqa.io.results import load_results

    run_res = load_results(Path(result_file))

    tool_name = run_res.tool.get("name", "?")
    tool_version = run_res.tool.get("version", "?")

    typer.echo(f"\n=== Run: {run_res.run_id} ===")
    typer.echo(f"Started:  {run_res.started}")
    typer.echo(f"Finished: {run_res.finished}")
    typer.echo(f"Tool:     {tool_name} {tool_version}")

    typer.echo("\nDatasets:")
    for ds in run_res.datasets:
        typer.echo(
            f"  - {ds.name} v{ds.version} "
            f"({ds.n_images} images)"
        )

    directions = measure_directions(run_res.measures)

    typer.echo("\nMeasures:")
    for measure in run_res.measures:
        higher_is_better = directions.get(measure.name, True)
        direction = (
            "higher is better"
            if higher_is_better
            else "lower is better"
        )
        typer.echo(
            f"  - {measure.name} v{measure.version} "
            f"({direction})"
        )

    all_records = list(run_res.results)
    all_skipped = list(run_res.skipped)

    if dataset is not None:
        dataset_names = [dataset]
    else:
        dataset_names = sorted({rec.dataset for rec in all_records} | {rec.dataset for rec in all_skipped})

    if not dataset_names:
        typer.echo("\nNo result records found.")
        raise typer.Exit(code=0)

    typer.echo("\nRun summary:")
    typer.echo(f"  Results: {len(all_records)}")
    typer.echo(f"  Skipped: {len(all_skipped)}")
    if dataset is not None:
        typer.echo(f"  Dataset filter: {dataset}")

    for dataset_name in dataset_names:
        records = [rec for rec in all_records if rec.dataset == dataset_name]
        skipped = [rec for rec in all_skipped if rec.dataset == dataset_name]

        if dataset is not None and not records and not skipped:
            typer.echo(f"\nNo results found for dataset: {dataset_name}")
            raise typer.Exit(code=1)

        typer.echo(f"\n\n=== Dataset: {dataset_name} ===")
        typer.echo(f"Results: {len(records)}")
        typer.echo(f"Skipped: {len(skipped)}")

        if not records:
            typer.echo("No successful metric results for this dataset.")
            if skipped:
                _print_skipped(skipped)
            continue

        summary = rich_measure_summary(
            records,
            higher_is_better_by_measure=directions,
            top=top,
        )

        _print_metric_statistics(summary)
        _print_best_worst_examples(summary)

        if skipped:
            _print_skipped(skipped)

def _print_metric_statistics(summary: dict) -> None:
    """Print per-measure statistics produced by rich_measure_summary."""

    typer.echo("\nStatistics:")
    typer.echo(
        f"{'Measure':<25}"
        f"{'N':>8}"
        f"{'Mean':>12}"
        f"{'Std':>12}"
        f"{'Median':>12}"
        f"{'Min':>12}"
        f"{'Max':>12}"
    )
    typer.echo("-" * 93)

    for stat in summary["statistics"]:
        typer.echo(
            f"{stat['measure']:<25}"
            f"{stat['n']:>8}"
            f"{stat['mean']:>12.4f}"
            f"{stat['std']:>12.4f}"
            f"{stat['median']:>12.4f}"
            f"{stat['min']:>12.4f}"
            f"{stat['max']:>12.4f}"
        )


def _print_best_worst_examples(summary: dict) -> None:
    """Print best/worst examples produced by rich_measure_summary."""

    for measure_name, groups in summary["examples"].items():
        direction = (
            "higher is better"
            if groups["higher_is_better"]
            else "lower is better"
        )

        typer.echo(f"\n--- {measure_name} ---")
        typer.echo(f"Direction: {direction}")

        typer.echo("\nWorst examples:")
        for row in groups["worst"]:
            typer.echo(
                f"  {row['value']:>10.4f}  "
                f"{row['dataset']}/{row['image_id']}"
            )

        typer.echo("\nBest examples:")
        for row in groups["best"]:
            typer.echo(
                f"  {row['value']:>10.4f}  "
                f"{row['dataset']}/{row['image_id']}"
            )


def _print_skipped(skipped: list) -> None:
    """Print skipped result records."""

    typer.echo("\nSkipped:")
    for rec in skipped[:10]:
        typer.echo(
            f"  {rec.dataset}/{rec.image_id} | "
            f"{rec.measure} | {rec.reason}"
        )

    if len(skipped) > 10:
        typer.echo(f"  ... and {len(skipped) - 10} more")