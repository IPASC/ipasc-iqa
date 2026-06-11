"""Evaluation orchestration — the single entry point for CLI and GUI."""

from __future__ import annotations

import platform
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from ipasc_iqa.core.measure import BaseMeasure
from ipasc_iqa.core.result import (
    DatasetMeta,
    EnvironmentInfo,
    MeasureMeta,
    ResultRecord,
    RunConfig,
    RunResults,
    SkippedRecord,
)

if TYPE_CHECKING:
    from ipasc_iqa.core.cache import ResultStore
    from ipasc_iqa.core.dataset import Dataset


def _make_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


def _env_info() -> EnvironmentInfo:
    torch_ver: str | None = None
    cuda_ver: str | None = None
    try:
        import torch
        torch_ver = torch.__version__
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
    except ImportError:
        pass
    return EnvironmentInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch_ver,
        cuda=cuda_ver,
    )


def _pick_reference(
    record_references: dict,
    measure: BaseMeasure,
    reference_preference: list[str] | None,
) -> tuple[object, str | None]:
    """Return (array_or_None, reference_type_or_None)."""
    if not measure.metadata.full_reference:
        return None, None

    accepted = measure.metadata.accepted_reference_types

    def _acceptable(rtype: str) -> bool:
        return accepted is None or rtype in accepted

    # Try preference order first
    if reference_preference:
        for rtype in reference_preference:
            if rtype in record_references and _acceptable(rtype):
                return record_references[rtype], rtype

    # Fall back to any available acceptable reference
    for rtype, arr in record_references.items():
        if _acceptable(rtype):
            return arr, rtype

    return None, None


def run_evaluation(
    datasets: list[Dataset],
    measures: list[BaseMeasure],
    *,
    reference_preference: list[str] | None = None,
    output_path: Path | None = None,
    cache: ResultStore | None = None,
    progress: bool = True,
) -> RunResults:
    from ipasc_iqa import __version__
    from ipasc_iqa.core.cache import make_cache_key
    from ipasc_iqa.core.registry import registry

    run_id = _make_run_id()
    started = datetime.now(UTC)

    dataset_metas = []
    for ds in datasets:
        dataset_metas.append(
            DatasetMeta(
                name=ds.name,
                version=ds.version,
                manifest_sha256=ds.manifest_sha256,
                n_images=len(ds),
            )
        )

    measure_metas = []
    for m in measures:
        dist = registry.distribution_for(m.metadata.name)
        measure_metas.append(
            MeasureMeta(
                name=m.metadata.name,
                version=m.metadata.version,
                plugin_distribution=dist,
                metadata={
                    "full_reference": m.metadata.full_reference,
                    "higher_is_better": m.metadata.higher_is_better,
                },
            )
        )

    results: list[ResultRecord] = []
    skipped: list[SkippedRecord] = []

    total = sum(len(ds) for ds in datasets) * len(measures)
    pbar = tqdm(total=total, disable=not progress, desc="Evaluating")

    for ds in datasets:
        for record in ds:
            for measure in measures:
                pbar.update(1)

                # Check ROI requirements
                if measure.metadata.requires_signal_roi and record.signal_roi is None:
                    skipped.append(
                        SkippedRecord(
                            dataset=ds.name,
                            image_id=record.id,
                            measure=measure.metadata.name,
                            reason="signal_roi required but not available",
                        )
                    )
                    continue

                if measure.metadata.requires_noise_roi and record.noise_roi is None:
                    skipped.append(
                        SkippedRecord(
                            dataset=ds.name,
                            image_id=record.id,
                            measure=measure.metadata.name,
                            reason="noise_roi required but not available",
                        )
                    )
                    continue

                reference, reference_type = _pick_reference(
                    record.references, measure, reference_preference
                )

                if measure.metadata.full_reference and reference is None:
                    skipped.append(
                        SkippedRecord(
                            dataset=ds.name,
                            image_id=record.id,
                            measure=measure.metadata.name,
                            reason="no reference of accepted type available",
                        )
                    )
                    continue

                ref_sha = record.hashes.references.get(reference_type) if reference_type else None

                # Cache lookup: skip recomputation when inputs are unchanged.
                cache_key = make_cache_key(
                    measure_name=measure.metadata.name,
                    measure_version=measure.metadata.version,
                    image_sha256=record.hashes.image,
                    reference_type=reference_type,
                    reference_sha256=ref_sha,
                    signal_roi_sha256=record.hashes.signal_roi,
                    noise_roi_sha256=record.hashes.noise_roi,
                    measure_config_json=measure.config_canonical_json(),
                )
                if cache is not None:
                    cached = cache.get(cache_key)
                    if cached is not None:
                        cached.from_cache = True
                        results.append(cached)
                        continue

                t0 = time.perf_counter()
                try:
                    result = measure.evaluate(
                        record.image,
                        reference=reference,
                        reference_type=reference_type,
                        signal_roi=record.signal_roi,
                        noise_roi=record.noise_roi,
                    )
                except Exception as exc:
                    skipped.append(
                        SkippedRecord(
                            dataset=ds.name,
                            image_id=record.id,
                            measure=measure.metadata.name,
                            reason=f"evaluation error: {exc}",
                        )
                    )
                    continue
                wall_time = time.perf_counter() - t0

                if isinstance(result, dict):
                    value = None
                    sub_scores = result
                else:
                    value = float(result)
                    sub_scores = None

                record_out = ResultRecord(
                    dataset=ds.name,
                    image_id=record.id,
                    image_sha256=record.hashes.image,
                    measure=measure.metadata.name,
                    measure_version=measure.metadata.version,
                    reference_type=reference_type,
                    reference_sha256=ref_sha,
                    signal_roi_sha256=record.hashes.signal_roi,
                    noise_roi_sha256=record.hashes.noise_roi,
                    value=value,
                    sub_scores=sub_scores,
                    wall_time_s=wall_time,
                    device=measure.device,
                )
                if cache is not None:
                    cache.set(cache_key, record_out)
                results.append(record_out)

    pbar.close()
    finished = datetime.now(UTC)

    run = RunResults(
        run_id=run_id,
        tool={"name": "ipasc_iqa", "version": __version__},
        environment=_env_info(),
        started=started,
        finished=finished,
        config=RunConfig(
            reference_preference=reference_preference,
            cache_enabled=cache is not None,
        ),
        datasets=dataset_metas,
        measures=measure_metas,
        results=results,
        skipped=skipped,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    return run
