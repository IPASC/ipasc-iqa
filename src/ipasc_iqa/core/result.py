"""Result data models (Pydantic)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SkippedRecord(BaseModel):
    dataset: str
    image_id: str
    measure: str
    reason: str


class ResultRecord(BaseModel):
    dataset: str
    image_id: str
    image_sha256: str
    measure: str
    measure_version: str
    reference_type: str | None
    reference_sha256: str | None = None
    signal_roi_sha256: str | None
    noise_roi_sha256: str | None
    value: float | None
    sub_scores: dict[str, float] | None
    wall_time_s: float
    device: str
    from_cache: bool = False


class DatasetMeta(BaseModel):
    name: str
    version: str
    manifest_sha256: str
    n_images: int


class MeasureMeta(BaseModel):
    name: str
    version: str
    plugin_distribution: str
    metadata: dict[str, Any]


class EnvironmentInfo(BaseModel):
    python: str
    platform: str
    torch: str | None = None
    cuda: str | None = None


class RunConfig(BaseModel):
    reference_preference: list[str] | None = None
    cache_enabled: bool = False


class RunResults(BaseModel):
    schema_version: str = "1.0.0"
    run_id: str
    tool: dict[str, str] = Field(default_factory=lambda: {"name": "ipasc_iqa", "version": "0.1.0"})
    environment: EnvironmentInfo
    started: datetime
    finished: datetime | None = None
    config: RunConfig
    datasets: list[DatasetMeta] = Field(default_factory=list)
    measures: list[MeasureMeta] = Field(default_factory=list)
    results: list[ResultRecord] = Field(default_factory=list)
    skipped: list[SkippedRecord] = Field(default_factory=list)
