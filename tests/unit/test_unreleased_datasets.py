"""The unreleased Alladien studies must not request placeholder URLs."""
from pathlib import Path
from unittest.mock import patch

import pytest

from ipasc_iqa.datasets.alladien import (
    DatasetNotPublicError,
    DavoudiDataset,
    OADATDataset,
    EFADataset,
    PHIREDataset,
    UNRELEASED_MESSAGE,
)
from ipasc_iqa.gui.jobs import _JobState


@pytest.mark.parametrize("adapter", [DavoudiDataset, OADATDataset, EFADataset, PHIREDataset])
def test_unreleased_dataset_skips_without_network(tmp_path: Path, adapter) -> None:
    with patch("ipasc_iqa.datasets.alladien.urllib.request.urlretrieve") as download:
        with pytest.raises(DatasetNotPublicError, match="not yet publicly available") as error:
            adapter(tmp_path / adapter.DATASET_NAME)
    assert str(error.value) == UNRELEASED_MESSAGE
    download.assert_not_called()


def test_unreleased_dataset_can_use_existing_manifest(tmp_path: Path) -> None:
    import json
    root = tmp_path / "davoudi"
    converted = root / "converted"
    converted.mkdir(parents=True)
    (converted / "manifest.json").write_text(json.dumps({"dataset_name": "davoudi", "images": []}))
    with patch("ipasc_iqa.datasets.alladien.urllib.request.urlretrieve") as download:
        adapter = DavoudiDataset(root)
    assert adapter.name == "davoudi"
    download.assert_not_called()


def test_unavailable_dataset_is_not_reported_as_failed() -> None:
    job = _JobState("job")
    job.init_preparation(["davoudi"])
    job.update_preparation("davoudi", "skipped", UNRELEASED_MESSAGE)
    job.record_dataset_unavailable("davoudi", UNRELEASED_MESSAGE)
    snapshot = job.snapshot()
    assert snapshot["preparation"][0]["state"] == "skipped"
    assert snapshot["dataset_errors"] == []
    assert snapshot["dataset_skips"][0]["reason"] == UNRELEASED_MESSAGE


def test_publication_flag_controls_download_independently_of_url(tmp_path: Path) -> None:
    """A released adapter attempts download even when hosted at a non-Zenodo URL."""
    with (
        patch.object(DavoudiDataset, "SOURCE_PUBLICLY_AVAILABLE", True),
        patch("ipasc_iqa.datasets.alladien.ZENODO_BASE", "https://example.org/data"),
        patch("ipasc_iqa.datasets.alladien.urllib.request.urlretrieve", side_effect=OSError("network unavailable")) as download,
    ):
        with pytest.raises(OSError, match="network unavailable"):
            DavoudiDataset(tmp_path / "davoudi")
    assert download.call_args.args[0].startswith("https://example.org/data/")
