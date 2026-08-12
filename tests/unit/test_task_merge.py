"""Auto-merging annotations that ended up under a duplicate/absent task name."""

from __future__ import annotations

from ipasc_iqa.gui.service import canonical_task_map, manifest_task_names


def _comparison_only_manifest() -> dict:
    """A published 2AFC dataset: names its task only in ``human_comparisons``."""
    return {
        "images": [{"id": "i0", "metadata": {"stem": "s0"}}],
        "human_comparisons": [{"task": "overall_quality", "reference": "s0", "comparisons": []}],
    }


def test_comparison_only_dataset_still_names_its_task() -> None:
    """Regression: reading only metadata.annotations made these look task-less,
    so contributions landed on task=None and showed up as a separate bucket."""
    assert manifest_task_names(_comparison_only_manifest()) == ["overall_quality"]


def test_unnamed_and_synonym_contributions_fold_onto_the_manifest_task() -> None:
    mapping = canonical_task_map(_comparison_only_manifest())
    assert mapping[None] == "overall_quality"
    assert mapping["overall"] == "overall_quality"
    assert "overall_quality" not in mapping  # the canonical name is left alone


def test_mos_and_comparison_task_names_are_unioned() -> None:
    manifest = {
        "images": [{"id": "i0", "metadata": {"annotations": {"edge_visibility": {"mos": 3}}}}],
        "human_comparisons": [{"task": "overall_quality", "comparisons": []}],
    }
    assert manifest_task_names(manifest) == ["edge_visibility", "overall_quality"]


def test_nothing_is_merged_when_the_intent_is_ambiguous() -> None:
    """Several declared tasks → an unnamed annotation could have meant any of them."""
    manifest = {
        "images": [
            {
                "id": "i0",
                "metadata": {
                    "annotations": {"overall_quality": {"mos": 3}, "contrast": {"mos": 2}}
                },
            }
        ]
    }
    mapping = canonical_task_map(manifest)
    assert None not in mapping
    # Synonyms are still unambiguous: only one member of the group is declared.
    assert mapping["overall"] == "overall_quality"


def test_synonym_group_with_two_declared_names_is_left_alone() -> None:
    """If the manifest itself uses both names, we cannot pick a winner for it."""
    manifest = {
        "images": [
            {
                "id": "i0",
                "metadata": {"annotations": {"overall": {"mos": 3}, "overall_quality": {"mos": 2}}},
            }
        ]
    }
    assert canonical_task_map(manifest) == {}
