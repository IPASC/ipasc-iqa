"""Reference type constants."""

GROUND_TRUTH = "ground_truth"
PSEUDO_REFERENCE = "pseudo_reference"
EXPERT = "expert"

CANONICAL_TYPES: frozenset[str] = frozenset({GROUND_TRUTH, PSEUDO_REFERENCE, EXPERT})
