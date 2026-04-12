#!/usr/bin/env python3
"""Resume test-split extraction only."""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

sys.path.insert(0, str(Path(__file__).parent))

from fase0.pre_embeddings import run_luna_patches

if __name__ == "__main__":
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"
    # Only extract test; train and val are already complete
    result = run_luna_patches(
        datasets_dir=datasets_dir,
        workers=6,
        neg_ratio=10,
        luna_subsets=None,
    )
    print("FINAL RESULT:", result)
