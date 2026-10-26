"""Export a standalone, NumPy-readable orbital training sample."""

import json
from pathlib import Path

import numpy as np

from ..io.xmo_dataset import XmoDatasetReader


def xmo2dataset_file(xmo_file: str | Path, output: str | Path | None = None) -> Path:
    """Export final VBSCF coefficients and metadata to a new sample directory."""
    source = Path(xmo_file)
    destination = Path(output) if output is not None else source.with_name(source.stem + "_dataset")
    metadata, arrays = XmoDatasetReader(source).read()
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "arrays.npz", **arrays)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination
