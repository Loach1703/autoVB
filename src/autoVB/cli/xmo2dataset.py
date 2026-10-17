import argparse
from pathlib import Path

from ..vbkit.xmo2dataset import xmo2dataset_file


def xmo2dataset(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export final VBSCF orbitals as JSON metadata and NPZ arrays.")
    parser.add_argument("xmo_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="new output directory; default: <stem>_dataset")
    args = parser.parse_args(argv)
    destination = xmo2dataset_file(args.xmo_file, args.output)
    print(f"Dataset written to: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(xmo2dataset())
