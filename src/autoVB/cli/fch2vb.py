import argparse
from pathlib import Path

from ..vbkit.fch2vb import fch2vb_impl


def fch2vb(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fch2vb", description="Convert Gaussian .fch orbitals to a minimal XMVB .xmi initial guess.")
    parser.add_argument("fch", type=Path, help="Gaussian formatted checkpoint file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output .xmi path")
    parser.add_argument("--basis", default="", help="basis text written to the XMVB header")
    parser.add_argument("--norb", type=int, default=None, help="number of orbitals written to the initial guess, default occupied orbitals")
    args = parser.parse_args(argv)

    fch2vb_impl(args.fch, args.output, args.basis, args.norb)
    return 0


if __name__ == "__main__":
    raise SystemExit(fch2vb())
