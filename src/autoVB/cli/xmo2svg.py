import argparse
from pathlib import Path

from ..vbkit.xmo2svg import (
    DEFAULT_XMO_ACTIVE_SPACE_COLOR,
    DEFAULT_XMO_ACTIVE_SPACE_WIDTH,
    DEFAULT_XMO_MAX_STRUCTURES,
    DEFAULT_XMO_STRUCTURES_PER_ROW,
    DEFAULT_XMO_WEIGHT_TABLE,
    xmo2svg_file,
    xmo2svg_report_lines,
)


def parse_draw_xmo_max_structures(value: str) -> int | None:
    normalized_value = value.strip().lower()
    if normalized_value == "all":
        return None

    try:
        max_structures = int(normalized_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--max-structures must be a positive integer or 'all'."
        ) from exc
    if max_structures <= 0:
        raise argparse.ArgumentTypeError(
            "--max-structures must be a positive integer or 'all'."
        )
    return max_structures


def parse_draw_xmo_structures_per_row(value: str) -> int:
    try:
        structures_per_row = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--structures-per-row must be a positive integer."
        ) from exc
    if structures_per_row <= 0:
        raise argparse.ArgumentTypeError(
            "--structures-per-row must be a positive integer."
        )
    return structures_per_row


def xmo2svg(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="xmo2svg",
        description=(
            "Read an XMO file and generate valence-bond SVG files using "
            "$orb labels for molecular connectivity."
        ),
    )
    parser.add_argument("xmo_file", help="input .xmo file")
    parser.add_argument(
        "--connectivity",
        choices=("orb", "rdkit"),
        default="orb",
        help="connectivity source: $orb labels or RDKit bond perception, default: orb",
    )
    parser.add_argument(
        "--weight",
        "-w",
        choices=("lowdin", "cc", "inverse", "renormalized", "both"),
        default=DEFAULT_XMO_WEIGHT_TABLE,
        help=(
            "weight table to display; both uses Lowdin weights for selection "
            f"and displays both values, default: {DEFAULT_XMO_WEIGHT_TABLE}"
        ),
    )
    parser.add_argument(
        "--max-structures",
        "-m",
        type=parse_draw_xmo_max_structures,
        default=DEFAULT_XMO_MAX_STRUCTURES,
        help=(
            "maximum number of highest-weight structures to draw; "
            f"use all to draw every structure, default: {DEFAULT_XMO_MAX_STRUCTURES}"
        ),
    )
    parser.add_argument(
        "--baseline-index",
        type=int,
        default=None,
        help=(
            "structure index used as the initial electron distribution; "
            "the highest-weight structure is used by default"
        ),
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="total charge used by RDKit when assigning bond orders, default: 0",
    )
    parser.add_argument(
        "--projection",
        choices=("rdkit", "pca", "optimized3d", "contact"),
        default="rdkit",
        help="atom layout method, default: rdkit",
    )
    parser.add_argument(
        "--show-hydrogens",
        action="store_true",
        help="show hydrogen atoms; hydrogens are hidden by default",
    )
    parser.add_argument(
        "--no-condensed-hydrogens",
        action="store_false",
        dest="condense_hydrogens",
        help="do not show hidden hydrogens as compact heteroatom or isolated-C labels",
    )
    parser.add_argument(
        "--write-individual-svgs",
        action="store_true",
        help="write one SVG per structure in addition to the grid SVG",
    )
    parser.add_argument(
        "--structures-per-row",
        "-n",
        type=parse_draw_xmo_structures_per_row,
        default=DEFAULT_XMO_STRUCTURES_PER_ROW,
        help=(
            "number of structures per row in the grid SVG, "
            f"default: {DEFAULT_XMO_STRUCTURES_PER_ROW}"
        ),
    )
    parser.add_argument(
        "--hide-atom-labels",
        action="store_true",
        help="hide atom-number labels",
    )
    parser.add_argument(
        "--hide-connection-labels",
        action="store_true",
        help="hide bond-pair and radical labels after each structure weight",
    )
    parser.add_argument(
        "--hide-lone-pairs",
        action="store_true",
        help="hide lone-pair dots",
    )
    args = parser.parse_args(argv)

    result = xmo2svg_file(
        args.xmo_file,
        connectivity=args.connectivity,
        weight_table=args.weight,
        max_structures=args.max_structures,
        baseline_index=args.baseline_index,
        charge=args.charge,
        hide_hydrogens=not args.show_hydrogens,
        write_individual_svgs=args.write_individual_svgs,
        show_atom_labels=not args.hide_atom_labels,
        show_lone_pairs=not args.hide_lone_pairs,
        structures_per_row=args.structures_per_row,
        projection=args.projection,
        condense_hydrogens=args.condense_hydrogens,
        show_connection_labels=not args.hide_connection_labels,
    )
    if result is not None:
        for line in xmo2svg_report_lines(result):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(xmo2svg())
