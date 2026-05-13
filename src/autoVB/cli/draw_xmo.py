import argparse
from pathlib import Path
import sys

DEFAULT_XMO_MAX_STRUCTURES = 20
DEFAULT_XMO_WEIGHT_TABLE = "lowdin"
DEFAULT_XMO_ACTIVE_SPACE_COLOR = "#B00000"
DEFAULT_XMO_ACTIVE_SPACE_WIDTH = 3.0
DEFAULT_XMO_STRUCTURES_PER_ROW = 2


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


def draw_xmo_file(
    xmo_file: str | Path,
    *,
    weight_table: str = DEFAULT_XMO_WEIGHT_TABLE,
    max_structures: int | None = DEFAULT_XMO_MAX_STRUCTURES,
    baseline_index: int = 1,
    charge: int = 0,
    hide_hydrogens: bool = True,
    write_individual_svgs: bool = False,
    show_atom_labels: bool = True,
    show_lone_pairs: bool = True,
    structures_per_row: int = DEFAULT_XMO_STRUCTURES_PER_ROW,
):
    from ..draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
    from ..draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
    from ..draw_xmo.xmo_output_parser import XmoParser

    xmo_path = Path(xmo_file)
    if not xmo_path.exists():
        raise FileNotFoundError(f"XMO file not found: {xmo_path}")
    if not xmo_path.is_file():
        raise ValueError(f"XMO path is not a file: {xmo_path}")

    output_dir = xmo_path.parent
    parsed_data = XmoParser(xmo_path).parse()
    converter = XmoToDrawerInputConverter(
        parsed_data,
        output_dir,
        hide_hydrogens=hide_hydrogens,
        max_structures=max_structures,
        baseline_index=baseline_index,
        weight_table=weight_table,
    )
    drawer_input = converter.convert()

    drawer = MoleculeBondVariantDrawer(
        xyz_file=drawer_input.xyz_file,
        output_dir=output_dir,
        charge=charge,
        active_bond_atom=drawer_input.active_bond_atom,
        active_space=drawer_input.active_space,
        active_space_color=DEFAULT_XMO_ACTIVE_SPACE_COLOR,
        active_space_width=DEFAULT_XMO_ACTIVE_SPACE_WIDTH,
        color_active_space=True,
        show_atom_labels=show_atom_labels,
        hide_hydrogens=hide_hydrogens,
        show_lone_pairs=show_lone_pairs,
        write_individual_svgs=write_individual_svgs,
        structures_per_row=structures_per_row,
    )
    result = drawer.draw()

    print(f"Read XMO from: {parsed_data.source_file.resolve()}")
    print(f"Generated XYZ: {drawer_input.xyz_file.resolve()}")
    print(f"Active orbital -> atom: {drawer_input.orbital_to_atom}")
    print(f"Weight table: {drawer_input.weight_table}")
    print(f"active_bond_atom: {drawer_input.active_bond_atom}")
    print(f"Drawn structures: {len(drawer_input.active_space)}")
    print(f"Output directory: {result.output_dir.resolve()}")
    for out_file in result.written_files:
        print(f" - {out_file.name}")

    return result


def draw_xmo(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="draw_xmo",
        description="Read an XMO file and generate valence-bond SVG files in the same directory.",
    )
    parser.add_argument("xmo_file", help="input .xmo file")
    parser.add_argument(
        "--weight",
        "-w",
        choices=("lowdin", "cc"),
        default=DEFAULT_XMO_WEIGHT_TABLE,
        help=f"weight table to use, default: {DEFAULT_XMO_WEIGHT_TABLE}",
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
        default=1,
        help="structure index used as the initial electron distribution, default: 1",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="total charge used by RDKit when perceiving bonds from XYZ, default: 0",
    )
    parser.add_argument(
        "--show-hydrogens",
        action="store_true",
        help="show hydrogen atoms; hydrogens are hidden by default",
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
        "--hide-lone-pairs",
        action="store_true",
        help="hide lone-pair dots",
    )
    args = parser.parse_args(argv)

    draw_xmo_file(
        args.xmo_file,
        weight_table=args.weight,
        max_structures=args.max_structures,
        baseline_index=args.baseline_index,
        charge=args.charge,
        hide_hydrogens=not args.show_hydrogens,
        write_individual_svgs=args.write_individual_svgs,
        show_atom_labels=not args.hide_atom_labels,
        show_lone_pairs=not args.hide_lone_pairs,
        structures_per_row=args.structures_per_row,
    )
    return 0