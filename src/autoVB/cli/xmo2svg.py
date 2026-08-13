import argparse
from pathlib import Path

from .draw_xmo import (
    DEFAULT_XMO_ACTIVE_SPACE_COLOR,
    DEFAULT_XMO_ACTIVE_SPACE_WIDTH,
    DEFAULT_XMO_MAX_STRUCTURES,
    DEFAULT_XMO_STRUCTURES_PER_ROW,
    DEFAULT_XMO_WEIGHT_TABLE,
    parse_draw_xmo_max_structures,
    parse_draw_xmo_structures_per_row,
)


def xmo2svg_file(
    xmo_file: str | Path,
    *,
    weight_table: str = DEFAULT_XMO_WEIGHT_TABLE,
    max_structures: int | None = DEFAULT_XMO_MAX_STRUCTURES,
    baseline_index: int | None = None,
    charge: int = 0,
    hide_hydrogens: bool = True,
    write_individual_svgs: bool = False,
    show_atom_labels: bool = True,
    show_lone_pairs: bool = True,
    structures_per_row: int = DEFAULT_XMO_STRUCTURES_PER_ROW,
    projection: str = "rdkit",
    condense_hydrogens: bool = True,
    show_connection_labels: bool = True,
):
    """使用 XMO ``$orb`` 标签建立键连并生成价键结构 SVG。"""
    from ..draw_xmo.orbital_connectivity_molecule_drawer import (
        OrbitalConnectivityMoleculeDrawer,
    )
    from ..draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
    from ..io.xmo_output_parser import XmoParser
    from ..utils import constants

    print(f"XMO2SVG version: {constants.VERSION}")
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
        show_connection_labels=show_connection_labels,
    )
    drawer_input = converter.convert()
    hide_hydrogens = converter.hide_hydrogens

    drawer = OrbitalConnectivityMoleculeDrawer(
        xyz_file=drawer_input.xyz_file,
        output_dir=output_dir,
        charge=charge,
        orbital_atom_rows=parsed_data.orb,
        active_bond_atom=drawer_input.active_bond_atom,
        active_space=drawer_input.active_space,
        baseline_unpaired_atoms=drawer_input.baseline_unpaired_atoms,
        active_space_color=DEFAULT_XMO_ACTIVE_SPACE_COLOR,
        active_space_width=DEFAULT_XMO_ACTIVE_SPACE_WIDTH,
        color_active_space=True,
        show_atom_labels=show_atom_labels,
        hide_hydrogens=hide_hydrogens,
        show_lone_pairs=show_lone_pairs,
        write_individual_svgs=write_individual_svgs,
        structures_per_row=structures_per_row,
        projection=projection,
        condense_hydrogens=condense_hydrogens,
    )
    result = drawer.draw()
    grid_path = output_dir / f"{xmo_path.stem}_grid.svg"
    output_path = output_dir / f"{xmo_path.stem}.svg"
    grid_path.replace(output_path)
    result.written_files = [
        output_path if written_file == grid_path else written_file
        for written_file in result.written_files
    ]

    print(f"Read XMO from: {parsed_data.source_file.resolve()}")
    print(f"Generated XYZ: {drawer_input.xyz_file.resolve()}")
    print(f"Active orbital -> atom: {drawer_input.orbital_to_atom}")
    print(f"Weight table: {drawer_input.weight_table}")
    print(f"active_bond_atom: {drawer_input.active_bond_atom}")
    print(f"Connectivity source: $orb")
    print(f"Projection: {projection}")
    print(f"Drawn structures: {len(drawer_input.active_space)}")
    print(f"Output directory: {result.output_dir.resolve()}")
    for out_file in result.written_files:
        print(f" - {out_file.name}")

    return result


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

    xmo2svg_file(
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
        projection=args.projection,
        condense_hydrogens=args.condense_hydrogens,
        show_connection_labels=not args.hide_connection_labels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(xmo2svg())
