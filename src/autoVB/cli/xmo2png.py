import argparse
from pathlib import Path

from ..vbkit.xmo2png import xmo2png_file


def xmo2png(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="xmo2png", description="Render XMO VB structures as PyVista 3D PNGs or interactive HTML.",
    )
    parser.add_argument("xmo_file", type=Path)
    parser.add_argument("-o", "--output", type=Path, help=".png or interactive .html; default: <stem>_3d.png")
    parser.add_argument("-w", "--weight", choices=("both", "lowdin", "cc", "inverse", "renormalized"), default="both")
    parser.add_argument("-m", "--max-structures", type=int, default=3)
    parser.add_argument("-n", "--structures-per-row", type=int, default=3)
    parser.add_argument("--panel-size", type=int, default=640, help="pixels per square panel")
    parser.add_argument("--hide-hydrogens", action="store_true", help="hide H except atoms participating in active pairs")
    parser.add_argument("--show-atom-labels", action="store_true", help="show element and original geometry atom numbers")
    parser.add_argument("--show-connection-labels", action="store_true")
    parser.add_argument("--hide-charges", action="store_true", help="hide charge changes relative to the reference structure")
    parser.add_argument("--baseline-index", type=int, help="reference structure index for charge changes; defaults to highest weight")
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--projection", choices=("pca", "optimized3d"), default="pca",
                        help="camera selection: PCA-based view (default) or automatic optimized3d search")
    parser.add_argument("--azimuth", type=float,
                        help="manual camera azimuth in degrees (default 35); overrides automatic search")
    parser.add_argument("--elevation", type=float,
                        help="manual camera elevation in degrees (default 55); overrides automatic search")
    args = parser.parse_args(argv)
    result = xmo2png_file(
        args.xmo_file, args.output, weight_table=args.weight, max_structures=args.max_structures,
        structures_per_row=args.structures_per_row, panel_size=args.panel_size,
        show_hydrogens=not args.hide_hydrogens, show_atom_labels=args.show_atom_labels,
        show_connection_labels=args.show_connection_labels,
        show_charges=not args.hide_charges, baseline_index=args.baseline_index,
        transparent_background=args.transparent_background, projection=args.projection,
        azimuth=args.azimuth, elevation=args.elevation,
    )
    output_type = "HTML" if result.output_file.suffix.lower() == ".html" else "PNG"
    print(f"PyVista {output_type}: {result.output_file.resolve()}")
    print(f"Structures: {result.structure_count}; atoms: {result.displayed_atom_count}; weight: {result.weight_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(xmo2png())
