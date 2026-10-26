"""Unified command for SVG, PyVista PNG and interactive HTML output."""

import argparse
from pathlib import Path


def _indices(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--indices requires comma-separated positive integers") from exc
    if not result or min(result) < 1 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--indices requires distinct positive integers")
    return result


def _visibility(parser, name, *, default, help):
    group = parser.add_mutually_exclusive_group()
    dest = "show_" + name.replace("-", "_")
    group.add_argument("--show-" + name, dest=dest, action="store_true",
                       help=argparse.SUPPRESS if default else help)
    group.add_argument("--hide-" + name, dest=dest, action="store_false",
                       help=help if default else argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def xmoview(argv=None, *, default_format="svg", prog="xmoview") -> int:
    from .xmo2svg import parse_draw_xmo_max_structures, parse_draw_xmo_structures_per_row

    parser = argparse.ArgumentParser(
        prog=prog, description="View XMO VB structures as 2D SVG, 3D PNG or interactive HTML.",
    )
    parser.add_argument("xmo_file", type=Path, help="input .xmo file")
    parser.add_argument("-f", "--format", choices=("svg", "png", "html"),
                        help=f"output format; inferred from -o, otherwise {default_format}")
    parser.add_argument("-o", "--output", type=Path, help="output path; default: <input stem>.<format>")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("-m", "--max-structures", type=parse_draw_xmo_max_structures,
                           default=argparse.SUPPRESS, help="highest-weight structures: positive integer or all (default: 6)")
    selection.add_argument("--indices", type=_indices, help="original XMO structure indices, e.g. 12,35,80; preserves order")
    parser.add_argument("-w", "--weight", choices=("both", "lowdin", "cc", "inverse", "renormalized"),
                        default="both", help="weight table (default: both); both ranks by Lowdin and displays CC/Lowdin")
    parser.add_argument("--baseline-index", type=int, help="reference structure index; default: highest weight")
    parser.add_argument("-n", "--structures-per-row", type=parse_draw_xmo_structures_per_row,
                        help="SVG/PNG panels per row (default: 3); not applicable to HTML tabs")
    parser.add_argument("--panel-size", type=int, help="SVG/PNG panel side length (default: SVG native size, PNG 640)")
    hydrogens = parser.add_mutually_exclusive_group()
    hydrogens.add_argument("--show-hydrogens", dest="show_hydrogens", action="store_true",
                           help="show all H atoms (default for PNG/HTML)")
    hydrogens.add_argument("--hide-hydrogens", dest="show_hydrogens", action="store_false",
                           help="hide ordinary H atoms (default for SVG); active H atoms are retained")
    parser.set_defaults(show_hydrogens=None)
    _visibility(parser, "atom-labels", default=False, help="show original geometry atom numbers")
    _visibility(parser, "connection-labels", default=False, help="show active pairs and radical atom numbers in legends")
    _visibility(parser, "lone-pairs", default=True, help="hide active lone-pair dots (shown by default)")
    _visibility(parser, "charges", default=True, help="hide charge changes relative to the reference structure (shown by default)")
    parser.add_argument("--circle-charges", action="store_true", help="draw a circle around each charge-change label (SVG/PNG/HTML; off by default)")
    parser.add_argument("--transparent-background", action="store_true", help="transparent SVG/PNG background")

    two_d = parser.add_argument_group("2D SVG options")
    two_d.add_argument("--connectivity", choices=("orb", "rdkit"), help="connectivity source (default: orb)")
    two_d.add_argument("--charge", type=int, help="total charge for RDKit bond-order perception (default: 0)")
    two_d.add_argument("--layout", choices=("rdkit", "pca", "optimized3d", "contact"), help="2D layout (default: rdkit)")
    condensed = two_d.add_mutually_exclusive_group()
    condensed.add_argument("--condense-hydrogens", dest="condense_hydrogens", action="store_true", default=None)
    condensed.add_argument("--no-condense-hydrogens", "--no-condensed-hydrogens",
                           dest="condense_hydrogens", action="store_false", help="disable compact NH2 etc. labels")
    two_d.add_argument("--write-individual", "--write-individual-svgs", action="store_true",
                       help="also write an SVG for each structure")

    three_d = parser.add_argument_group("3D PNG/HTML options")
    three_d.add_argument("--view", choices=("pca", "optimized3d"), help="initial camera selection (default: optimized3d)")
    three_d.add_argument("--color-scheme", choices=("element", "highlight"),
                        help="element colors (default), or highlight: light-gray framework and bright-red active pairs; atom colors are retained")
    three_d.add_argument("--hide-legend", action="store_true", help="hide the atom-color legend (PNG/HTML); structure weights remain visible")
    three_d.add_argument("--azimuth", type=float, help="manual camera azimuth (default: 35); overrides automatic search")
    three_d.add_argument("--elevation", type=float, help="manual camera elevation (default: 55); overrides automatic search")
    parser.add_argument("--projection", choices=("rdkit", "pca", "optimized3d", "contact"),
                        help="compatibility alias for --layout (SVG) or --view (PNG/HTML)")
    args = parser.parse_args(argv)

    suffix = args.output.suffix.lower().lstrip(".") if args.output else None
    if suffix is not None and suffix not in ("svg", "png", "html"):
        parser.error("--output must have a .svg, .png or .html extension")
    if args.format and suffix and args.format != suffix:
        parser.error("--format conflicts with the --output extension")
    output_format = args.format or suffix or default_format
    if args.show_hydrogens is None:
        args.show_hydrogens = output_format != "svg"
    output = args.output if args.output is not None else args.xmo_file.with_suffix("." + output_format)
    if args.baseline_index is not None and args.baseline_index < 1:
        parser.error("--baseline-index must be a positive original structure index")
    if args.panel_size is not None and args.panel_size < 128:
        parser.error("--panel-size must be at least 128")

    if output_format == "svg":
        if args.hide_legend:
            parser.error("--hide-legend is only supported for PNG/HTML output; SVG has no atom-color legend")
        for name in ("view", "azimuth", "elevation", "color_scheme"):
            if getattr(args, name) is not None:
                parser.error(f"--{name.replace('_', '-')} is only supported for PNG/HTML output")
        if args.layout and args.projection and args.layout != args.projection:
            parser.error("--layout conflicts with --projection")
        layout = args.layout or args.projection or "rdkit"
        if args.connectivity == "rdkit" and layout != "rdkit":
            parser.error("--connectivity rdkit currently requires --layout rdkit")
    else:
        for name in ("connectivity", "charge", "layout", "condense_hydrogens"):
            if getattr(args, name) is not None:
                parser.error(f"--{name.replace('_', '-')} is only supported for SVG output")
        if args.write_individual:
            parser.error("--write-individual is only supported for SVG output")
        if args.projection not in (None, "pca", "optimized3d"):
            parser.error("PNG/HTML --projection must be pca or optimized3d; use --layout for SVG")
        if args.view and args.projection and args.view != args.projection:
            parser.error("--view conflicts with --projection")
        view = args.view or args.projection or "optimized3d"
    if output_format == "html":
        if args.structures_per_row is not None or args.panel_size is not None:
            parser.error("HTML uses responsive Rank tabs; --structures-per-row and --panel-size apply to SVG/PNG")
        if args.transparent_background:
            parser.error("--transparent-background is only supported for SVG/PNG output")

    common = dict(
        weight_table=args.weight, max_structures=getattr(args, "max_structures", 6),
        baseline_index=args.baseline_index, indices=args.indices,
        structures_per_row=args.structures_per_row or 3,
        show_atom_labels=args.show_atom_labels, show_connection_labels=args.show_connection_labels,
        show_lone_pairs=args.show_lone_pairs, show_charges=args.show_charges,
        circle_charges=args.circle_charges,
        transparent_background=args.transparent_background,
    )
    if output_format == "svg":
        from .xmo2svg import xmo2svg_file, xmo2svg_report_lines

        result = xmo2svg_file(
            args.xmo_file, output=output, connectivity=args.connectivity or "orb",
            charge=args.charge if args.charge is not None else 0,
            projection=layout, hide_hydrogens=not args.show_hydrogens,
            condense_hydrogens=args.condense_hydrogens is not False,
            write_individual_svgs=args.write_individual, panel_size=args.panel_size, **common,
        )
        if result is not None:
            for line in xmo2svg_report_lines(result):
                print(line)
    else:
        from .xmo2png import xmo2png_file

        result = xmo2png_file(
            args.xmo_file, output, show_hydrogens=args.show_hydrogens,
            panel_size=args.panel_size or 640, projection=view,
            color_scheme=args.color_scheme or "element",
            show_legend=not args.hide_legend,
            azimuth=args.azimuth, elevation=args.elevation, **common,
        )
        print(f"PyVista {output_format.upper()}: {result.output_file.resolve()}")
        print(f"Structures: {result.structure_count}; atoms: {result.displayed_atom_count}; weight: {result.weight_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(xmoview())
