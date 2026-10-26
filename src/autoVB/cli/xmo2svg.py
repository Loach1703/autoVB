import argparse

from ..vbkit.xmo2svg import (
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
    """Compatibility alias for the unified xmoview command."""
    from .xmoview import xmoview
    return xmoview(argv, prog="xmo2svg")


if __name__ == "__main__":
    raise SystemExit(xmo2svg())
