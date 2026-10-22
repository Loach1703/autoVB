"""Backward-compatible alias for :mod:`autoVB.cli.xmoview`."""

from .xmo2svg import parse_draw_xmo_max_structures, parse_draw_xmo_structures_per_row
from .xmoview import xmoview


def draw_xmo(argv=None) -> int:
    return xmoview(argv, prog="draw_xmo")


if __name__ == "__main__":
    raise SystemExit(draw_xmo())
