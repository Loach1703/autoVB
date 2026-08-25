"""Backward-compatible alias for :mod:`autoVB.cli.xmo2svg`."""

from .xmo2svg import (
    parse_draw_xmo_max_structures,
    parse_draw_xmo_structures_per_row,
    xmo2svg,
)

draw_xmo = xmo2svg
