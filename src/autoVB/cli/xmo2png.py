"""Compatibility entry point: xmoview with PNG as the default format."""

from ..vbkit.xmo2png import xmo2png_file


def xmo2png(argv=None) -> int:
    from .xmoview import xmoview
    return xmoview(argv, default_format="png", prog="xmo2png")


if __name__ == "__main__":
    raise SystemExit(xmo2png())
