"""Convenience entry point: xmoview with HTML as the default format."""

from .xmoview import xmoview


def xmo2html(argv=None) -> int:
    return xmoview(argv, default_format="html", prog="xmo2html")


if __name__ == "__main__":
    raise SystemExit(xmo2html())
