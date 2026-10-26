"""Resolve output dialect settings before generating an XMI file."""


def resolve_xmi_iscf(xmitype: str, iscf: int | None) -> int:
    if not isinstance(xmitype, str) or xmitype not in ("4", "2026", "open"):
        raise ValueError("xmitype must be a string: '4', '2026', or 'open'")
    if xmitype == "open":
        if iscf is None:
            return 8
        if type(iscf) is not int or iscf not in (7, 8, 9):
            raise ValueError("xmitype=open requires iscf=7, 8, or 9 (default: 8)")
    return 5 if iscf is None else iscf
