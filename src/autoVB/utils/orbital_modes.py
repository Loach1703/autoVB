"""Translate autoVB orbital modes into XMVB orbital expansion domains."""

import re

ORB_MODES = ("hao", "mixed", "oeo", "oeo_direct")


def orbital_definition(section: str, mode: str, nao: int, natoms: int) -> tuple[str, str]:
    """Keep orbital numbering; only expand the requested orbital domains."""
    if mode == "hao":
        return "hao", section
    rows = [line for line in section.splitlines() if line.split("#", 1)[0].strip()][1:]
    whole_system = " ".join(str(atom) for atom in range(1, natoms + 1))
    inactive_count = len(rows) - nao
    if mode == "mixed":
        active_rows = rows[inactive_count:]
        sizes = [natoms] * inactive_count + [len(row.split("#", 1)[0].split()) for row in active_rows]
        rows = [whole_system] * inactive_count + active_rows
        orbtyp = "hao"
    else:
        sizes = [natoms] * len(rows)
        rows = [whole_system] * len(rows)
        orbtyp = "oeo"
    return orbtyp, "\n".join([" ".join(map(str, sizes)), *rows])


def without_orbital_options(lines: list[str]) -> list[str]:
    """Replace only managed domain options; preserve unrelated passthrough text."""
    pattern = re.compile(r"(?<!\S)(?:orbtyp|frgtyp)\s*=\s*\S+", re.IGNORECASE)
    return [cleaned for line in lines if (cleaned := pattern.sub("", line).strip())]
