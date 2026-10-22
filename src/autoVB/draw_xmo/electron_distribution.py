"""Electron bookkeeping shared by the SVG and PyVista renderers.

Charge notes describe electron redistribution relative to a reference structure,
not absolute formal charges or population-analysis charges.
"""

from collections import Counter
from collections.abc import Iterable, Mapping


def electron_counts(
    atom_groups: Iterable[Iterable[int]], unpaired_atoms: Iterable[int] = (),
) -> dict[int, int]:
    """Count one electron per atom occurrence, including both ends of self-pairs."""
    counts = Counter(atom for group in atom_groups for atom in group)
    counts.update(unpaired_atoms)
    return dict(counts)


def charge_note_from_delta(delta: int) -> str:
    """Convert excess electrons to a negative note and missing electrons to positive."""
    if delta > 0:
        return "-" if delta == 1 else f"{delta}-"
    if delta < 0:
        return "+" if delta == -1 else f"{-delta}+"
    return ""


def charge_notes(
    expected_counts: Mapping[int, int],
    bond_pairs: Iterable[Iterable[int]],
    unpaired_atoms: Iterable[int] = (),
) -> dict[int, str]:
    """Map zero-based atom indices to notes; inputs use one-based atom numbers."""
    actual_counts = electron_counts(bond_pairs, unpaired_atoms)
    notes = {}
    for atom in sorted(set(expected_counts) | set(actual_counts)):
        note = charge_note_from_delta(actual_counts.get(atom, 0) - expected_counts.get(atom, 0))
        if note:
            notes[atom - 1] = note
    return notes
