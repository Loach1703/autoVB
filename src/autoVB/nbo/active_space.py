"""Exact-size selection of NBO representatives; no file or molecule dependencies."""

from dataclasses import dataclass
from math import isfinite


TWO_ELECTRON_TYPES = frozenset({"BD", "LP", "CR"})
ZERO_ELECTRON_TYPES = frozenset({"BD*", "LP*", "RY", "RY*"})


def nbo_space_contribution(orbital_type: str, n_centers: int) -> tuple[int, int]:
    """Use formal electrons and the number of atom-local orbitals after expansion."""
    return (2 if orbital_type in TWO_ELECTRON_TYPES else 0), n_centers


@dataclass(frozen=True)
class ActiveSpaceCandidate:
    index: int  # Zero-based NBO index.
    occupancy: float
    electrons: int
    orbitals: int


def select_fixed_active_space(
    candidates: list[ActiveSpaceCandidate], nae: int, nao: int, spin: int = 0,
) -> list[int]:
    """Minimize summed occupancy subject to exact electron and expanded-orbital counts.

    Spin is PySCF's number of unpaired electrons. As in autoVB's existing AOI
    counting, subtract it once from the total formal electron contribution.
    Equal-cost combinations are resolved by NBO index, independently of input order.
    """
    if any(type(value) is not int for value in (nae, nao, spin)):
        raise ValueError("Active electrons, orbitals and spin must be integers")
    if (nae <= 0 or nao <= 0 or nae > 2 * nao or spin < 0
            or spin > min(nae, 2 * nao - nae) or (nae - spin) % 2):
        raise ValueError(f"Invalid active space ({nae} electrons / {nao} orbitals) for spin={spin}")

    by_index = {candidate.index: candidate for candidate in candidates}
    if len(by_index) != len(candidates):
        raise ValueError("Duplicate NBO candidate indices")
    for candidate in candidates:
        if (any(type(value) is not int for value in
                (candidate.index, candidate.electrons, candidate.orbitals))
                or candidate.index < 0 or candidate.electrons < 0 or candidate.orbitals <= 0
                or not isfinite(candidate.occupancy)):
            raise ValueError(f"Invalid active-space candidate: {candidate}")

    target_electrons = nae + spin
    # State -> (cost, selected indices). Snapshotting prevents reusing a candidate.
    states = {(0, 0): (0.0, ())}
    for candidate in sorted(candidates, key=lambda item: item.index):
        for (electrons, orbitals), (cost, selected) in list(states.items()):
            key = electrons + candidate.electrons, orbitals + candidate.orbitals
            if key[0] > target_electrons or key[1] > nao:
                continue
            proposal = cost + candidate.occupancy, selected + (candidate.index,)
            if key not in states or proposal < states[key]:
                states[key] = proposal

    result = states.get((target_electrons, nao))
    if result is None:
        raise ValueError(
            f"Cannot select requested active space ({nae} electrons / {nao} orbitals) "
            f"from {len(candidates)} occupied NBO candidates "
            f"(total formal contributions: {sum(c.electrons for c in candidates)} electrons / "
            f"{sum(c.orbitals for c in candidates)} orbitals; spin correction: {spin}). "
            "No exact combination exists; review the requested space or provide explicit AOI."
        )
    return sorted(result[1], key=lambda index: (by_index[index].occupancy, index))
