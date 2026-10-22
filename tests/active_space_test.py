from types import SimpleNamespace

import numpy as np
import pytest

from autoVB.main import VBSettings
from autoVB.nbo.active_space import ActiveSpaceCandidate, select_fixed_active_space
from autoVB.nbo.nbo import XMVBNBO


def nbo_model(orbitals, nae, nao, spin=0, n_occupied=None):
    """Synthetic NBO metadata; no molecular calculations or file parsing."""
    model = XMVBNBO.__new__(XMVBNBO)
    model.input_data = SimpleNamespace(vbsettings=VBSettings(nae=nae, nao=nao), debug=False)
    model.mol = SimpleNamespace(spin=spin)
    model.active_electron, model.active_orbital = nae, nao
    model.occ_indices = list(range(len(orbitals) if n_occupied is None else n_occupied))
    model.occupation_numbers = np.array([occupancy for _, _, occupancy in orbitals])
    model.nbo_parser = SimpleNamespace(nbo_data=[
        SimpleNamespace(index=i + 1, orbital_type=kind, connection=atoms)
        for i, (kind, atoms, _) in enumerate(orbitals)
    ])
    return model


def test_explicit_12_10_selects_four_bonds_and_two_lone_pairs():
    model = nbo_model(
        [("BD", [2 * i + 1, 2 * i + 2], 1.6 + i * 0.01) for i in range(6)]
        + [("LP", [13], 1.98), ("LP", [14], 1.99)], 12, 10,
    )
    assert model.get_as_from_aoi(list(range(6))) == (12, 12)  # Previous greedy result.
    nae, nao, indices = model.get_aoi(auto_set=True)
    assert (nae, nao) == (12, 10)
    assert indices == [0, 1, 2, 3, 6, 7]
    assert model.get_as_from_aoi(indices) == (12, 10)
    assert model.active_indices == indices
    assert (model.active_electron, model.active_orbital) == (12, 10)


def test_lowest_occupancy_exact_combination_wins():
    candidates = [
        ActiveSpaceCandidate(0, 1.9, 2, 2), ActiveSpaceCandidate(1, 1.5, 2, 2),
        ActiveSpaceCandidate(2, 1.8, 2, 1), ActiveSpaceCandidate(3, 1.6, 2, 1),
    ]
    assert select_fixed_active_space(candidates, 4, 3) == [1, 3]


def test_ties_are_deterministic_and_each_candidate_is_used_once():
    candidates = [ActiveSpaceCandidate(i, 1.8, 2, 2) for i in range(3)]
    assert select_fixed_active_space(candidates, 4, 4) == [0, 1]
    assert select_fixed_active_space(candidates[::-1], 4, 4) == [0, 1]
    with pytest.raises(ValueError, match="No exact combination"):
        select_fixed_active_space(candidates[:1], 4, 4)


def test_no_exact_size_raises_instead_of_returning_12_12_or_partial_space():
    model = nbo_model([("BD", [1, 2], 1.8)] * 6, 12, 10)
    with pytest.raises(ValueError, match=r"12 electrons / 10 orbitals.*6 occupied NBO candidates"):
        model.get_aoi()
    with pytest.raises(ValueError, match="No exact combination"):
        nbo_model([("BD", [1, 2], 1.8)], 12, 10).get_aoi()


def test_spin_correction_is_applied_once_and_alpha_occupancies_are_supported():
    model = nbo_model([("BD", [1, 2], 0.95), ("LP", [3], 0.99)], 3, 3, spin=1)
    nae, nao, indices = model.get_aoi()
    assert (nae, nao, indices) == (3, 3, [0, 1])
    assert model.get_as_from_aoi(indices) == (3, 3)


def test_virtual_and_antibonding_orbitals_do_not_fill_an_impossible_target():
    model = nbo_model([
        ("BD", [1, 2], 1.8), ("BD*", [1, 2], 1.1), ("LP", [3], 1.9),
    ], 4, 3, n_occupied=2)
    with pytest.raises(ValueError, match="No exact combination"):
        model.get_aoi()


@pytest.mark.parametrize("nae,nao,spin", [(11, 10, 0), (12, 5, 0), (12, 10, 1), (2, 2, -1)])
def test_invalid_electron_or_spin_targets_are_rejected(nae, nao, spin):
    with pytest.raises(ValueError, match="Invalid active space"):
        select_fixed_active_space([], nae, nao, spin)


def test_explicit_aoi_keeps_precedence_and_existing_size_check():
    model = nbo_model([("BD", [1, 2], 1.8), ("LP", [3], 1.9)], 4, 3)
    model.orbital_matrix = np.zeros((2, 2))
    model.input_data.vbsettings.aoi = [2, 1]
    assert model.get_aoi() == (4, 3, [1, 0])
    model.active_orbital = 4
    with pytest.raises(ValueError, match="does not match expected"):
        model.get_aoi()
