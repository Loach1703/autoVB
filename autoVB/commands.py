import sys
from pathlib import Path
import argparse
from typing import TYPE_CHECKING
from autoVB import GaussianNBO, XMVBNBO
from .utils import generate_fch_from_chk
from mokit.lib.gaussian import load_mol_from_fch
from pyscf import gto
if TYPE_CHECKING:
    pass

def autovb_nbo_impl(xyz: Path, basis: str, charge: int, spin: int) -> int:
    mol = gto.M(
        atom=str(xyz),
        basis=basis,
        charge=charge,
        spin=spin,
    )
    name = xyz.stem
    gn = GaussianNBO(name, mol)
    gn.write_gjf()
    print(f"Wrote Gaussian NBO input file to {name}.gjf")
    print(f'You need to manually verify if the charge and spin multiplicity are correct.')
    return 0

def autovb_xmi_impl(name: str, mol: gto.Mole, basis:str, nae: int, nao: int, aoa:list[list[int]], threshold: float, reorder: bool, atom_slice: bool) -> int:
    wxp = XMVBNBO(name, mol)
    wxp.set_basis_set(basis)

    if aoa and nae > 0 and nao > 0:
        wxp.set_active_space(nae, nao)
        wxp.set_active_orbital_atom(aoa)
    elif aoa:
        gaoi = wxp.get_active_orbital_indices_from_atom(aoa)
        nao = len([j for i in aoa for j in i])
        nae = len(gaoi) * 2
        wxp.set_active_space(nae, nao)
        wxp.set_active_orbital_atom(aoa)
    elif nae > 0 and nao > 0:
        wxp.set_active_space(nae, nao)
    else:
        nae, nao = wxp.auto_select_active_space(threshold=threshold, auto_set=True)
        wxp.set_active_space(nae, nao)
    inact, act = wxp.split_inactive_active_orbitals()
    wxp.write_xmi(inact, act, reorder=reorder, atom_slice=atom_slice)
    return 0