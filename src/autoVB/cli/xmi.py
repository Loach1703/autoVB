import argparse
from pathlib import Path
import sys

from mokit.lib.gaussian import load_mol_from_fch
from pyscf import gto

from ..main import XMVBNBO, VBSettings, autoVBInputData
from ..utils import generate_fch_from_chk, pyscf_to_xyz


def autovb_xmi_impl(name: str, mol: gto.Mole, input_data: autoVBInputData) -> int:
    from ..writers import write_xmi_file
    wxp = XMVBNBO(name, mol)
    wxp.set_basis_set(input_data.basis)

    if input_data.vbsettings.aoa:
        aoi = wxp.get_active_orbital_indices_from_active_atoms(input_data.vbsettings.aoa)
        nao = len(input_data.vbsettings.aoa)
        nae = len(aoi) * 2
        wxp.set_active_space(nae, nao)
    elif input_data.vbsettings.nae > 0 and input_data.vbsettings.nao > 0:
        wxp.set_active_space(input_data.vbsettings.nae, input_data.vbsettings.nao)
        aoi = wxp.get_active_orbital_indices()
    else:
        nae, nao, aoi = wxp.auto_select_active_space_default(auto_set=True)
        wxp.set_active_space(nae, nao)
    wxp.split_inactive_active_orbitals(aoi)
    xmidata = wxp.get_xmidata()
    write_xmi_file(name, xmidata, input_data.xmi_passthrough)
    return 0


def autovb_xmi(argv=None):
    parser = argparse.ArgumentParser(prog="autovb-xmi", description="Generate XMVB .xmi from .fch/.chk or basename")
    parser.add_argument("file", type=Path, help="NBO output .fch/.chk file or basename (without extension)")
    parser.add_argument("basis", help="basis set override for xmi header")
    parser.add_argument("--threshold", "-t", type=float, default=1.96, help="threshold for selecting important structures based on cc weight, default 1.96 (corresponding to 95%% cumulative weight)")
    parser.add_argument("--active_electron", "-nae", type=int, default=0, help="number of active electrons, default 0")
    parser.add_argument("--active_orbital", "-nao", type=int, default=0, help="number of active orbitals, default 0")
    parser.add_argument("--active_orbital_atom", "-aoa", nargs='*', type=int, default=[], help="list of active orbital atoms, default empty")
    args = parser.parse_args(argv)

    p: Path = args.file
    fchname = p.with_suffix(".fch")
    if p.suffix != ".fch":
        chkname = p.with_suffix(".chk")
        generate_fch_from_chk(chkname, fchname)
    
    mol = load_mol_from_fch(fchname)
    name = p.stem
    aoa = args.active_orbital_atom
    nae = args.active_electron
    nao = args.active_orbital
    threshold = args.threshold if hasattr(args, 'threshold') else 0
    geometry = pyscf_to_xyz(mol)
    atvb_input = autoVBInputData(
        title=name,
        filepath=p,
        filename=p.stem,
        method='vbscf',
        basis=args.basis,
        geometry=geometry,
    )
    vbsetting = VBSettings(
        nae=nae,
        nao=nao,
        aoa=aoa,
        threshold=threshold,
    )
    atvb_input.vbsettings = vbsetting

    return autovb_xmi_impl(name, mol, atvb_input)
