from pathlib import Path
from .main import XMVBNBO, VBSettings, autoVBInputData
from pyscf import gto

def autovb_nbo_impl(xyz: Path, basis: str, charge: int, spin: int) -> int:
    from .writers import write_gjf_nbo_file
    mol = gto.M(
        atom=str(xyz),
        basis=basis,
        charge=charge,
        spin=spin,
    )
    name = xyz.stem
    write_gjf_nbo_file(mol, name, mem='4GB', nproc=4)
    print(f"Wrote Gaussian NBO input file to {name}.gjf")
    print(f'You need to manually verify if the charge and spin multiplicity are correct.')
    return 0

def autovb_xmi_impl(name: str, mol: gto.Mole, input_data: autoVBInputData) -> int:
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
    wxp.write_xmi()
    return 0