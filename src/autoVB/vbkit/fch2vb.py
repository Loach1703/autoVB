import subprocess
import math
from pathlib import Path
from types import SimpleNamespace

from mokit.lib.gaussian import load_mol_from_fch

from ..io.writers import XMIData, write_xmi_file
from ..utils.utils import main_read_gamess_dat, make_xmvb_format_text, pyscf_to_xyz


def build_init_guess_section(orbital_matrix):
    head_text = (' ' + str(orbital_matrix.shape[1])) * orbital_matrix.shape[0]
    orb_text = ''
    for i, orb in enumerate(orbital_matrix):
        orb_text += f'# ORBITAL        {i+1}  NAO =    {len(orb)}\n'
        orb_text += make_xmvb_format_text(orb, per_line=4)
        orb_text += '\n'
    return head_text + '\n' + orb_text.strip("\n")


def fch2vb_impl(fch_path: Path, output: Path | None = None, basis: str = "", norb: int | None = None) -> Path:
    fch_path = Path(fch_path)
    subprocess.run(["fch2inp", str(fch_path)], check=True)

    inp_path = fch_path.with_suffix(".inp")
    mol = load_mol_from_fch(fch_path)
    norb = norb or math.ceil(mol.nelectron / 2)
    orbital_matrix = main_read_gamess_dat(inp_path, all_orbital_number=norb)
    xmi_path = output or fch_path.with_suffix(".xmi")

    xmidata = XMIData(
        molecule_name=fch_path.stem,
        method="vbscf",
        stru_type="full",
        int_type="libcint",
        iscf=5,
        nae=0,
        nao=0,
        ncharge=mol.charge,
        nmul=mol.spin + 1,
        basis_set=basis,
        sort=False,
        orb_section="",
        geo_section=pyscf_to_xyz(mol),
        init_guess_section=build_init_guess_section(orbital_matrix),
    )
    passthrough = SimpleNamespace(ctrl_extra_lines=["orbtyp=oeo"], str_section_text=None)
    write_xmi_file(str(xmi_path), xmidata, passthrough)
    return xmi_path.with_suffix(".xmi")
