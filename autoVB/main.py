# TODO 重构代码块
import subprocess
import shutil
from pathlib import Path
import os
from pyscf import gto, scf
import re

from .utils import find_executable_in_env, find_tool, generate_fch_from_chk
from .autoVB import GaussianNBO, XMVBNBO
from .commands import autovb_xmi_impl
from mokit.lib.gaussian import load_mol_from_fch

class autoVBMain:
    def __init__(self, filepath: Path, mem: str = "4G", nproc: str = "4"):
        self._check_gaussian_env()
        self.filepath = filepath
        self.filename = filepath.stem
        self.nbo_gjf_name = f"{self.filename}_nbo"
        self.nbo_gjf_name_upper = f"{self.filename}_NBO"
        self.xmi_name = f"{self.filename}_vb"
        self.mem = mem
        self.nproc = nproc

    def _check_gaussian_env(self):
        self.gaussian_exe = find_executable_in_env()
        if not self.gaussian_exe:
            raise RuntimeError(
                "can not find Gaussian execution, check environment variable GAUSS_EXE or PATH for Gaussian executable.\n"
            )
        
        else:
            print(f"find Gaussian execution: {self.gaussian_exe}")

        # 检查 formchk
        self.formchk_exe = find_tool("formchk")
        if not self.formchk_exe:
            raise RuntimeError(
                "can not find formchk execution, check if formchk is in PATH or specify its location in the configuration file.\n"
            )
    
        else:
            print(f"find formchk execution: {self.formchk_exe}")

    def _check_xmvb_env(self):
        self.xmvb_exe = find_tool("xmvb")
        if not self.xmvb_exe:
            raise RuntimeError(
                "can not find xmvb execution, check if xmvb is in PATH or specify its location in the configuration file.\n"
            )
        else:
            print(f"find xmvb execution: {self.xmvb_exe}")

    def generate_gjf_from_xyz(self, basis: str, charge: int = 0, spin: int = 0):
        xyz_path = self.filepath.with_suffix(".xyz")
        mol = gto.M(
            atom=str(xyz_path),
            basis=basis,
            charge=charge,
            spin=spin,
        )
        gn = GaussianNBO(self.nbo_gjf_name, mol)
        gn.write_gjf(self.mem, self.nproc)
        print(f"Wrote Gaussian NBO input file to {self.nbo_gjf_name}.gjf with basis {basis}, charge {charge}, spin {spin}")

    def run_gaussian_to_fch(self, input_name: str):
        gaussian_cmd = f"{self.gaussian_exe} < {input_name}.gjf 1>{input_name}.out 2>{input_name}.err"
        print(f"Running Gaussian command: {gaussian_cmd}")
        proc_return = subprocess.run(gaussian_cmd, shell=True, check=False)
        if proc_return.returncode != 0:
            print(f"Gaussian execution failed with return code {proc_return.returncode}. Check {input_name}.err for details.")
            raise RuntimeError(f"Gaussian execution failed for {input_name}.gjf")
        
        formchk_cmd = f"{self.formchk_exe} {input_name}.chk {input_name}.fch"
        print(f"Running formchk command: {formchk_cmd}")
        proc_return = subprocess.run(formchk_cmd, shell=True, check=False)
        if proc_return.returncode != 0:
            print(f"formchk execution failed with return code {proc_return.returncode}. Check {input_name}.err for details.")
            raise RuntimeError(f"formchk execution failed for {input_name}.gjf")

    def generate_nbo_to_xmi(self, 
                       basis: str, 
                       nae: int=0, 
                       nao: int=0, 
                       active_orbital_atom: list[list[int]]=[], 
                       threshold: float=1.9, 
                       reorder: bool=False, 
                       atom_slice: bool=False):
        
        fchname = Path(f"{self.nbo_gjf_name}.fch")

        mol = load_mol_from_fch(fchname)

        def split_into_chunks(lst, n):
            return [lst[i : i + n] for i in range(0, len(lst), n)]

        if active_orbital_atom:
            active_orbital_atom = split_into_chunks(active_orbital_atom, 2)

        # 检查nbo输出文件是否存在，是大写还是小写
        nbo_out_path_upper = Path(f"{self.nbo_gjf_name_upper}.37")
        nbo_out_path_lower = Path(f"{self.nbo_gjf_name}.37")
        if nbo_out_path_upper.exists():
            self.nbo_gjf_name_true = self.nbo_gjf_name_upper
        elif nbo_out_path_lower.exists():
            self.nbo_gjf_name_true = self.nbo_gjf_name
        else:
            raise RuntimeError(f"can not find NBO output file for {self.nbo_gjf_name}, may be Gaussian NBO calculation did not finish successfully.")

        wxp = XMVBNBO(self.nbo_gjf_name_true, mol)
        wxp.set_basis_set(basis)

        if active_orbital_atom and nae > 0 and nao > 0:
            wxp.set_active_space(nae, nao)
            wxp.set_active_orbital_atom(active_orbital_atom)
        elif active_orbital_atom:
            gaoi = wxp.get_active_orbital_indices_from_atom(active_orbital_atom)
            nao = len([j for i in active_orbital_atom for j in i])
            nae = len(gaoi) * 2
            wxp.set_active_space(nae, nao)
            wxp.set_active_orbital_atom(active_orbital_atom)
        elif nae > 0 and nao > 0:
            wxp.set_active_space(nae, nao)
        else:
            nae, nao = wxp.auto_select_active_space(threshold=threshold, auto_set=True)
            wxp.set_active_space(nae, nao)
        inact, act = wxp.split_inactive_active_orbitals()
        xmi_path = Path(f"{self.xmi_name}.xmi")
        wxp.write_xmi(inact, act, reorder=reorder, atom_slice=atom_slice, xmi_path=xmi_path)

        # return autovb_xmi_impl(self.xmi_name, mol, basis, nae, nao, active_orbital_atom, threshold, reorder, atom_slice)

    def run_xmvb(self):
        """
        if [ -z ${program_version} ]; then
    program_version="latest"
fi

PROGRAM_PATH="${XMVB_PROGRAM_PATH}/${program_version}"
        if [ ${SLURM_JOB_PARTITION} == "pc" ]; then
        PROGRAM="${PROGRAM_PATH_PC}/bin/xmvb"
    elif [ ${SLURM_JOB_PARTITION} == "6226r" ]; then
        #module load intel/19.1.114
        #module load libcint/git
        module load mpich/4.0.2-gcc13
        PROGRAM="${PROGRAM_PATH}/bin/xmvb"
    elif [ ${SLURM_JOB_PARTITION} == "slater" ]; then
        PROGRAM="${PROGRAM_PATH}/bin/xmvb"
    else
        module add gcc/13.2.0
        PROGRAM="${PROGRAM_PATH}/bin/xmvb"
    fi"""
        xmvb_path = '/share/apps/xmvb/latest/bin/xmvb'
        xmvb_cmd = f"{xmvb_path} -n {self.nproc} {self.xmi_name}.xmi 1> {self.xmi_name}.xmo  2> {self.xmi_name}.err"
        print(f"Running formchk command: {xmvb_cmd}")
        proc_return = subprocess.run(xmvb_cmd, shell=True, check=False)
        if proc_return.returncode != 0:
            print(f"formchk execution failed with return code {proc_return.returncode}. Check {self.xmi_name}.err for details.")
            raise RuntimeError(f"formchk execution failed for {self.xmi_name}.gjf")