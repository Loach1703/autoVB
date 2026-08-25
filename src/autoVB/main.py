import subprocess
import os
import re
import io
import math
import numpy as np
import datetime
from pathlib import Path

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import Counter
from contextlib import redirect_stdout

from .utils.utils import (
    make_xmvb_format_text, 
    replace_col_orbital_numbers, 
    get_orbital_atom_contribution,
    pyscf_to_xyz,
    find_executable_in_env,
    find_tool,
)
from .io.logging_config import get_logger
from .nbo.nbo import XMVBNBO

from mokit.lib.gaussian import load_mol_from_fch
from pyscf import gto

if TYPE_CHECKING:
    from .io.readers import NBOOrbital
    from .io.writers import XMIData
    from .io.xmo_output_parser import XmoParsedData


logger = get_logger(__name__)

def log_subroutine(message: str) -> None:
    """用 logger 输出 autoVB 子流程分隔信息。"""
    logger.info("=" * 40)
    logger.info(message)
    logger.info("=" * 40)

@dataclass
class VBSettings:
    '''
    设置VB计算的相关参数，如活性空间选择、重排序选项、原子切片选项等
    '''
    nae: int = 0
    nao: int = 0
    aoa: list[int] = field(default_factory=list) # 活性原子列表 active orbital atoms，例如 [1, 2, 3, 4] 表示活性的原子共有4个，索引从1开始计数
    aoa_bond: list[list[int]] = field(default_factory=list) # 旧的活性原子列表，包含每个轨道对应的原子，例如 [[1, 2], [2, 3], [3, 4]] 表示第一个轨道对应原子1和2，第二个轨道对应原子2和3，第三个轨道对应原子3和4
    aoi: list[int] = field(default_factory=list) # 活性轨道列表 active orbital indices，例如 [1, 2, 3] 表示活性的nbo轨道共有3个，索引从1开始计数
    inte: str = "libcint"
    iscf: int = 5
    atom_slice: bool = False
    bond_first: bool = False
    nolp: bool = False
    threshold: float = 0
    rethre: float = 0
    stru: str = "default"
    sort: bool = False
    novb: bool = False
    bovb_stream: bool = True
    nogvb: bool = False
    nojson: bool = False
    guess: str = "nbo"
    active_order: str = "default"
    nbo_file: Path = None
    draw_xmo: bool = False
    xmo2svg: Optional[str] = "optimized3d"
    svgweight: str = "both"
    hide_svg_labels: bool = True
    draw_rumer: bool = False
    nbo: str = 'hf' # nbo计算方法，默认为hf，可以设为b3lyp等

    def validate(self) -> None:
        """
        验证 VBSettings 各字段的合法性，发现非法值则抛出 ValueError。
        """
        if self.nae < 0:
            raise ValueError("VBSettings: 'nae' must be >= 0")
        if self.nao < 0:
            raise ValueError("VBSettings: 'nao' must be >= 0")

        # threshold 检查
        try:
            self.threshold = float(self.threshold)
        except Exception:
            raise ValueError("VBSettings: 'threshold' must be a number")
        if self.threshold < 0:
            raise ValueError("VBSettings: 'threshold' must be >= 0")

        # aoa 和 aoa_bond 不能同时设置
        if self.aoa and self.aoa_bond:
            raise ValueError("VBSettings: 'aoa' and 'aoa_bond' cannot both be set")

        # bond_first 是 aoa 的子选项，如果 bond_first=True 则必须设置 aoa
        if self.bond_first and not self.aoa:
            raise ValueError("VBSettings: 'bond_first' is a sub-option of 'aoa', it requires 'aoa' to be set")

        # guess参数可选值：nbo, pnbo, gvb
        if self.guess not in ("nbo", "pnbo", "gvb"):
            raise ValueError("VBSettings: 'guess' must be 'nbo', 'pnbo', or 'gvb'")

        if self.xmo2svg is not None:
            self.xmo2svg = self.xmo2svg.strip().lower()
            if self.xmo2svg not in (
                "rdkit",
                "pca",
                "optimized3d",
                "contact",
            ):
                raise ValueError(
                    "VBSettings: 'xmo2svg' must be 'rdkit', 'pca', "
                    "'optimized3d', 'contact', or None"
                )

        self.svgweight = self.svgweight.strip().lower()
        if self.svgweight not in ("cc", "lowdin", "both"):
            raise ValueError(
                "VBSettings: 'svgweight' must be 'cc', 'lowdin', or 'both'"
            )

        # acitve_order的动态默认值：如果有aoa，则默认按照aoa顺序，否则设为rumer
        if self.active_order == "default":
            if self.aoa:
                self.active_order = "aoa"
            else:
                self.active_order = "rumer"
        # active_order参数可选值：rumer, none, seq, aoa
        if self.active_order not in ("rumer", "none", "seq", "aoa"):
            raise ValueError("VBSettings: 'active_order' must be 'rumer', 'none', 'seq', or 'aoa'")
        if not self.aoa and self.active_order == "aoa":
            raise ValueError("VBSettings: 'active_order' set to 'aoa' requires 'aoa' to be set")

        self.validate_stru()
        self.validate_nbo_file()

    def validate_stru(self) -> None:
        # stru 检查：合法值为 'full', 'cov', 或 'ion(...)'
        if not isinstance(self.stru, str):
            raise ValueError("VBSettings: 'stru' must be a string")

        s = self.stru.strip().lower()
        if s in ("full", "cov", 'default'):
            return
        # ion(...) 格式校验，括号内可以是逗号分隔的整数列表或用短横连接的两个整数范围
        m = re.fullmatch(r"ion\(([^)]*)\)", s)
        if not m:
            raise ValueError("VBSettings: 'stru' must be 'full', 'cov' or 'ion(...)' with proper contents")
        
        inner = m.group(1).strip()
        if inner == "":
            raise ValueError("VBSettings: 'ion(...)' must contain indices or a range, e.g. ion(0,1,3) or ion(0-3)")

        # 检查是否是逗号分隔的整数列表
        if re.fullmatch(r"\s*\d+(\s*,\s*\d+)*\s*", inner):
            return

        # 或者是范围 a-b（两个整数，a<=b）
        m2 = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", inner)
        if m2:
            a = int(m2.group(1))
            b = int(m2.group(2))
            if a > b:
                raise ValueError("VBSettings: 'ion(a-b)' requires a <= b")
            return

        raise ValueError("VBSettings: invalid contents for ion(...). Use comma-separated integers or a single range a-b.")

    def validate_nbo_file(self) -> None:
        if self.nbo_file is not None:
            self.nbo_file = Path(self.nbo_file)
            # 检查文件是否存在
            if not self.nbo_file.is_file():
                raise ValueError(f"VBSettings: 'nbo_file' {self.nbo_file} does not exist or is not a file")

@dataclass
class XMIPassthrough:
    '''
    存储从输入 .xmi 中透传到输出 .xmi 的附加信息
    '''
    ctrl_extra_lines: list[str] = field(default_factory=list)
    str_section_text: Optional[str] = None

@dataclass
class autoVBInputData:
    '''
    定义输入数据结构，包含方法、基组、分子结构、计算参数等
    '''
    title: str
    filepath: Path
    filename: str
    method: str
    basis: str
    geometry: str
    charge: int = 0
    spin: int = 1
    mem: str = "4GB"
    nproc: int = 1
    debug: bool = False
    vbsettings: VBSettings = field(default_factory=VBSettings)
    xmi_passthrough: XMIPassthrough = field(default_factory=XMIPassthrough)

@dataclass
class OrbitalData:
    '''
    定义轨道数据结构，包含轨道矩阵、原子标签等
    填充该信息即可作为生成 .xmi 的数据
    '''
    index: int
    orbital_matrix: np.ndarray 
    atoms: list[int]
    occupation_numbers: float

class autoVBMain:
    """
    autoVBMain类负责整个流程，包括检查环境、生成Gaussian NBO输入文件、从NBO输出中提取轨道信息、选择活性空间、以及最终生成XMVB输入文件。
    """

    def __init__(self, input_data: autoVBInputData):
        self.input_data = input_data
        filename = self.input_data.filename
        self.nbo_gjf_name = f"{filename}_nbo"
        self.automr_gvb_name = f"{filename}_gvb"
        self.xmi_name = f"{filename}_vb"
        self.bovb_name = f"{filename}_bovb"
        self.blw_name = f"{filename}_blw"
        self._check_environment()

    def _use_bovb_stream(self) -> bool:
        return (
            self.input_data.method.lower() == 'bovb'
            and self.input_data.vbsettings.bovb_stream
            and not self.input_data.vbsettings.novb
        )

    def _check_environment(self):
        # 检查各种环境变量，但不是每个软件都需要的
        self._check_gaussian_env()
        self._check_formchk_env()
        self._check_xmvb_env()
        self._check_automr_env()
        self._check_gamess_env()

    def _check_gaussian_env(self):
        self.gaussian_exe = find_executable_in_env()
        logger.info(f"find Gaussian execution: {self.gaussian_exe}")

    def _check_formchk_env(self):
        # 检查 formchk
        self.formchk_exe = find_tool("formchk")
        logger.info(f"find formchk execution: {self.formchk_exe}")

    def _check_xmvb_env(self):
        self.xmvb_exe = find_tool("xmvb")
        logger.info(f"find XMVB execution: {self.xmvb_exe}")

    def _check_automr_env(self):
        self.automr_exe = find_tool("automr")
        logger.info(f"find MOKIT automr execution: {self.automr_exe}")

    def _check_gamess_env(self):
        # GAMESS的环境变量检查比较特殊，优先检查GMS环境变量，如果存在则直接使用；如果不存在，则尝试在系统路径中寻找rungms工具
        gms_env = os.environ.get("GMS")
        if gms_env:
            gms_path = Path(gms_env).expanduser()
            if gms_path.exists() and os.access(gms_path, os.X_OK):
                self.gamess_exe = str(gms_path)
                os.environ["GMS"] = self.gamess_exe
                logger.info(f"find GAMESS execution from GMS: {self.gamess_exe}")
                return
            gms_tool = find_tool(gms_env)
            if gms_tool:
                self.gamess_exe = gms_tool
                os.environ["GMS"] = self.gamess_exe
                logger.info(f"find GAMESS execution from GMS: {self.gamess_exe}")
                return

        self.gamess_exe = find_tool("rungms")
        logger.info(f"find GAMESS execution: {self.gamess_exe}")

    def read_nbo(self) -> 'XMVBNBO':
        '''
        读取NBO计算结果，从Gaussian的.fch文件中加载分子信息。
        '''
        from .io.writers import write_xmi_file
        fchname = Path(f"{self.nbo_gjf_name}.fch")
        mol = load_mol_from_fch(fchname)
        basis = self.input_data.basis

        wxp = XMVBNBO(self.nbo_gjf_name, mol, self.input_data)
        wxp.set_basis_set(basis)
        return wxp

    def generate_gjf_from_geo(self):
        basis = self.input_data.basis
        charge = self.input_data.charge
        spin = self.input_data.spin
        method = self.input_data.vbsettings.nbo
        # 电子数
        if spin == 1:
            pass
        elif spin % 2 == 0:
            pass
        elif spin > 1 and spin % 2 == 1:
            spin = 1
            logger.info(f"Spin multiplicity {self.input_data.spin} is odd and greater than 1, use R {method} not U{method} for NBO calculation.")
        mol = gto.M(
            atom=self.input_data.geometry,
            basis=basis,
            charge=charge,
            spin=spin - 1,  # Gaussian的自旋多重度是2S+1，而pyscf的spin是2S
        )
        from .io.writers import write_gjf_nbo_file
        write_gjf_nbo_file(mol, self.nbo_gjf_name, method=method, mem=self.input_data.mem, nproc=self.input_data.nproc)
        logger.info(f"Wrote Gaussian NBO input file to {self.nbo_gjf_name}.gjf with basis {basis}, charge {charge}, spin {spin}")

    def generate_automr_gvb(self):
        basis = self.input_data.basis
        charge = self.input_data.charge
        spin = self.input_data.spin
        mol = gto.M(
            atom=self.input_data.geometry,
            basis=basis,
            charge=charge,
            spin=spin - 1,  # Gaussian的自旋多重度是2S+1，而pyscf的spin是2S
        )
        from .io.writers import write_automr
        write_automr(
            mol,
            self.automr_gvb_name,
            method='GVB',
            mem=self.input_data.mem,
            nproc=self.input_data.nproc,
            # mokit_keywords="ist=1"
        )
        logger.info(f"Wrote MOKIT automr GVB input file to {self.automr_gvb_name}.gjf with basis {basis}, charge {charge}, spin {spin}")

    def generate_nbo_to_xmi(self):
        '''
        将Gaussian NBO计算的结果转换为XMVB输入文件，核心步骤包括：
        1. 从Gaussian的.fch文件中加载分子信息。
        2. 根据VBSettings中的参数设置，选择活性空间（NAE/NAO或基于原子选择活性轨道）。
        3. 使用XMVBNBO类处理NBO输出，进行轨道重排序和切片（如果需要）。
        4. 将处理后的轨道信息写入.xmi文件，供XMVB使用。
        '''
        from .io.writers import write_xmi_file
        wxp = self.wxp
        method = self.input_data.method.lower()

        if method == 'blw':
            log_subroutine("Entry BLW Method")
            logger.info("BLW method detected, no active space will be set.")
            xmi_path = Path(f"{self.blw_name}.xmi")
            xmidata = wxp.get_xmidata()

        else:
            log_subroutine(f"Entry auto active space selection")
            nae, nao, active_indices = wxp.get_aoi(auto_set=True)
            wxp.split_inactive_active_orbitals(active_indices)
            xmi_path = Path(f"{self.xmi_name}.xmi")
            log_subroutine(f"Entry write .xmi file")
            xmidata = wxp.get_xmidata()

        if self._use_bovb_stream():
            xmidata.method = 'vbscf'
            xmidata.iscf = 5
        
        passthrough = self.input_data.xmi_passthrough
        write_xmi_file(xmi_path, xmidata, passthrough)
        logger.info(f"Generated XMVB input file {xmi_path} successfully.")

    def generate_gvb_to_xmi(self):
        from .gvb.gvb import XMVBGVB
        from .io.writers import write_xmi_file
        wxp = self.wxp
        fch = self.get_gvb_filename()
        xg = XMVBGVB(fch, self.input_data, orbital_atoms=wxp.occ_orb_atom)
        xg.set_basis_set(self.input_data.basis)
        xmi_path = Path(f"{self.xmi_name}.xmi")
        xmidata = xg.get_xmidata()
        if self._use_bovb_stream():
            xmidata.method = 'vbscf'
            xmidata.iscf = 5
        write_xmi_file(xmi_path, xmidata, self.input_data.xmi_passthrough)
        logger.info(f"Generated XMVB input file {xmi_path} successfully from GVB orbitals.")

    def run_subprocess_command(self, command: str, success_message: str, error_message: str):
        logger.info(f"Running command: {command}")
        proc_return = subprocess.run(command, shell=True, check=False)
        if proc_return.returncode != 0:
            logger.error(f"{error_message} with return code {proc_return.returncode}. Check error output for details.")
            raise RuntimeError(error_message)
        else:
            logger.info(f"{success_message}")

    def run_gaussian(self, input_name: str):
        gaussian_cmd = f"{self.gaussian_exe} < {input_name}.gjf 1>{input_name}.out 2>{input_name}.err"
        self.run_subprocess_command(gaussian_cmd, f"Gaussian execution completed successfully for {input_name}.gjf.", f"Gaussian execution failed for {input_name}.gjf, check {input_name}.err and {input_name}.out for details.")

    def run_formchk(self, input_name: str):
        formchk_cmd = f"{self.formchk_exe} {input_name}.chk {input_name}.fch"
        self.run_subprocess_command(formchk_cmd, f"formchk execution completed successfully for {input_name}.chk.", f"formchk execution failed for {input_name}.chk, may be Gaussian calculation failed.")

    def run_automr_gvb(self):
        automr_cmd = f"{self.automr_exe} {self.automr_gvb_name}.gjf 1>{self.automr_gvb_name}.out 2>{self.automr_gvb_name}.err"
        self.run_subprocess_command(automr_cmd, f"MOKIT automr GVB execution completed successfully for {self.automr_gvb_name}.gjf.", f"MOKIT automr GVB execution failed for {self.automr_gvb_name}.gjf, check {self.automr_gvb_name}.out and {self.automr_gvb_name}.err for details.")

    def run_xmvb(self, filename: str | None = None):
        if filename is None:
            if self.input_data.method.lower() == 'blw':
                filename = self.blw_name
            else:
                filename = self.xmi_name
        xmvb_cmd = f"{self.xmvb_exe} -n {self.input_data.nproc} {filename}.xmi 1> {filename}.xmo  2> {filename}.err"
        self.run_subprocess_command(xmvb_cmd, f"XMVB execution completed successfully for {filename}.xmi.", f"XMVB execution failed for {filename}.xmi, check {filename}.xmo for details.")

    def get_gvb_filename(self) -> Path:
        automr_out = Path(f"{self.automr_gvb_name}.out")
        line = next(line for line in automr_out.read_text(errors='ignore').splitlines() if line.strip().startswith("$$GMS"))
        gvb_stem = Path(line.split()[1]).stem
        return Path(f"{gvb_stem}_s.fch")

    def draw_xmo(self, parsed_data: 'XmoParsedData', weight_table: str = 'cc', max_str: int = 20):
        '''
        使用XMVB的输出文件（.xmo）来绘制价键结构，核心步骤包括：
        1. 解析.xmo文件，提取分子结构、活性空间信息、以及每个价键结构的权重。
        2. 根据提取的信息，使用MoleculeBondVariantDrawer类来绘制
        3. 将绘制的结果保存到当前目录，并记录输出文件的信息。
        Args:
            parsed_data ('XmoParsedData'): 从.xmo文件解析得到的数据对象，包含分子结构、活性空间信息、以及每个价键结构的权重等。
            weight_table (str): 权重表的选择，默认为'cc'，可以是 'lowdin', 'inverse', 'renormalized'等。
            max_str (int): 最大绘制的价键结构数量，默认为20。
        Returns:
            None
        '''
        from .draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
        from .draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter

        WEIGHT = weight_table
        MAX_STR = max_str
        output_dir = Path.cwd()
        hide_hydrogens = True

        converter = XmoToDrawerInputConverter(
            parsed_data,
            output_dir,
            hide_hydrogens=hide_hydrogens,
            max_structures=MAX_STR,
            weight_table=WEIGHT,
        )
        drawer_input = converter.convert()
        hide_hydrogens = converter.hide_hydrogens

        drawer = MoleculeBondVariantDrawer(
            xyz_file=drawer_input.xyz_file,
            output_dir=output_dir,
            charge=int(parsed_data.ctrl_options.get("ncharge", self.input_data.charge)),
            active_bond_atom=drawer_input.active_bond_atom,
            active_space=drawer_input.active_space,
            baseline_unpaired_atoms=drawer_input.baseline_unpaired_atoms,
            active_space_color="#B00000",
            active_space_width=3.0,
            color_active_space=True,
            show_atom_labels=True,
            hide_hydrogens=hide_hydrogens,
            show_lone_pairs=True,
            write_individual_svgs=False,
        )
        result = drawer.draw()

        logger.info(f"Read XMO from: {parsed_data.source_file.resolve()}")
        logger.info(f"Generated XYZ: {drawer_input.xyz_file.resolve()}")
        logger.info(f"Active orbital -> atom: {drawer_input.orbital_to_atom}")
        logger.info(f"Weight table: {drawer_input.weight_table}")
        logger.info(f"active_bond_atom: {drawer_input.active_bond_atom}")
        logger.info(f"Bond perception mode: {drawer.bond_perception_mode}")
        logger.info(f"Drawn structures: {len(drawer_input.active_space)}")
        logger.info(f"Output directory: {result.output_dir.resolve()}")
        for out_file in result.written_files:
            logger.info(f" - {out_file.name}")

    def draw_xmo2svg(
        self,
        xmo_file: Path,
        projection: str,
        hide_svg_labels: bool = False,
        weight_table: str = "both",
    ):
        """调用 xmo2svg，按指定三维或二维排布方式生成 SVG。"""
        from .cli.xmo2svg import xmo2svg_file

        return xmo2svg_file(
            xmo_file,
            projection=projection,
            weight_table=weight_table,
            show_atom_labels=not hide_svg_labels,
            show_connection_labels=not hide_svg_labels,
        )

    def parser_xmo(self, xmo_file: Path, method: str | None = None) -> 'XmoParsedData':
        '''
        解析XMVB输出文件，提取相关信息。
        Args:
            xmo_file (Path): XMVB输出文件的路径，通常是.xmo文件。
        Returns:
            parsed_data (XmoParsedData): 解析后的数据对象。
        '''
        from .io.xmo_output_parser import XmoParser
        logger.info(f"Parsing XMVB output file {xmo_file} to extract information...")
        self.parsed_data = XmoParser(xmo_file).parse()
        method = (method or self.input_data.method).upper()

        if self.parsed_data.converged is True:
            logger.info(
                f"Successfully! {method} converged in "
                f"{self.parsed_data.steps} iterations "
                f"with ({self.parsed_data.nae},{self.parsed_data.nao}) active space."
            )
        elif self.parsed_data.converged is False:
            logger.warning(
                f"{method} failed to converge in "
                f"{self.parsed_data.steps} iterations "
                f"with ({self.parsed_data.nae},{self.parsed_data.nao}) active space."
            )
        else:
            logger.warning(
                f"Could not determine whether {method} converged "
                f"with ({self.parsed_data.nae},{self.parsed_data.nao}) active space."
            )
        self._log_xmo_energy_summary(self.parsed_data, method)
        return self.parsed_data

    def _log_xmo_energy_summary(
        self,
        parsed_data: 'XmoParsedData',
        method: str | None = None,
    ) -> None:
        '''
        记录XMVB能量摘要信息，根据不同的方法（VBPT2, LAM-DFVB等）记录不同的能量项。
        Args:
            parsed_data ('XmoParsedData'): 从.xmo文件解析得到的数据对象，包含能量信息等。
        Returns:
            None
        '''
        method = (method or self.input_data.method).upper()
        energy_labels = {
            "vbscf_energy": "VBSCF Energy",
            "total_energy": "Total Energy",
            "correlation_energy": "Correlation Energy",
            "lam_dfvb_energy": "LAM-DFVB Energy",
            "dfvb_correlation_energy": "DFVB Correlation Energy",
            "lambda_parameter": "LAMBDA Parameter",
        }
        method_energy_keys = {
            "VBPT2": (
                "vbscf_energy",
                "correlation_energy",
            ),
            "LAM-DFVB": (
                "vbscf_energy",
                "dfvb_correlation_energy",
                "lambda_parameter",
            ),
        }

        if parsed_data.energy is None:
            logger.warning(f"Cannot find {method.upper()} energy in XMVB output.")
            return

        energy_prefix = "E" if parsed_data.converged is not False else "Last reported E"
        logger.info(f"{energy_prefix}({method.upper()}) = {parsed_data.energy:.8f} a.u.")

        if method in method_energy_keys:
            for key in method_energy_keys[method]:
                if key not in parsed_data.energy_terms:
                    continue
                value = parsed_data.energy_terms[key]
                unit = "" if key == "lambda_parameter" else " a.u."
                logger.info(f"{energy_labels[key]} = {value:.8f}{unit}")

    def run_bovb_stream(self) -> Path:
        """先运行 VBSCF，再用收敛轨道生成并运行 BOVB 输入。"""
        from .cli.xmo2xmi import xmo2xmi_file
        from .io.writers import write_json_summary

        vbscf_xmo = Path(f"{self.xmi_name}.xmo")
        bovb_xmo = Path(f"{self.bovb_name}.xmo")

        log_subroutine("Entry XMVB VBSCF Calculation")
        self.timed_call("run_xmvb_vbscf", self.run_xmvb, self.xmi_name)
        vbscf_data = self.timed_call(
            "parser_xmo_vbscf", self.parser_xmo, vbscf_xmo, "vbscf"
        )
        if not self.input_data.vbsettings.nojson:
            self.timed_call(
                "write_vbscf_json_summary",
                write_json_summary,
                self.input_data,
                vbscf_data,
                Path(f"{self.xmi_name}.json"),
            )

        log_subroutine("Entry VBSCF to BOVB Conversion")
        self.timed_call(
            "xmo2xmi_bovb",
            xmo2xmi_file,
            Path(f"{self.xmi_name}.xmi"),
            Path(f"{self.bovb_name}.xmi"),
            "bovb",
            2,
        )

        log_subroutine("Entry XMVB BOVB Calculation")
        self.timed_call("run_xmvb_bovb", self.run_xmvb, self.bovb_name)
        # TODO: BOVB 的 breathing-orbital 编号会改变并跨行输出结构，
        # 等 XmoParser 支持该映射后再恢复 BOVB 的能量、权重和 JSON 解析。
        logger.warning(
            f"Skipping BOVB output parsing for {bovb_xmo}; "
            "the XMVB output file is kept for manual inspection."
        )
        return bovb_xmo

    def timed_call(self, step_name: str, func, *args, **kwargs):
        step_start = datetime.datetime.now()
        logger.debug(f"Start: {step_name} @ {step_start.strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)
        step_elapsed = (datetime.datetime.now() - step_start).total_seconds()
        logger.debug(f"End:   {step_name} | elapsed = {step_elapsed:.2f} s \n")
        return result

    def main(self):
        workflow_start = datetime.datetime.now()

        if self.input_data.vbsettings.guess == 'gvb':
            if not self.automr_exe:
                raise EnvironmentError("MOKIT automr executable not found in environment. Please install MOKIT and ensure 'automr' is in your PATH.")
            if not self.gamess_exe:
                raise EnvironmentError("GAMESS executable not found in environment. Please install GAMESS and ensure 'rungms' is in your PATH or set GMS environment variable.")
            if not self.input_data.vbsettings.nogvb:
                log_subroutine("Entry MOKIT automr GVB Calculation")
                self.timed_call("generate_automr_gvb", self.generate_automr_gvb)
                self.timed_call("run_automr_gvb", self.run_automr_gvb)
            else:
                logger.info("GVB calculation is skipped due to nogvb setting.")

        # 进行 NBO 计算，生成 .fch 文件供后续提取轨道信息使用
        if self.input_data.vbsettings.nbo_file:
            self.nbo_gjf_name = self.input_data.vbsettings.nbo_file.stem
            logger.info(f"User specified the NBO file directly, skipping Gaussian NBO calculation. NBO file: {self.input_data.vbsettings.nbo_file}")
        else:
            log_subroutine("Entry Gaussian NBO Calculation")
            if not self.gaussian_exe:
                raise EnvironmentError("Gaussian executable not found in environment. Please install Gaussian and ensure it is in your PATH.")
            if not self.formchk_exe:
                raise EnvironmentError("formchk executable not found in environment. Please ensure Gaussian's formchk tool is in your PATH.")
            self.timed_call("generate_gjf_from_geo", self.generate_gjf_from_geo)
            self.timed_call("run_gaussian", self.run_gaussian, self.nbo_gjf_name)
            self.timed_call("run_formchk", self.run_formchk, self.nbo_gjf_name)

        self.wxp = self.read_nbo()
        is_bovb = self.input_data.method.lower() == 'bovb'
        bovb_stream = self._use_bovb_stream()
        if bovb_stream:
            xmo_path = Path(f"{self.bovb_name}.xmo")
        elif self.input_data.method.lower() == 'blw':
            xmo_path = Path(f"{self.blw_name}.xmo")
        else:
            xmo_path = Path(f"{self.xmi_name}.xmo")
        # 生成 .xmi 文件
        if self.input_data.vbsettings.guess == 'gvb':
            log_subroutine("Entry GVB to XMI Conversion")
            self.timed_call("generate_gvb_to_xmi", self.generate_gvb_to_xmi)
        else:
            log_subroutine("Entry NBO to XMI Conversion")
            self.timed_call("generate_nbo_to_xmi", self.generate_nbo_to_xmi)

        # VB计算是可选的，如果novb设置为True，则跳过VB计算步骤，仅生成 .xmi 文件
        if self.input_data.vbsettings.novb:
            logger.info("VB calculation is skipped due to novb setting.(only generate xmi file from NBO orbitals)")
        else:
            log_subroutine("Entry XMVB Calculation")
            if not self.xmvb_exe:
                raise EnvironmentError("XMVB executable not found in environment. Please install XMVB and ensure it is in your PATH.")
            if bovb_stream:
                xmo_path = self.run_bovb_stream()
            else:
                self.timed_call("run_xmvb", self.run_xmvb)
                if is_bovb:
                    # TODO: 恢复支持 breathing-orbital 映射后的 BOVB 输出解析。
                    logger.warning(
                        f"Skipping BOVB output parsing for {xmo_path}; "
                        "the XMVB output file is kept for manual inspection."
                    )
                else:
                    self.timed_call("parser_xmo", self.parser_xmo, xmo_path)

        # draw_xmo 调用
        if self.input_data.vbsettings.draw_xmo and is_bovb:
            logger.warning("Skipping draw_xmo because BOVB output parsing is not supported yet.")
        elif self.input_data.vbsettings.draw_xmo:
            log_subroutine("Entry draw_xmo")
            # novb模式下没有生成xmo文件，因此需要先解析xmo文件，如果没有解析到数据则跳过绘制步骤
            if not hasattr(self, 'parsed_data'):
                try:
                    self.timed_call("parser_xmo", self.parser_xmo, xmo_path)
                except Exception as e:
                    logger.warning("No parsed .xmo data available for drawing. Skipping draw_xmo step.")
                    logger.warning("If you want to draw the .xmo, you can use command line tool 'draw_xmo' with the generated .xmo file after running XMVB.")
            self.timed_call("draw_xmo", self.draw_xmo, self.parsed_data, 'cc')

        if self.input_data.vbsettings.xmo2svg is not None and is_bovb:
            logger.warning("Skipping xmo2svg because BOVB output parsing is not supported yet.")
        elif self.input_data.vbsettings.xmo2svg is not None:
            log_subroutine("Entry xmo2svg")
            self.timed_call(
                "xmo2svg",
                self.draw_xmo2svg,
                xmo_path,
                self.input_data.vbsettings.xmo2svg,
                self.input_data.vbsettings.hide_svg_labels,
                self.input_data.vbsettings.svgweight,
            )

        if (
            not self.input_data.vbsettings.nojson
            and hasattr(self, 'parsed_data')
            and not is_bovb
        ):
            from .io.writers import write_json_summary
            self.timed_call("write_json_summary", write_json_summary, self.input_data, self.parsed_data)

        workflow_elapsed = (datetime.datetime.now() - workflow_start).total_seconds()

        log_subroutine(f"autoVB workflow completed successfully!\nTotal workflow elapsed = {workflow_elapsed:.2f} s")
