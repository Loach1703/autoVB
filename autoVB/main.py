# TODO 重构代码块
import subprocess
import shutil
from pathlib import Path
import os
from pyscf import gto, scf
import re
from dataclasses import dataclass, field
from typing import Optional

from .utils import find_executable_in_env, find_tool, generate_fch_from_chk
from .autoVB import GaussianNBO, XMVBNBO
from .constants import SUPPORTED_METHODS
from mokit.lib.gaussian import load_mol_from_fch
from pyssian import GaussianInFile

STRU_CHOICES = ["full", "cov", "ion()"]

@dataclass
class VBSettings:
    '''
    设置VB计算的相关参数，如活性空间选择、重排序选项、原子切片选项等
    '''
    nae: int = 0
    nao: int = 0
    aoa: list[list[int]] = field(default_factory=list)
    reorder: bool = False
    atom_slice: bool = False
    threshold: float = 0
    stru: str = "full"

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

        self.validate_stru()

    def validate_stru(self) -> None:
        # stru 检查：合法值为 'full', 'cov', 或 'ion(...)'
        if not isinstance(self.stru, str):
            raise ValueError("VBSettings: 'stru' must be a string")

        s = self.stru.strip().lower()
        if s in ("full", "cov"):
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
    vbsettings: VBSettings = field(default_factory=VBSettings)

class autoVBMain:
    def __init__(self, input_data: autoVBInputData):
        self.input_data = input_data
        filename = self.input_data.filename
        self.nbo_gjf_name = f"{filename}_nbo"
        self.nbo_gjf_name_upper = f"{filename}_NBO"
        self.xmi_name = f"{filename}_vb"
        self._check_gaussian_env()
        self._check_xmvb_env()

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
                "can not find XMVB execution, check if xmvb is in PATH or specify its location in the configuration file.\n"
            )
        else:
            print(f"find XMVB execution: {self.xmvb_exe}")

    def generate_gjf_from_geo(self):
        basis = self.input_data.basis
        charge = self.input_data.charge
        spin = self.input_data.spin
        mol = gto.M(
            atom=self.input_data.geometry,
            basis=basis,
            charge=charge,
            spin=spin - 1,  # Gaussian的自旋多重度是2S+1，而pyscf的spin是2S
        )
        gn = GaussianNBO(self.nbo_gjf_name, mol)
        gn.write_gjf(self.input_data.mem, self.input_data.nproc)
        print(f"Wrote Gaussian NBO input file to {self.nbo_gjf_name}.gjf with basis {basis}, charge {charge}, spin {spin}")

    def generate_nbo_to_xmi(self):
        '''
        将Gaussian NBO计算的结果转换为XMVB输入文件，核心步骤包括：
        1. 从Gaussian的.fch文件中加载分子信息。
        2. 根据VBSettings中的参数设置，选择活性空间（NAE/NAO或基于原子选择活性轨道）。
        3. 使用XMVBNBO类处理NBO输出，进行轨道重排序和切片（如果需要）。
        4. 将处理后的轨道信息写入.xmi文件，供XMVB使用。
        '''
        fchname = Path(f"{self.nbo_gjf_name}.fch")
        mol = load_mol_from_fch(fchname)
        basis = self.input_data.basis
        nae = self.input_data.vbsettings.nae
        nao = self.input_data.vbsettings.nao
        active_orbital_atom = self.input_data.vbsettings.aoa
        threshold = self.input_data.vbsettings.threshold
        reorder = self.input_data.vbsettings.reorder
        atom_slice = self.input_data.vbsettings.atom_slice

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
        elif threshold > 1:
            nae, nao = wxp.auto_select_active_space(threshold=threshold, auto_set=True)
        else:
            nae, nao = wxp.auto_select_active_space_iter(auto_set=True)
            # nae, nao = wxp.auto_select_active_space(threshold=threshold, auto_set=True)
            # wxp.set_active_space(nae, nao)
        inact, act = wxp.split_inactive_active_orbitals()
        xmi_path = Path(f"{self.xmi_name}.xmi")
        wxp.write_xmi(inact, act, reorder=reorder, atom_slice=atom_slice, xmi_path=xmi_path, stru_type=self.input_data.vbsettings.stru)
        # return autovb_xmi_impl(self.xmi_name, mol, basis, nae, nao, active_orbital_atom, threshold, reorder, atom_slice)

    def run_subprocess_command(self, command: str, success_message: str, error_message: str):
        print(f"Running command: {command}")
        proc_return = subprocess.run(command, shell=True, check=False)
        if proc_return.returncode != 0:
            print(f"{error_message} with return code {proc_return.returncode}. Check error output for details.")
            raise RuntimeError(error_message)
        else:
            print(f"{success_message}")

    def run_gaussian(self, input_name: str):
        gaussian_cmd = f"{self.gaussian_exe} < {input_name}.gjf 1>{input_name}.out 2>{input_name}.err"
        self.run_subprocess_command(gaussian_cmd, f"Gaussian execution completed successfully for {input_name}.gjf.", f"Gaussian execution failed for {input_name}.gjf, check {input_name}.log for details.")

    def run_formchk(self, input_name: str):
        formchk_cmd = f"{self.formchk_exe} {input_name}.chk {input_name}.fch"
        self.run_subprocess_command(formchk_cmd, f"formchk execution completed successfully for {input_name}.chk.", f"formchk execution failed for {input_name}.chk, may be Gaussian calculation failed.")

    def run_xmvb(self):
        xmvb_cmd = f"{self.xmvb_exe} -n {self.input_data.nproc} {self.xmi_name}.xmi 1> {self.xmi_name}.xmo  2> {self.xmi_name}.err"
        self.run_subprocess_command(xmvb_cmd, f"XMVB execution completed successfully for {self.xmi_name}.xmi.", f"XMVB execution failed for {self.xmi_name}.xmi, check {self.xmi_name}.xmo for details.")

class autoVBInputParser:
    '''
    解析输入文件，提取必要的信息，如分子结构、基组、计算参数等
    '''
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.input_data = self.parse()
        settings = self.parse_autovb_options(self.input_data.title)
        self.input_data.vbsettings = settings

    def parse(self) -> autoVBInputData:
        with GaussianInFile(self.input_path) as input_file:
            input_file.read()
        # method和basis不会自动读取，原因是GaussianInFile不支持VB方法的读取，它会识别成一整个参数
        # 识别包含VB或/的行，提取method和basis
        cmd_line = input_file.commandline
        for i in cmd_line.items():
            key: str = i[0]
            if "/" in key:
                method_basis = key.split('/')
                method = method_basis[0].lower()
                basis = method_basis[1]
                if method not in SUPPORTED_METHODS:
                    raise ValueError(f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS}")
                break
        if not method or not basis:
            raise ValueError(f"Failed to parse method and basis from input file {self.input_path}. Ensure that the method and basis in the format 'method/basis'.")
        atvb_input = autoVBInputData(
            title=input_file.title,
            filepath=self.input_path,
            filename=self.input_path.stem,
            method=method,
            basis=basis,
            geometry=input_file.geometry,
            charge=input_file.charge,
            spin=input_file.spin,
            mem=input_file.mem,
            nproc=input_file.nprocs,
        )
        print(f"Parsed input file {self.input_path} successfully with method {method} and basis {basis}")

        return atvb_input
    
    def parse_value_by_type(self, raw: str, target_type, key: str):
        raw = raw.strip()
        # 布尔判断（支持 1/0/true/false/yes/no）
        if target_type is bool:
            return raw.lower() in ("1", "true", "yes", "y")
        # 整数
        if target_type is int:
            try:
                return int(raw)
            except Exception:
                return int(float(raw))
        # 浮点
        if target_type is float:
            return float(raw)
        # 列表（尝试解析数字或字符串列表）
        if getattr(target_type, "__origin__", None) is list or target_type is list:
            s = raw
            # 去除括号/中括号
            if s.startswith(("(", "[")) and s.endswith((")", "]")):
                s = s[1:-1]
            # 用分号或空白分割
            parts = [p for p in re.split(r'[;,\s]+', s) if p != ""]
            parsed = []
            for p in parts:
                if re.fullmatch(r'[+-]?\d+', p):
                    parsed.append(int(p))
                else:
                    try:
                        parsed.append(float(p))
                    except Exception:
                        parsed.append(p)
            if key == "aoa":
                if all(isinstance(x, int) for x in parsed):
                    return [parsed[i : i + 2] for i in range(0, len(parsed), 2)]
            return parsed
        # 默认当字符串返回（去掉外层单/双引号）
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            return raw[1:-1]
        return raw

    def parse_autovb_options(self, s: str) -> VBSettings:
        """
        解析形如: autovb{nae=4, nao=4}的字符串
        提取大括号内部的键值对，并把识别到的字段注入到 VBSettings dataclass 中。
        - 只注入 VBSettings 中已声明的字段，类型会尝试转换（int/float/bool/list/str）。
        - 未识别或非 VBSettings 字段会被忽略。
        """
        m = re.search(r"\{([^}]*)\}", s, re.DOTALL)
        if not m:
            return VBSettings()  # 无选项，返回默认

        inner = m.group(1)
        pattern = r'\s*(?:\([^()]*\)|[^,])+\s*'
        pair_list = [p.strip() for p in re.findall(pattern, inner) if p.strip()]
        settings = VBSettings()
        for pair in pair_list:
            if "=" not in pair:
                key = pair.strip()
                value = True
            else:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip(' ()')

            if not hasattr(settings, key):
                continue
            target_type = VBSettings.__annotations__.get(key, str)
            try:
                parsed_value = self.parse_value_by_type(value, target_type, key)
                setattr(settings, key, parsed_value)
                # print(f"Set VBSettings.{key} = {parsed_value} (parsed from '{value}')")
            except Exception as e:
                print(f"Warning: failed to parse value for key '{key}' with raw value '{value}'. Error: {e}. Skipping this option.")
                continue
        
        # 验证 VBSettings 合法性（若不合法会抛错并中止流程）
        settings.validate()

        return settings