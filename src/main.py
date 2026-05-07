import subprocess
import os
import re
import numpy as np
import datetime
from pathlib import Path

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from collections import Counter

from .constants import FLOAT_RE, BASIS_FUNCTION_DICT, D_ORBITAL_3TO4, D_ORBITAL_4TO3, F_ORBITAL_3TO4, F_ORBITAL_4TO3, SUPPORTED_METHODS, STRU_CHOICES
from .utils import (
    read_gvb_pair_information,
    array_to_orb,
    read_gamess_dat, 
    read_dat_orbital,
    make_xmvb_format_text, 
    replace_col_orbital_numbers, 
    replace_row_orbital_numbers,
    get_orbital_atom_contribution,
    pyscf_to_xyz,
    read_nbo_orbital,
    find_executable_in_env,
    find_tool,
    print_subroutine,
    print_warning,
)
from .writers import write_xmi_file, write_gjf_nbo_file

from mokit.lib.gaussian import load_mol_from_fch
from pyscf import gto

if TYPE_CHECKING:
    from .readers import NBOOrbital

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
    stru: str = "full"
    sort: bool = False
    novb: bool = False
    guess: str = "nbo"
    active_order: str = "default"
    nbo_file: Path = None
    draw_xmo: bool = False
    draw_rumer: bool = False

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

        # guess参数可选值：nbo, pnbo
        if self.guess not in ("nbo", "pnbo"):
            raise ValueError("VBSettings: 'guess' must be 'nbo' or 'pnbo'")

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
class XMIData:
    '''
    定义生成XMVB .xmi文件所需的数据结构，包含分子信息、轨道信息、VB设置等
    '''
    molecule_name: str
    method: str
    stru_type: str
    int_type: str
    iscf: int
    nae: int
    nao: int
    basis_set: str
    sort: bool
    orb_section: str
    geo_section: str
    init_guess_section: str

# GVB GI orbital可以通过mokit的一个接口生成，但是很难用
# gen_cf_orb(datname='xxxx.dat',ndb=10,nopen=0)
# ndb为双占的离域轨道数量，nopen是开壳层的单占轨道数量
# 详情查看 https://doc.mokit.xyz/chap4-6.html#4634-gen_cf_orb

class GVBGI:
    # GI ORBITAL格式：
    # 基函数编号 原子符号 原子序号 基函数类型  数值1   数值2
    # 1  C  1  S    0.000004  -0.000012
    # 基函数类型包括： S, X, Y, Z, XX, YY, ZZ, XY, XZ, YZ
    # 对应原始基函数： S, PX, PY, PZ, DXX, DYY, DZZ, DXY, DXZ, DYZ
    # XXX  1
    # YYY  2
    # ZZZ  3
    # XXY  4
    # XXZ  5
    # YYX  6
    # YYZ  7
    # ZZX  8
    # ZZY  9
    # XYZ  10
    # F_ORBITAL_3TO4 = {1: 1, 2: 7, 3: 10, 4: 2, 5: 3, 6: 4, 7: 8, 8: 6, 9: 9, 10: 5}
    # F_ORBITAL_4TO3 = {1: 1, 2: 4, 3: 5, 4: 6, 5: 10, 6: 8, 7: 2, 8: 7, 9: 9, 10: 3}

    def __init__(
            self, 
            input_path: Path, 
            active_orbital: int=0, 
            active_electron: int=0, 
            all_orbital_number: int=0, 
        ):
        print('init GVBGI class')
        self.input_path = input_path
        self.molecule_name = input_path.stem
        self.formula_name = ''
        self.active_orbital = active_orbital
        self.active_electron = active_electron
        self.all_orbital_number = all_orbital_number
        self.inactive_orbital = self.all_orbital_number - self.active_orbital

    def find_file_path(self,stem:str, suffix:str) -> list[Path]:
        directory = self.input_path.parent

        # 正则：匹配 suffix 前最后一个数字
        pattern = re.compile(rf"(\d+)(?={re.escape(suffix)}$)")

        results = []

        for file in directory.iterdir():
            if file.is_file() and file.name.startswith(stem) and file.suffix == f".{suffix}":
                # 提取文件名中的所有数字
                nums = re.findall(r"\d+", file.stem)
                if nums:
                    last_num = int(nums[-1])
                    results.append((file, last_num))

        return results

    def main_get_gi_data(self):
        self.gi_orbital_matrix = self.read_gi_orbital()
        self.total_basis_functions = self.get_total_basis_functions()
        self.atom_labels = self.get_atom_labels(self.gi_orbital_matrix[0, :, :])
        self.dxx_indices = self.get_dxx_indices()
        self.fxxx_indices = self.get_fxxx_indices()
        self.all_pair_text = self.get_all_pair_text()
        file_head = f'{self.gi_orbital_matrix.shape[1]}  ' * self.gi_orbital_matrix.shape[0] * 2
        self.all_pair_text = file_head + '\n' + self.all_pair_text
        self.write_gvbgi()

    def main_get_dat_data(self):
        self.dat_text = self.read_gamess_dat()
        self.dat_orbital_matrix = self.read_dat_orbital(self.dat_text)
        # 对活性轨道最大编号进行截断
        if self.all_orbital_number:
            self.dat_orbital_matrix = self.dat_orbital_matrix[:self.all_orbital_number, :]
            print(f"Truncated dat_orbital_matrix to first {self.all_orbital_number} orbitals.")
        self.all_dat_text = self.get_all_dat_text()
        file_head = f'{self.dat_orbital_matrix.shape[1]}  ' * self.dat_orbital_matrix.shape[0]
        self.all_dat_text = file_head + '\n' + self.all_dat_text
        self.write_xmvb_orb()

    def main_make_xmi(self):
        # 读取输入文件信息
        self.basis_set, self.charge, self.multiplicity, self.geometry_text, self.geometry = self.get_input_file()
        # 获得其他的输入输出文件信息
        self.out_path: Path = self.find_file_path(self.molecule_name,'gms')[0][0]
        self.dat_path: Path = self.find_file_path(self.out_path.stem,'dat')[0][0]
        self.fch_path: Path = self.find_file_path(self.out_path.stem,'fch')[0][0]
        print(f"Found output file: {self.out_path}, dat file: {self.dat_path}, fch file: {self.fch_path}")
        # 调用fch2inp读取.fch文件，生成GAMESS输入文件，这个才是需要读的dat文件
        # TODO pyscf -> XMVB
        system_cmd = f"fch2inp {self.fch_path}"
        print(f"Running command: {system_cmd}")
        os.system(system_cmd)
        self.dat_path: Path = self.fch_path.with_suffix('.inp')
        print(f"Updated dat file path to: {self.dat_path}")

        # 读取GVB pair信息，选活性空间和轨道
        gvb_pairs = read_gvb_pair_information(self.out_path)
        need_pair = [] # 活性轨道对列表
        self.gvb_pair = gvb_pairs.shape[0]
        # pair[6]是反键占据数
        for pair in gvb_pairs:
            if pair[6] > 0.02:
                need_pair.append(int(pair[0]))
                print(f"Selected active pair: {int(pair[0])} with antibonding occupation {pair[6]:.4f}")
        print(f"Total active pairs selected: {len(need_pair)}")
        self.active_orbital = len(need_pair) * 2
        self.active_electron = len(need_pair) * 2
        print(f"Active orbitals: {self.active_orbital}, Active electrons: {self.active_electron}")
        unselected_gvb_pairs = len(gvb_pairs) - len(need_pair)
        gvb_max_orbital_number = int(gvb_pairs[-1][2])
        frozen_orbital = gvb_max_orbital_number - self.gvb_pair * 2
        self.inactive_orbital = frozen_orbital + unselected_gvb_pairs
        max_active_orbital_index = self.inactive_orbital + self.active_orbital
        print(f"Inactive orbital: {self.inactive_orbital} Max active orbital index: {max_active_orbital_index}")

        # 读取GI ORBITAL信息（变换后的GVB轨道，类似原子轨道）
        self.gi_orbital_matrix = self.read_gi_orbital()
        self.total_basis_functions = self.get_total_basis_functions()
        self.atom_labels = self.get_atom_labels(self.gi_orbital_matrix[0, :, :])
        self.dxx_indices = self.get_dxx_indices()
        self.fxxx_indices = self.get_fxxx_indices()
        start_gvb_act = int((self.gvb_pair - (self.active_orbital / 2)))
        if not need_pair:
            self.all_pair_text = self.get_all_pair_text2(need_pair=range(1, self.gvb_pair+1)[start_gvb_act:])
        else:
            self.all_pair_text = self.get_all_pair_text2(need_pair=need_pair)
        file_head = f'{self.gi_orbital_matrix.shape[1]}  ' * self.active_orbital
        self.all_pair_text = file_head + '\n' + self.all_pair_text
        self.write_gvbgi()

        # 读取GAMESS .dat文件的轨道信息
        self.dat_text = self.read_gamess_dat()
        self.dat_orbital_matrix = self.read_dat_orbital(self.dat_text)
        # 对活性轨道最大编号进行截断，截断到活性空间最大轨道数
        self.dat_orbital_matrix = self.dat_orbital_matrix[:max_active_orbital_index, :]
        print(f"Truncated dat_orbital_matrix to first {max_active_orbital_index} orbitals(max active orbital index).")
        self.dat_orbital_matrix = replace_col_orbital_numbers(self.dat_orbital_matrix, self.dxx_indices)
        self.all_dat_text = self.array_to_orb(self.dat_orbital_matrix, reorder_d=False, reorder_f=False)
        file_head = f'{self.dat_orbital_matrix.shape[1]}  ' * self.dat_orbital_matrix.shape[0]
        self.all_dat_text = file_head + '\n' + self.all_dat_text
        self.write_any_suffix(self.all_dat_text, 'orb')

        # 重新截断，只保留非活性部分轨道
        self.inactive_orbital_matrix = self.dat_orbital_matrix[:self.inactive_orbital, :]
        print(f"Truncated dat_orbital_matrix to first {self.inactive_orbital} orbitals(inactive orbital).")
        inactive_orbital_text = self.array_to_orb(self.inactive_orbital_matrix, reorder_d=False, reorder_f=False)
        file_head = f'{self.inactive_orbital_matrix.shape[1]}  ' * self.inactive_orbital_matrix.shape[0]
        inactive_orbital_text = file_head + '\n' + inactive_orbital_text

        # for i, orb_matrix in enumerate(self.inactive_orbital_matrix):
        #     self.print_localized_orbitals_info(i+1, orb_matrix)
        print(self.atom_labels)
        self.write_any_suffix(inactive_orbital_text, 'inaorb')

        self.write_xmi('hao')

    def print_localized_orbitals_info(self, number: int, orbital: np.ndarray):
        """
        打印轨道的部分信息
        """
        # ao_labels = self.mol.ao_labels()
        # print(f"Orbital {i+1}:")
        # print(f"轨道 {number}:")
        square_sum_coeff = sum(abs(orbital)**2)
        # print("Square sum Coefficients:", square_sum_coeff)
        # print("系数平方和：", square_sum_coeff)
        # print("="*80)
        orbital_atom = []
        for j in range(len(self.atom_labels)):
            # 找到第j个原子的起始和结束索引
            atom_info: Dict = self.atom_labels[j+1]
            all_basis_index = list(atom_info['basis_functions'].keys())
            # print(all_basis_index)
            start_index = min(all_basis_index) - 1
            end_index = max(all_basis_index) - 1
            contribution = sum(abs(orbital[start_index:end_index])**2) # sum of the square of the coefficients of the basis functions of the atom
            # 计算该原子的基函数系数平方和
            if contribution / square_sum_coeff >= 0.1:
                # 如果原子的贡献大于 10%，则打印
                orbital_atom.append((j+1, atom_info['atom'], contribution / square_sum_coeff))
                # print(f"原子 {j+1} ({atom_info['atom']}) 贡献: {contribution / square_sum_coeff:.1%}")
                # print("系数:", orbital[start_index:end_index]) # 打印该原子的基函数系数
        # print("="*80,"\n")
        return orbital_atom

    def get_input_file(self) -> str:
        text = self.input_path.read_text(errors='ignore').splitlines()

        basis = None
        charge = None
        multiplicity = None
        geometry = []
        geometry_text = []

        state = "header"

        for line in text:
            line_stripped = line.strip()

            # 跳过空行
            if not line_stripped:
                if state == "header":
                    state = "blank"
                continue

            # 解析基组（在 #p ... 那行）
            if basis is None and line_stripped.startswith("#"):
                # 例如 "#p CASSCF/cc-pVDZ"
                if "/" in line_stripped:
                    basis = line_stripped.split("/")[-1].strip()
                continue

            # 解析电荷与多重度
            if state == "blank":
                parts = line_stripped.split()
                if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                    charge, multiplicity = map(int, parts)
                    state = "geom"
                    continue

            # 解析几何坐标
            if state == "geom":
                geometry_text.append(line_stripped)
                parts = line_stripped.split()
                if len(parts) >= 4:
                    atom = parts[0]
                    try:
                        x, y, z = map(float, parts[1:4])
                        geometry.append((atom, x, y, z))
                    except ValueError:
                        pass
                else:
                    # 几何段结束
                    break
        
        geometry_text = '\n'.join(geometry_text)
        print(f'Read input file:{self.input_path}, basis set: {basis}')
        # print(basis, charge, multiplicity, geometry_text, geometry)
        return basis, charge, multiplicity, geometry_text, geometry

    def get_atom_labels(self, gi_orbital_matrix: np.ndarray) -> Dict[int, Dict[str, object]]:
        ''''''
        atom_list = {}
        for basis_function_row in gi_orbital_matrix:
            if int(basis_function_row[2]) not in atom_list:
                basis_function_dict = {int(basis_function_row[0]):BASIS_FUNCTION_DICT[basis_function_row[3]]}
                atom_list[int(basis_function_row[2])] = {'atom':str(basis_function_row[1]), 'basis_functions':basis_function_dict}
            else:
                atom_list[int(basis_function_row[2])]['basis_functions'][int(basis_function_row[0])] = BASIS_FUNCTION_DICT[basis_function_row[3]]
        self.atom_labels = atom_list
        # 计算分子式
        formula_dict = {}
        for atom_info in atom_list.values():
            atom_symbol = atom_info['atom']
            if atom_symbol not in formula_dict:
                formula_dict[atom_symbol] = 1
            else:
                formula_dict[atom_symbol] += 1
        for atom_symbol, count in formula_dict.items():
            if count == 1:
                self.formula_name += atom_symbol
            else:
                self.formula_name += f"{atom_symbol}{count}"
        return atom_list

    def read_gamess_dat(self) -> str:
        return read_gamess_dat(self.dat_path)
    
    def read_dat_orbital(self, vec_text: str) -> np.ndarray:
        return read_dat_orbital(vec_text)

    def read_fch_orbital(self) -> np.ndarray:
        pass

    def array_to_orb(self, dat_orbital_matrix: np.ndarray, reorder_d: bool=True, reorder_f: bool=True) -> str:
        # 这和下面的get_all_dat_text功能是一样的
        return array_to_orb(dat_orbital_matrix, reorder_d, reorder_f, self.dxx_indices, self.fxxx_indices)

    def get_all_dat_text(self, reorder_d: bool=True, reorder_f: bool=True) -> str:
        """
        将从.dat文件中读取到的GAMESS轨道信息转换为XMVB的orb格式
        Args:
            reorder_d: bool 是否重排D轨道次序，默认True
            reorder_f: bool 是否重排F轨道次序，默认True
        Returns:
            str: 所有轨道的文本信息
        """
        all_pair = []
        for i, orb_matrix in enumerate(self.dat_orbital_matrix):
            title_line = f"# ORBITAL        {i+1}  NAO =    {len(orb_matrix)}"
            xmvb_text = make_xmvb_format_text(orb_matrix)
            # self.print_localized_orbitals_info(i+1, orb_matrix)
            all_pair.append(f"{title_line}\n{xmvb_text}\n")
        all_pair_text = ''.join(all_pair)
        if reorder_d:
            if not self.dxx_indices:
                print("No DXX basis functions found; skipping D orbital reordering.")
            else:
                all_pair_text = self.replace_orbital_numbers(all_pair_text, self.dxx_indices, 3, 'd')
                print("D orbitals reordered.")
        if reorder_f:
            if not self.fxxx_indices:
                print("No FXXX basis functions found; skipping F orbital reordering.")
            else:
                all_pair_text = self.replace_orbital_numbers(all_pair_text, self.fxxx_indices, 3, 'f')
                print("F orbitals reordered.")
        return all_pair_text

    def read_gi_orbital(self) -> np.ndarray:
        """
        从 GAMESS .out 中解析 GI ORBITALS 段，返回np.ndarray三维数组:
        Returns:
            np.ndarray: 形状为 (轨道对数, 基函数数目, 6) 的三维数组
        """
        text = self.out_path.read_text(errors='ignore')
        # 定位 GI ORBITALS 标题
        m_hdr = re.search(r'^\s*-+\s*\n\s*GI ORBITALS\s*\n\s*-+\s*$',
                        text, flags=re.MULTILINE)
        if not m_hdr:
            raise RuntimeError("未在文件中找到 'GI ORBITALS' 段")

        start_idx = m_hdr.end()
        tail = text[start_idx:]
        # 为每个 PAIR 暂存行并在末尾生成矩阵
        pair_rows: Dict[int, List[List[object]]] = {}
        # 轨道对编号
        cur_pair = None

        for line in tail.splitlines():
            # 终止条件
            if line.strip().startswith('ddikick.x:') or \
            'END OF ROHF-GVB SCF CALCULATION' in line:
                break

            # 新的 PAIR 标题
            m_pair = re.match(r'^\s*PAIR\s+(\d+)\s*$', line)
            if m_pair:
                cur_pair = int(m_pair.group(1))
                # 初始化该 PAIR 的行缓存
                pair_rows[cur_pair] = []
                continue

            if cur_pair is None:
                continue  # 还没进入任何一个 PAIR

            s = line.strip()
            if not s:
                continue

            # 跳过列号行，例如 "1          2"
            if re.match(r'^\d+\s+\d+\s*$', s):
                continue

            # 数据行
            toks = s.split()
            if len(toks) != 6:
                print(f'tokens 格式不正确: {toks}')
                continue

            c1, c2 = toks[-2], toks[-1]
            if FLOAT_RE.match(c1) and FLOAT_RE.match(c2):
                # 在追加到 pairs 后，同步累积到矩阵行，并构建当前 PAIR 的矩阵
                idx = int(toks[0])
                atom = toks[1]
                atom_idx = int(toks[2])
                shell = toks[3]
                row = [idx, atom, atom_idx, shell, float(c1), float(c2)]
                pair_rows[cur_pair].append(row)

        if not pair_rows:
            raise RuntimeError("未解析到任何 PAIR 数据（检查 GI ORBITALS 段格式）")
        # print(pair_rows)
        keys = sorted(pair_rows.keys())
        pair_matrix = np.array([pair_rows[k] for k in keys])
        # print(pair_matrix)
        # print(pair_matrix.shape)
        print(f"Parsed GI ORBITALS with {pair_matrix.shape[0]} PAIRs")
        return pair_matrix

    def get_total_basis_functions(self) -> int:
        return self.gi_orbital_matrix.shape[1]
    
    def get_dxx_indices(self) -> List[int]:
        '''
        获取所有DXX基函数的索引（从1开始计数）
        Returns:
            List[int]: DXX基函数的索引列表
        '''
        dxx_indices = []
        for i in range(self.gi_orbital_matrix.shape[1]):
            if self.gi_orbital_matrix[0, i, 3] == 'XX':
                dxx_indices.append(i + 1)
        if not dxx_indices:
            print("No DXX basis functions found.")
        else:
            print(f"Found DXX basis function indices: {dxx_indices}")
        return dxx_indices
    
    def get_fxxx_indices(self) -> List[int]:
        '''
        获取所有FXXX基函数的索引（从1开始计数）
        Returns:
            List[int]: FXXX基函数的索引列表
        '''
        fxxx_indices = []
        for i in range(self.gi_orbital_matrix.shape[1]):
            if self.gi_orbital_matrix[0, i, 3] == 'XXX':
                fxxx_indices.append(i + 1)
        if not fxxx_indices:
            print("No FXXX basis functions found.")
        else:
            print(f"Found FXXX basis function indices: {fxxx_indices}")
        return fxxx_indices

    def get_all_pair_text(self, reorder: bool=True, reorder_d: bool=True, reorder_f: bool=True) -> str:
        """
        获得所有轨道的文本信息
        Args:
            reorder: bool 是否重排轨道次序，按照原子序数从小到大，默认True
            reorder_d: bool 是否重排D轨道次序，默认True
            reorder_f: bool 是否重排F轨道次序，默认True
        Returns:
            str: 所有轨道的文本信息
        """
        all_pair = []
        for i, pair_matrix in enumerate(self.gi_orbital_matrix):
            tuple1, tuple2 = self.get_pair_text(pair_matrix, i+1)
            all_pair.append(tuple1)
            all_pair.append(tuple2)
        if reorder:
            # 按照原子序数从小到大重排轨道
            all_pair.sort(key=lambda x: x[1])
            print("Orbitals reordered by atomic index.")
        all_pair_text = ''.join(pair[0] for pair in all_pair)
        if reorder_d:
            if not self.dxx_indices:
                print("No DXX basis functions found; skipping D orbital reordering.")
            else:
                all_pair_text = self.replace_orbital_numbers(all_pair_text, self.dxx_indices, 3, 'd')
                print("D orbitals reordered.")
        if reorder_f:
            if not self.fxxx_indices:
                print("No FXXX basis functions found; skipping F orbital reordering.")
            else:
                all_pair_text = self.replace_orbital_numbers(all_pair_text, self.fxxx_indices, 3, 'f')
                print("F orbitals reordered.")
        return all_pair_text
    
    def get_all_pair_text2(self, reorder: bool=True, reorder_d: bool=True, reorder_f: bool=True, need_pair: list[int]=[]) -> str:
        """
        获得所有轨道的文本信息，使用数组进行重排
        Args:
            reorder: bool 是否重排轨道次序，按照原子序数从小到大，默认True
            reorder_d: bool 是否重排D轨道次序，默认True
            reorder_f: bool 是否重排F轨道次序，默认True
            need_pair (list[int]): 需要的轨道对索引列表，默认空表示全部
        Returns:
            str: 所有轨道的文本信息
        """
        all_pair = []
        for i, pair_matrix in enumerate(self.gi_orbital_matrix):
            if need_pair and (i + 1) not in need_pair:
                print(f'Skipping pair {i+1} as it is inactive space.')
                continue
            print(f'Processing pair {i+1}...')
            if reorder_d:
                if not self.dxx_indices:
                    print("No DXX basis functions found; skipping D orbital reordering.")
                else:
                    pair_matrix = replace_row_orbital_numbers(pair_matrix, self.dxx_indices, 3, 'd')
                    print("D orbitals reordered.")
            if reorder_f:
                if not self.fxxx_indices:
                    print("No FXXX basis functions found; skipping F orbital reordering.")
                else:
                    all_pair_text = replace_row_orbital_numbers(pair_matrix, self.fxxx_indices, 3, 'f')
                    print("F orbitals reordered.")
            tuple1, tuple2 = self.get_pair_text(pair_matrix, i+1)
            all_pair.append(tuple1)
            all_pair.append(tuple2)
        if reorder:
            # 按照原子序数从小到大重排轨道
            all_pair.sort(key=lambda x: x[1])
            print("Orbitals reordered by atomic index.")
        self.gvbgi_orb_atom_indices = [pair[1] for pair in all_pair]
        all_pair_text = ''.join(pair[0] for pair in all_pair)
        return all_pair_text

    def get_pair_text(self, matrix: np.ndarray, pair_index: int=0) -> tuple[tuple[str, int, str], tuple[str, int, str]]:
        """
        将一个 PAIR 的矩阵数据按原脚本转换，返回(第一列(轨道文本, 原子索引, 原子名称), 第二列(轨道文本, 原子索引, 原子名称))
        Args:
            matrix: np.ndarray 形状为 (基函数数目, 6) 的矩阵
            pair_index: int 轨道对索引，默认0
        Returns:
            tuple ([tuple[str, int, str], tuple[str, int, str]]): 第一列和第二列的(轨道文本, 原子索引, 原子名称)
        """
        def get_orbital_text(pair_arr: np.ndarray, pair_index: int, orbital_index: int) -> tuple[str, int, str]:
            """
            Returns:
                tuple[str, int, str]: (轨道文本, 原子索引, 原子名称)
            """
            pair_text = make_xmvb_format_text(pair_arr, per_line=4)
            # 解析它属于哪个原子
            idx = self.get_max_coefficient(pair_arr)[0]
            max_line = matrix[idx, :]
            atom_index = int(max_line[2])
            atom_name = str(max_line[1])
            max_str = f'atom {max_line[2]} {max_line[1]} {BASIS_FUNCTION_DICT[max_line[3]]}'
            # 标题行构建
            title = (f"# PAIR  {pair_index} orbital {orbital_index} -- {max_str}")
            pair_final_text = f'{title}\n{pair_text}\n'
            return pair_final_text, atom_index, atom_name
        
        pair1 = matrix[:, 4]
        pair1 = pair1.astype(float)
        pair2 = matrix[:, 5]
        pair2 = pair2.astype(float)
        tuple1 = get_orbital_text(pair1, pair_index, 1)
        tuple2 = get_orbital_text(pair2, pair_index, 2)
        return tuple1, tuple2

    def get_max_coefficient(self, arr: np.ndarray) -> tuple[int, float]:
        """
        获取 GI ORBITAL 矩阵中数值列的最大绝对值系数及其索引
        
        Args:
            arr (np.ndarray): 一维浮点数组。
        Returns:
            tuple[int, float]: (最大值索引, 最大绝对值系数)。
        """
        max = np.max(np.abs(arr))
        idx = np.argmax(np.abs(arr))
        return idx, max
        
    def replace_orbital_numbers(self, replace_text: str, orbital_index: list[int], replace_type: int = 3, orbital_type: str = 'd') -> str:
        """
        根据替换规则字典，将 ORBITAL 块中的编号进行替换。
        Args:
            replace_text: str 输入的轨道文本内容
            orbital_index: list[int] 首个轨道的编号，例如需要替换D轨道，DXX基函数位于第11号与22号，则传入[11, 22]
            replace_type: int 替换类型，3表示XMVB3.0->XMVB4.0，4表示4.0->3.0，默认3
            orbital_type: str 轨道类型，默认'd'
        Returns:
            str: 替换后的轨道文本内容
        """

        # 构建正则表达式，匹配轨道编号
        # (\s*-?\d+\.\d+\s+) 匹配轨道系数（包括可能的负号和小数点）
        # (\d+) 匹配编号
        # (\s|$) 确保编号后面是空格或行尾
        orbital_pattern = re.compile(r"(\s*-?\d+\.\d+\s+)(\d+)(\s|$)")

        # 构建替换规则字典
        replace_dict = {}
        if orbital_type.lower() == 'f':
            if replace_type == 3:
                need_dict = F_ORBITAL_3TO4
            elif replace_type == 4:
                need_dict = F_ORBITAL_4TO3
        elif orbital_type.lower() == 'd':
            if replace_type == 3:
                need_dict = D_ORBITAL_3TO4
            elif replace_type == 4:
                need_dict = D_ORBITAL_4TO3

        def build_shift_map(offset: int, base: dict[int, int]) -> dict[int, int]:
            """
            将 base 的键值整体平移 offset。
            例：offset=10, base={1:1,2:4,3:6,4:2,5:3,6:5}
            -> {11:11,12:14,13:16,14:12,15:13,16:15}
            """
            return {offset + k: offset + v for k, v in base.items()}

        for bf_index in orbital_index:
            # 合并每个起始 D 轨道对应的一组 6 个 D 基函数映射
            replace_dict.update(build_shift_map(bf_index, need_dict))

        def replace_match(match: re.Match) -> str:
            """
            替换匹配的编号，根据替换规则字典进行替换。
            """
            coefficient = match.group(1)  # 轨道系数
            number = int(match.group(2))  # 编号
            suffix = match.group(3)       # 空格或行尾
            # 根据替换规则字典替换编号，如果编号不在字典中，则保持不变
            new_number = replace_dict.get(number, number)
            return f"{coefficient}{new_number}{suffix}"

        replace_text_modified = orbital_pattern.sub(replace_match, replace_text)
        return replace_text_modified
    
    def get_xmi_orb_section(self) -> str:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本
        Returns:
            str: $orb 部分文本
        '''
        all_atom_number = len(self.atom_labels)
        orb_number_text = f'{all_atom_number}*{self.inactive_orbital} 1*{self.active_orbital}'
        if all_atom_number != 1:
            orb_inactive_text = f'1-{all_atom_number}\n' * self.inactive_orbital
        else:
            orb_inactive_text = '1\n' * self.inactive_orbital
        orb_active_text = ''
        for i in self.gvbgi_orb_atom_indices:
            orb_active_text += f'{i}\n'
        orb_active_text = orb_active_text.strip('\n')
        orb_text = f'{orb_number_text}\n{orb_inactive_text}{orb_active_text}'
        return orb_text
    
    def get_xmi_orb_section_hao(self) -> str:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，使用非活性HAO
        Returns:
            str: $orb 部分文本
        '''
        # 获得非活性部分文本
        inactive_orb_list = []
        for i, orb_matrix in enumerate(self.inactive_orbital_matrix):
            j = self.print_localized_orbitals_info(i+1, orb_matrix)
            orb_atom = []
            for _ in j:
                orb_atom.append(_[0])
            inactive_orb_list.append(orb_atom)
        print(inactive_orb_list)
        head_inactive_test = ''
        orb_inactive_text = ''
        for orb_atom in inactive_orb_list:
            head_inactive_test += f'{len(orb_atom)} '
            orb_inactive_text += f'{" ".join(str(i) for i in orb_atom)}\n'

        orb_number_text = f'{head_inactive_test} 1*{self.active_orbital}'
        orb_active_text = ''
        for i in self.gvbgi_orb_atom_indices:
            orb_active_text += f'{i}\n'
        orb_active_text = orb_active_text.strip('\n')
        orb_text = f'{orb_number_text}\n{orb_inactive_text}{orb_active_text}'
        return orb_text

    def write_xmi(self, orb_type: str='oeo') -> None:
        xmi_path = Path(f'{self.formula_name}.xmi')

        # 获取orb部分
        if orb_type == 'hao':
            orb_section = self.get_xmi_orb_section_hao()
        elif orb_type == 'oeo':
            orb_section = self.get_xmi_orb_section()

        # 获取初猜文件并组装
        inactive_orb_path = Path(f'{self.formula_name}.inaorb')
        active_orb_path = Path(f'{self.formula_name}.gvbgi')
        inactive_orb_text = inactive_orb_path.read_text(errors='ignore')
        active_orb_text = active_orb_path.read_text(errors='ignore')
        # 处理第一行的内容，将两个text的第一行拼起来
        inactive_lines = inactive_orb_text.splitlines(keepends=True)
        active_lines = active_orb_text.splitlines(keepends=True)
        # 防止文件可能为空
        def first_and_rest(lines):
            if not lines:
                return "", []
            return lines[0], lines[1:]
        # 拆分
        inactive_first, inactive_rest = first_and_rest(inactive_lines)
        active_first, active_rest = first_and_rest(active_lines)
        # 拼装初猜文本
        init_guess_text = (
            inactive_first.rstrip("\n") + active_first +
            "".join(inactive_rest) +
            "".join(active_rest).strip("\n")
        )

        xmi_text = f'''{self.formula_name} Created by autoVB(Loach1703@github) {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
$ctrl
vbscf
str=full
nao={self.active_orbital}
nae={self.active_electron}
iscf=5
iprint=3
orbtyp=hao
frgtyp=atom
int=libcint 
basis={self.basis_set}
itmax=2000
molden
$end

$orb
{orb_section}
$end

$geo
{self.geometry_text}
$end

$gus
{init_guess_text}
$end
'''

        with open(xmi_path, 'w') as f:
            f.write(xmi_text)
        print(f"Wrote XMVB .xmi to {xmi_path}")

    def write_gvbgi(self, file_name: str=None):
        if not file_name:
            file_name = f'{self.formula_name}.gvbgi'
        else:
            file_name = file_name + '.gvbgi'
        with open(file_name, 'w') as f:
            f.write(self.all_pair_text)
        print(f"Wrote GVB GI Orbitals data to {file_name}")

    def write_xmvb_orb(self):
        orb_path = self.out_path.with_suffix('.orb')
        with open(orb_path, 'w') as f:
            f.write(self.all_dat_text)
        print(f"Wrote XMVB .orb data to {orb_path}")
    
    def write_any_suffix(self, write_text: str,  suffix: str, file_name: str=None):
        if not file_name:
            file_name = f'{self.formula_name}.{suffix}'
        else:
            file_name = file_name + '.' + suffix
        with open(file_name, 'w') as f:
            f.write(write_text)
        print(f"Wrote .{suffix} data to {file_name}")

class XMVBNBO:
    """
    读取Pyscf的Mole对象和Gaussian NBO输出文件，提取轨道信息并转换为XMVB格式的写入器
    """
    def __init__(self, filename: str, mol: 'gto.Mole', input_data: autoVBInputData) -> None:
        '''
        读取Pyscf的Mole对象和Gaussian NBO输出文件，提取轨道信息并转换为XMVB格式的写入器
        Args:
            filename (str): 文件名（不带后缀）
            mol (pyscf.gto.Mole): Mole对象，包含分子信息
            input_data (autoVBInputData): 包含活性空间设置等输入数据的对象
        '''
        # 主要需要的是两个信息：pyscf的分子对象，以及轨道信息
        self.filename = filename
        if self.filename.endswith('_nbo'):
            self.origin_filename = filename[:-4]
        else:
            self.origin_filename = filename
        self.mol = mol
        self.input_data = input_data
        self.active_electron = input_data.vbsettings.nae
        self.active_orbital = input_data.vbsettings.nao
        self.basis_function_number = self.mol.nao_nr()
        self.dxx_indices = []
        self.fxxx_indices = []
        self._df_indices()

        # 检查nbo输出文件是否存在，是大写还是小写
        nbo_orb_path_upper = Path(f"{self.filename.upper()}.37")
        nbo_orb_path_lower = Path(f"{self.filename}.37")
        self.nbo_out_file = Path(f"{self.filename}.out")
        if nbo_orb_path_upper.exists():
            self.nbo_orb_file = nbo_orb_path_upper
        elif nbo_orb_path_lower.exists():
            self.nbo_orb_file = nbo_orb_path_lower
        else:
            raise RuntimeError(f"can not find NBO output file for {self.filename}, may be Gaussian NBO calculation did not finish successfully.")

        # 获取化学式
        # 1. 提取所有原子的符号列表 (例如: ['C', 'C', 'H', ...])
        symbols = [self.mol.atom_pure_symbol(i) for i in range(self.mol.natm)]
        # 2. 统计每个元素的个数
        counts = Counter(symbols)
        # 3. 拼接成化学式字符串 (如果数量是1则省略数字，例如 H1 变成 H)
        formula = "".join(f"{sym}{cnt if cnt > 1 else ''}" for sym, cnt in counts.items())
        self.formula = formula

        # 获取 XMVB 需要的几何坐标文本 (XYZ 格式)
        self.geometry_text = pyscf_to_xyz(self.mol)

        # 一些可能需要用到的：
        # 所有基函数的列表：int 原子序号（从0开始），str 原子符号+编号，str 基函数类型, str 基函数磁量子数，长度为基函数总数
        atom_labels: List[Tuple[int,str,str,str]] = self.mol.ao_labels(fmt=False)
        num_atoms = self.mol.natm
        # [壳层起始, 壳层结束, 起始基函数索引, 结束基函数索引]
        slices: List[List[int,int,int,int]] = self.mol.aoslice_by_atom()

        # 读取NBO轨道信息并处理
        self._read_orbital_from_nbo()
        # 根据DXX和FXXX基函数的索引重排轨道矩阵的行顺序
        self._change_orbital_order()
        # 将轨道矩阵分为占据轨道和虚轨道
        self._split_occupied_virtual()
        # 根据占据数对占据轨道从小到大进行排序
        self._sort_occupied_orbitals_by_occupation()

    def _check_active_space(self) -> None:
        '''
        检查活性空间设置，不合理会报错
        '''
        if not hasattr(self, 'active_orbital'):
            raise ValueError("Active orbital count is not set. Please set active orbital count to determine active orbitals.")
        if not hasattr(self, 'active_electron'):
            raise ValueError("Active electron count is not set. Please set active electron count to determine active electron.")
        if not isinstance(self.active_orbital, int) :
            raise ValueError("Active orbital count must be an integer.")
        if not isinstance(self.active_electron, int) :
            raise ValueError("Active electron count must be an integer.")
        if self.active_orbital < 1 or self.active_electron < 1:
            raise ValueError("Active orbital count and active electron count must be positive integers.")

    def _df_indices(self) -> None:
        '''
        内部方法，获取所有DXX和FXXX基函数的索引（从1开始计数）
        '''
        atom_bf_labels: List[Tuple[int,str,str,str]] = self.mol.ao_labels(fmt=False)
        for i,bf in enumerate(atom_bf_labels):
            if 'd' in bf[2] and bf[3] =='xx':
                self.dxx_indices.append(i + 1)
            elif 'f' in bf[2] and bf[3] == 'xxx':
                self.fxxx_indices.append(i + 1)

    def _read_orbital_from_nbo(self) -> None:
        '''
        内部方法，从NBO输出文件中读取轨道矩阵和占据数，存储在self.orbital_matrix和self.occupation_numbers中
        '''
        from .readers import GaussianNBOParser
        self.nbo_parser = GaussianNBOParser(self.nbo_out_file, self.nbo_orb_file, self.mol, debug=self.input_data.debug)

        if self.input_data.vbsettings.guess == 'pnbo':
            self.orbital_matrix = self.nbo_parser.pnbo_orbital_matrix
        else:
            self.orbital_matrix = self.nbo_parser.nbo_orbital_matrix

        self.occupation_numbers = self.nbo_parser.occupation_numbers
        self.orbital_atoms = self.nbo_parser.orbital_atoms

    def _change_orbital_order(self) -> None:
        '''
        内部方法，根据self.dxx_indices和self.fxxx_indices重排轨道矩阵的行顺序，存储在self.orbital_matrix中
        '''
        if self.dxx_indices:
            self.orbital_matrix = replace_col_orbital_numbers(self.orbital_matrix, self.dxx_indices)
        if self.fxxx_indices:
            self.orbital_matrix = replace_col_orbital_numbers(self.orbital_matrix, self.fxxx_indices, orbital_type='f')

    def _split_occupied_virtual(self) -> None:
        '''
        内部方法，将轨道矩阵分为占据轨道和虚轨道，分别存储在self.occupation_orbital_matrix和self.virtual_orbital_matrix中，存储occ_indices和vir_indices用于后续切片，存储occ_orb_atom和vir_orb_atom用于后续分析
        '''
        total_elec = self.mol.nelectron
        orb_number = int(total_elec / 2)
        self.occupation_orbital_matrix = self.orbital_matrix[:orb_number]
        self.virtual_orbital_matrix = self.orbital_matrix[orb_number:]
        self.occ_orb_atom = self.orbital_atoms[:orb_number]
        self.vir_orb_atom = self.orbital_atoms[orb_number:]
        self.occ_indices = list(range(orb_number))
        self.vir_indices = list(range(orb_number, self.orbital_matrix.shape[0]))
        # print(self.occ_indices, self.vir_indices)
        # print(self.occ_orb_atom, self.vir_orb_atom)
        print(f"Total electrons: {total_elec}, Occupied orbitals: {self.occupation_orbital_matrix.shape[0]}, Virtual orbitals: {self.virtual_orbital_matrix.shape[0]}")

    def _sort_occupied_orbitals_by_occupation(self) -> None:
        '''
        内部方法，根据占据数对占据轨道从小到大进行排序，存储在self.occupation_orbital_matrix和self.occupation_numbers中
        '''
        self.sorted_occ_indices = np.argsort(self.occupation_numbers[:self.occupation_orbital_matrix.shape[0]])
        self.sorted_occupation_orbital_matrix = self.occupation_orbital_matrix[self.sorted_occ_indices]
        self.sorted_occupation_numbers = self.occupation_numbers[self.sorted_occ_indices]
        self.sorted_occ_orb_atom = [
            self.occ_orb_atom[int(i)]
            for i in self.sorted_occ_indices
        ]

    def auto_select_active_space(self, threshold: float=1.8, auto_set=False) -> tuple[int, int]:
        '''
        根据NBO占据数自动选择活性空间，返回活性电子数, 活性轨道数
        Args:
            threshold (float): 活性空间选择的占据数阈值，选择占据数小于等于该阈值的轨道作为活性轨道
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            tuple: (tuple[int, int]): 活性电子数, 活性轨道数
        '''
        active_orbital = 0
        active_electron = 0
        active_orbital_indices = []
        active_orbital_matrix = []
        # 筛选占据数大于1且小于等于threshold的轨道
        valid_occupied_indices = np.where((self.occupation_numbers > 1.0) & (self.occupation_numbers <= threshold))[0]
        
        # 填充到 indices 和 matrix 中
        active_orbital_indices = valid_occupied_indices.tolist()
        active_orbital_matrix = self.orbital_matrix[valid_occupied_indices]

        # 调用 get_orbital_atom_contribution 判断这些轨道在哪些原子上
        # 返回结构类似 [[1, 2], [3, 4]]
        orb_atoms = get_orbital_atom_contribution(active_orbital_matrix, self.mol)
        orb_atoms_flat = [j for i in orb_atoms for j in i]

        for i, occ, j in zip(active_orbital_indices, self.occupation_numbers[valid_occupied_indices], range(len(active_orbital_indices))):
            atom_name_list = []
            for a in orb_atoms[j]:
                an = self.mol.atom_pure_symbol(a-1)
                atom_name_list.append(f'{an}{a}')
            atom_name = ','.join(atom_name_list)
            print(f"Selected NBO orbital {i+1}, Occupation number: {occ:.4f}, Atom(s): {atom_name}")

        # 计算出活性轨道数量（等于各条轨道包含的原子数） 活性电子数量（等于所需要使用的NBO轨道数量的两倍）
        active_orbital = len(orb_atoms_flat)
        active_electron = len(active_orbital_indices) * 2
        if auto_set == True:
            if active_orbital == 0 or active_electron == 0:
                raise ValueError("No active orbitals selected based on the given threshold. Please adjust the threshold or check the occupation numbers.")
            else:
                # 自动将该属性应用到类中
                print(f"Automatically setting active space: {active_electron} electrons / {active_orbital} orbitals")
                self.set_active_space(active_electron, active_orbital)
        # 自动覆盖成键原子关系网用于后续切片
        # self.set_active_orbital_atom(orb_atoms)
        
        return active_electron, active_orbital

    def auto_select_active_space_iter(self, auto_set=False) -> tuple[int, int]:
        '''
        通过最小大于1的占据数+0.1的方式选择活性轨道，返回活性电子数, 活性轨道数
        Args:
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            tuple: (tuple[int, int]): 活性电子数, 活性轨道数
        '''
        valid_occupations = self.occupation_numbers[self.occupation_numbers > 1.0]
        # 阈值不能超过1.96，基本可以排除大多数sigma键
        threshold = min(float(valid_occupations.min()) + 0.1, 1.96)
        nae, nao = self.auto_select_active_space(threshold=threshold, auto_set=False)
        # 检查选出的活性轨道数量
        print(f"Automatically selected active space with threshold {threshold:.2f}: {nae} electrons / {nao} orbitals.")
        # 如果活性轨道过多，则降低活性空间的选择阈值
        if nao >= 15 or nae >= 15:
            print(f"Warning: Automatically selected active space has  {nae} electrons / {nao} orbitals, trying to reduce the threshold to select fewer active orbitals......")
            for _ in range(100):
                if nao < 15:
                    break
                threshold -= 0.01
                nae, nao = self.auto_select_active_space(threshold=threshold, auto_set=False)
                print(f"Trying threshold {threshold:.2f}: {nae} electrons / {nao} orbitals")
        # 最终检查选出的活性空间是否合理，如果仍然过大则给出警告提示用户手动选择
        if nao >= 15 or nae >= 15:
            print_warning(f"Automatically selected active space has  {nae} electrons / {nao} orbitals, which may be too large for VB calculations. Consider manually selecting the active space.")
        if auto_set:
            print(f"Automatically setting active space: {nae} electrons / {nao} orbitals")
            self.set_active_space(nae, nao)
        return nae, nao
    
    def auto_select_active_space_default(self, auto_set=False) -> tuple[int, int, List[int]]:
        '''
        默认选择活性空间的方式，目标轨道为：
        1. BD轨道，成键占据数小于1.96的轨道
        2. BD轨道，成键小于1.99，同时反键大于0.08，（这个类型可以称为BD-BD*）
        3. LP轨道，占据数小于1.96的轨道
        Args:
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            tuple: (tuple[int, int, List[int]]): 活性电子数, 活性轨道数, 活性轨道索引列表
        '''
        threshold_bd_bonding = 1.96
        threshold_bd_bonding_min = 1.9
        threshold_bd_bonding_star = 1.99
        threshold_bd_bonding_star_min = 1.9
        threshold_bd_antibonding = 0.08
        threshold_bd_antibonding_min = 0.15
        threshold_lp = 1.96
        threshold_lp_min = 1.9
        debug = self.input_data.debug
        nolp = self.input_data.vbsettings.nolp

        # 开始挑轨道，计算nae和nao，从self.nbo_parser中读取 NBO 轨道信息
        # 每个BD轨道占据数 > 1 对应 nae + 2，占据数 > 0 且 < 1 对应 nae + 1，每个BD轨道对应 nao + 2
        # 每个LP轨道占据数 > 1 对应 nae + 2，占据数 > 0 且 < 1 对应 nae + 1，每个LP轨道对应 nao + 1

        def electron_count(occupation: float) -> int:
            if occupation > 1.0:
                return 2
            if 0.0 < occupation <= 1.0:
                return 1
            return 0

        # LP轨道对应1个NAO，BD轨道对应2个NAO，对应的就是 connection的长度
        def orbital_nao(orbital: 'NBOOrbital') -> int:
            # if orbital.orbital_type == "LP":
            #     return 1
            # if orbital.orbital_type == "BD":
            #     return 2
            return len(orbital.connection)

        def add_orbital(selected_orbitals: Dict[int, Dict], orbital: 'NBOOrbital', reason: str) -> None:
            # 同一个BD轨道可能同时满足BD低占据数和BD-BD*规则，这里用index去重，只追加筛选原因。
            selected = selected_orbitals.setdefault(
                orbital.index,
                {
                    "orbital": orbital,
                    "reasons": [],
                },
            )
            if reason not in selected["reasons"]:
                selected["reasons"].append(reason)

        # 子函数：根据给定的阈值选择轨道，返回选中的轨道数量、活性轨道索引列表，以及选中轨道的详细信息（包含轨道对象和筛选原因）
        def select_orbitals(
            bd_bonding_threshold: float,
            bd_bonding_star_threshold: float,
            bd_antibonding_threshold: float,
            lp_threshold: float,
        ) -> tuple[int, int, List[int], List[Dict]]:
            selected_orbitals: Dict[int, Dict] = {}
            rule_hits = {
                "BD": [],
                "BD-BD*": [],
                "LP": [],
            }

            # 挑选满足 BD 轨道占据数小于 bd_bonding_threshold 的轨道
            for orbital in self.nbo_parser.nbo_data:
                if orbital.orbital_type == "BD" and 0.0 < orbital.occupancy < bd_bonding_threshold:
                    add_orbital(selected_orbitals, orbital, f"BD<{bd_bonding_threshold:.3f}")
                    rule_hits["BD"].append(orbital.index)
            
            # 挑选满足 BD-BD* 规则的轨道：成键占据数小于 bd_bonding_star_threshold，同时反键占据数大于 bd_antibonding_threshold
            for pair in self.nbo_parser.bond_antibond_pairs:
                if (
                    0.0 < pair.bond.occupancy < bd_bonding_star_threshold
                    and pair.antibond.occupancy > bd_antibonding_threshold
                ):
                    reason = f"BD-BD*: BD<{bd_bonding_star_threshold:.3f}, BD*>{bd_antibonding_threshold:.3f}"
                    add_orbital(selected_orbitals, pair.bond, reason)
                    rule_hits["BD-BD*"].append(pair.bond.index)

            # 挑选满足 LP 轨道占据数小于 lp_threshold 的轨道
            if not nolp:
                for orbital in self.nbo_parser.nbo_data:
                    if orbital.orbital_type == "LP" and 0.0 < orbital.occupancy < lp_threshold:
                        add_orbital(selected_orbitals, orbital, f"LP<{lp_threshold:.3f}")
                        rule_hits["LP"].append(orbital.index)
            else:
                if debug:
                    print("[DEBUG][default_as] NOLP option is set, skipping LP orbital selection.")

            selected_items = sorted(selected_orbitals.values(), key=lambda item: item["orbital"].index)

            # NBO输出中的轨道编号从1开始；矩阵切片需要0-based索引，所以这里统一减一
            active_indices = [item["orbital"].index - 1 for item in selected_items]
            nae = sum(electron_count(item["orbital"].occupancy) for item in selected_items)
            nao = sum(orbital_nao(item["orbital"]) for item in selected_items)
            if debug:
                print(
                    f"[DEBUG][default_as] thresholds: BD<{bd_bonding_threshold:.3f}, "
                    f"BD-BD*: BD<{bd_bonding_star_threshold:.3f} and BD*>{bd_antibonding_threshold:.3f}, "
                    f"LP<{lp_threshold:.3f}"
                )
                print(f"[DEBUG][default_as] rule hits before dedupe: {rule_hits}")
            return nae, nao, active_indices, selected_items

        def under_limit(selected_nae: int, selected_nao: int) -> bool:
            return selected_nae < 15 and selected_nao < 15

        nae, nao, active_indices, selected_items = select_orbitals(
            threshold_bd_bonding,
            threshold_bd_bonding_star,
            threshold_bd_antibonding,
            threshold_lp,
        )

        # 打印选出的活性轨道数量，以及对应几号轨道
        print(
            f"Automatically selected active space by default thresholds: "
            f"{nae} electrons / {nao} orbitals."
        )
        print(
            f"Default thresholds: BD<{threshold_bd_bonding:.3f}, "
            f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
            f"LP<{threshold_lp:.3f}"
        )
        if selected_items:
            for item in selected_items:
                orbital: 'NBOOrbital' = item["orbital"]
                atom_name = ",".join(
                    f"{self.mol.atom_pure_symbol(atom - 1)}{atom}"
                    for atom in orbital.connection
                )
                reason_text = "; ".join(item["reasons"])
                antibond_occ_text = ""
                if orbital.orbital_type == "BD":
                    pair = self.nbo_parser.bond_antibond_pair_by_bond_index.get(orbital.index)
                    if pair is not None:
                        antibond_occ_text = f", BD* occ={pair.antibond.occupancy:.5f}"
                print(
                    f"Selected NBO orbital {orbital.index}: "
                    f"{orbital.orbital_type}({orbital.orbital_number}) "
                    f"occ={orbital.occupancy:.5f}{antibond_occ_text}, atom(s): {atom_name}, reason: {reason_text}"
                )
        else:
            print("No active orbitals selected by default thresholds.")

        # 如果 nae/nao 过多（大于15），则降低活性空间的选择阈值，每次降低0.01，直到 nae/nao 小于15
        # 降低阈值的逻辑（1，2，3每一步都尝试重新选取活性空间）：
        # 1. LP轨道，占据数阈值降低0.01（下限为1.9）
        # 2. BD-BD*轨道，成键阈值降低0.005，反键阈值提高0.005（下限为1.9，上限为0.15）
        # 3. BD轨道，占据数阈值降低0.01（下限为1.9）
        # 1-3如此循环，直到达到合理的活性空间大小或者所有阈值都降低到下限仍然过大则停止
        while not under_limit(nae, nao):
            changed = False

            if threshold_lp > threshold_lp_min:
                threshold_lp = max(threshold_lp_min, round(threshold_lp - 0.01, 10))
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_lp,
                )
                print(
                    f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                    f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                    f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                )
                if under_limit(nae, nao):
                    break

            if threshold_bd_bonding_star > threshold_bd_bonding_star_min or threshold_bd_antibonding < threshold_bd_antibonding_min:
                if threshold_bd_bonding_star > threshold_bd_bonding_star_min:
                    threshold_bd_bonding_star = max(threshold_bd_bonding_star_min, round(threshold_bd_bonding_star - 0.005, 10))
                if threshold_bd_antibonding < threshold_bd_antibonding_min:
                    threshold_bd_antibonding = min(threshold_bd_antibonding_min, round(threshold_bd_antibonding + 0.005, 10))
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_lp,
                )
                print(
                    f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                    f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                    f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                )
                if under_limit(nae, nao):
                    break

            if threshold_bd_bonding > threshold_bd_bonding_min:
                threshold_bd_bonding = max(threshold_bd_bonding_min, round(threshold_bd_bonding - 0.01, 10))
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_lp,
                )
                print(
                    f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                    f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                    f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                )
                if under_limit(nae, nao):
                    break

            if not changed:
                break
            
        # 最终检查选出的活性空间是否合理，如果仍然过大则给出警告提示用户手动选择
        if not under_limit(nae, nao):
            print_warning(f"Automatically selected active space has {nae} electrons / {nao} orbitals, which may be too large for VB calculations. Consider manually selecting the active space.")

        print(f"Final default active space: {nae} electrons / {nao} orbitals")
        print(f"Final default active orbital indices (0-based): {active_indices}")
        self.set_active_indices(active_indices)
        if auto_set:
            if nae == 0 or nao == 0:
                raise ValueError("No active orbitals selected by default rules. Please manually select the active space or check the NBO occupation numbers.")
            print(f"Automatically setting active space: {nae} electrons / {nao} orbitals")
            self.set_active_space(nae, nao)
        
        # 返回 nae, nao, 以及活性轨道的indices(self.active_indices)
        return nae, nao, self.active_indices

    def set_active_space(self, active_electron: int, active_orbital: int) -> None:
        '''
        设置活性空间的轨道数和电子数
        Args:
            active_electron (int): 活性电子数
            active_orbital (int): 活性轨道数
        '''
        self.active_orbital = active_orbital
        self.active_electron = active_electron
        try:
            self._check_active_space()
        except ValueError as e:
            self.active_orbital = None
            self.active_electron = None
            raise e
        print(f"Active space set: {self.active_electron} electrons / {self.active_orbital} orbitals")

    def set_basis_set(self, basis_set: str) -> None:
        '''
        设置基组名称，注意如果NBO计算的基组与这里设置的基组不一致，可能会导致生成的XMVB文件与实际计算不匹配
        Args:
            basis_set (str): 基组名称，例如 'cc-pVDZ'
        '''
        self.basis_set = basis_set

    def set_active_indices(self, active_indices: List[int]) -> None:
        '''
        设置活性轨道的索引（注意是从0开始计数）
        Args:
            active_indices (List[int]): 活性轨道的索引列表
        '''
        self.active_indices = active_indices

    ##### 获取AOI的方法 #####

    def get_aoi(self, auto_set=False) -> Tuple[int, int, List[int]]:
        '''
        获取活性轨道的索引（注意是从0开始计数）
        Args:
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            Tuple[int, int, List[int]]: 活性电子数, 活性轨道数, 活性轨道索引列表
        '''
        active_orbital_atom = self.input_data.vbsettings.aoa
        aoa_bond = self.input_data.vbsettings.aoa_bond
        aoi = self.input_data.vbsettings.aoi
        threshold = self.input_data.vbsettings.threshold
        nae = self.active_electron
        nao = self.active_orbital
        # aoa参数，根据原子来判断活性轨道
        if active_orbital_atom:
            print(f"AOA active orbital atom list provided: {active_orbital_atom}, selecting active orbitals based on these atoms...")
            active_indices = self.get_active_orbital_indices_from_active_atoms(active_orbital_atom)

        # 保留aoa_bond参数的兼容性，但不推荐使用
        elif aoa_bond:
            print(f"AOA_BOND active orbital bond list provided: {aoa_bond}, selecting active orbitals based on these bonds...")
            active_indices = self.get_active_orbital_indices_from_aoa_bond(aoa_bond)

        # aoi 参数
        elif aoi:
            # aoi 输入是从 1 开始计数，这里统一转换成 0 开始
            aoi_1based = aoi
            if len(set(aoi_1based)) != len(aoi_1based):
                raise ValueError(f"Active Orbital Indices contains duplicated values: {aoi_1based}. Please provide unique orbital indices.")

            occ_norb = self.orbital_matrix.shape[0]
            active_indices = []
            for idx in aoi_1based:
                if idx < 1 or idx > occ_norb:
                    raise ValueError(f"Active Orbital Indices index {idx} is out of range. Valid range is [1, {occ_norb}] for occupied orbitals.")
                active_indices.append(idx - 1)

            # 通过给出的 aoi 轨道索引获取原子贡献
            selected_orbital_matrix = self.occupation_orbital_matrix[active_indices]

        # 手动指定了活性空间，但没有aoa
        elif nae > 0 and nao > 0:
            active_indices = self.get_active_orbital_indices()

        # 手动设置了挑选阈值
        elif threshold > 1:
            nae, nao = self.auto_select_active_space(threshold=threshold, auto_set=True)
            active_indices = self.get_active_orbital_indices()

        # 没有任何设置，自动挑选
        else:
            nae, nao, active_indices = self.auto_select_active_space_default(auto_set=True)

        derived_nae, derived_nao = self.get_as_from_aoi(active_indices)
        if nae != 0 and nao != 0:
            if nae != derived_nae or nao != derived_nao:
                raise ValueError(f"Active space derived from active orbital indices does not match expected values: derived ({derived_nae} electrons / {derived_nao} orbitals) vs expected ({nae} electrons / {nao} orbitals). Please check the selected active orbital indices and the corresponding occupation numbers.")
        else:
            nae, nao = derived_nae, derived_nao

        if auto_set:
            self.set_active_space(nae, nao)
            self.set_active_indices(active_indices)

        return nae, nao, active_indices

    def get_active_orbital_indices(self) -> List[int]:
        '''
        自动获取活性轨道的索引（注意是从0开始计数），根据NBO占据数判断活性轨道，选择NBO占据数小的轨道，并且选择活性轨道数的一半的轨道。
        Returns:
            List[int]: 活性轨道的索引列表
        '''
        # 根据NBO占据数判断活性轨道，选择最小的half_orb个
        # 一半的轨道数量
        half_orb = int(self.active_electron / 2)

        not_greater_than_one = self.occupation_numbers <= 1
        if not np.any(not_greater_than_one):
            # 如果数组里全都是大于 1 的，则考虑整个数组
            break_idx = len(self.occupation_numbers)
        else:
            break_idx = np.argmax(not_greater_than_one)
        relevant_values = self.occupation_numbers[:break_idx]
        # 活性轨道数量
        actual_half_orb = min(half_orb, len(relevant_values))
        # 活性轨道索引
        actorb_indices = np.argsort(relevant_values)[:actual_half_orb]
        print(f"Automatically selected active orbital indices based on occupation numbers: {actorb_indices}, corresponding occupation numbers: {self.occupation_numbers[actorb_indices]} include atom(s): {get_orbital_atom_contribution(self.orbital_matrix[actorb_indices], self.mol)}")
        return actorb_indices
    
    def get_active_orbital_indices_from_active_atoms(self, active_atom:List[int]) -> List[int]:
        '''
        获取活性轨道的索引（注意是从0开始计数），根据输入的活性原子列表判断活性轨道，选择对应原子上贡献较大的轨道。
        Args:
            active_atom (List[int]): 活性原子索引列表，例如 [1, 2, 3, 4] 表示活性的原子共有4个。注意索引是从1开始计数的。
        Returns:
            List[int]: 活性轨道的索引列表(从0开始计数)
        '''
        debug = self.input_data.debug
        active_atom_copy = active_atom[::1]
        actorb_indices = []
        sorted_indices = self.sorted_occ_indices
        bond_first = self.input_data.vbsettings.bond_first
        nolp = self.input_data.vbsettings.nolp
        # 弃用了get_orbital_atom_contribution的调用，直接读取的是 NBO 输出给出的连接方式
        # orb_text_o = get_orbital_atom_contribution(self.sorted_occupation_orbital_matrix, self.mol)
        orb_text_o = self.sorted_occ_orb_atom
        if debug:
            print(f"[DEBUG][aoa] requested active atoms: {active_atom}")
            print(f"[DEBUG][aoa] orbital atom contribution (occupation-sorted): {orb_text_o}")

        def consume_orbital_candidates(candidates: List[Tuple[int, int, List[int], float, str]], stage_name: str) -> None:
            if debug:
                print(f"[DEBUG][aoa][{stage_name}] start scanning {len(candidates)} candidates, remaining atoms={active_atom_copy}")
            for sorted_orb_idx, orb_index, pair, occ, type_of in candidates:
                need_orb = all(atom in active_atom_copy for atom in pair)
                if debug:
                    print(f"[DEBUG][aoa][{stage_name}] candidate sorted_idx={sorted_orb_idx}, orb_index={orb_index}, occ={occ:.6f}, pair={pair}, need_orb={need_orb}, remaining_before={active_atom_copy}")
                if not need_orb:
                    continue
                if orb_index not in actorb_indices:
                    actorb_indices.append(orb_index)
                for atom in pair:
                    if atom in active_atom_copy:
                        active_atom_copy.remove(atom)
                if debug:
                    print(f"[DEBUG][aoa][{stage_name}] selected orb_index={orb_index}, selected_list={actorb_indices}, remaining_after={active_atom_copy}")
                if not active_atom_copy:
                    if debug:
                        print(f"[DEBUG][aoa][{stage_name}] all active atoms are covered, stop scanning.")
                    break

        if bond_first or nolp:
            strategy = "bond-first" if bond_first else "NOLP"
            print(f"Selecting active orbitals based on {strategy} strategy...")
            one_atom_orb_indices: List[Tuple[int, int, List[int], float, str]] = []
            two_atom_orb_indices: List[Tuple[int, int, List[int], float, str]] = []
            for sorted_orb_idx, pair in enumerate(orb_text_o):
                orb_index = int(sorted_indices[sorted_orb_idx])
                pair_list = list(pair)
                occ = float(self.occupation_numbers[orb_index])
                type_of = self.nbo_parser.orbitals_type_list[orb_index]
                row = (sorted_orb_idx, orb_index, pair_list, occ, type_of)
                if len(pair_list) == 1:
                    one_atom_orb_indices.append(row)
                else:
                    # len(pair)>=2（包含典型双原子和极少数多中心情况）均优先搜索
                    two_atom_orb_indices.append(row)
            if debug:
                print(f"[DEBUG][aoa][bond_first] two_atom_orb_indices={[(i, idx, pair, type_of) for i, idx, pair, _, type_of in two_atom_orb_indices]}")
                print(f"[DEBUG][aoa][bond_first] one_atom_orb_indices={[(i, idx, pair, type_of) for i, idx, pair, _, type_of in one_atom_orb_indices]}")

            # 先搜索成键轨道（双原子/多原子）
            consume_orbital_candidates(two_atom_orb_indices, "two_atom_first")
            # 如果还有未覆盖活性原子，再补单原子轨道
            if active_atom_copy:
                if debug:
                    print(f"[DEBUG][aoa][bond_first] remaining atoms after two-atom scan: {active_atom_copy}, now scanning one-atom candidates.")
                if nolp:
                    raise ValueError("NOLP option is set, but there are still uncovered active atoms after scanning multi-atom orbitals. Consider adjusting the active space or checking the NBO occupation numbers.(most like the AOA numbers is odd)")
                consume_orbital_candidates(one_atom_orb_indices, "one_atom_second")
        else:
            all_candidates: List[Tuple[int, int, List[int], float, str]] = []
            for sorted_orb_idx, pair in enumerate(orb_text_o):
                orb_index = int(sorted_indices[sorted_orb_idx])
                pair_list = list(pair)
                occ = float(self.occupation_numbers[orb_index])
                type_of = self.nbo_parser.orbitals_type_list[orb_index]
                all_candidates.append((sorted_orb_idx, orb_index, pair_list, occ, type_of))
            consume_orbital_candidates(all_candidates, "normal")

        if debug:
            print(f"[DEBUG][aoa] final selected orbital indices={actorb_indices}, remaining active atoms={active_atom_copy}")
        # 如果循环结束后活性原子列表还不空，说明没有找到足够的活性轨道
        if active_atom_copy:
            raise ValueError(f"Could not find enough active orbitals for the given active atoms. Remaining active atoms without orbitals: {active_atom_copy}. Consider adjusting the active space or checking the NBO occupation numbers.")
        return actorb_indices
    
    def get_active_orbital_indices_from_aoa_bond(self, active_atom:List[List[int]]) -> List[int]:
        '''
        这是一个旧版本的函数
        获取活性轨道的索引（注意是从0开始计数），根据输入的活性原子列表判断活性轨道，选择对应原子上贡献较大的轨道。
        Args:
            active_atom (List[List[int]]): 活性原子索引列表，例如 [[1,2], [3,4]] 表示活性轨道共有4个，1,2号原子是一对成键的原子。注意索引是从1开始计数的。
        Returns:
            List[int]: 活性轨道的索引列表
        '''
        actorb_indices = []
        for pair in active_atom:
            sao_list = self.get_selected_atom_orbital(pair)
            sao_list.sort(key=lambda x: x[1])  # 按照NBO占据数排序，从小到大
            is_match = False
            for sao in sao_list:
                orb_index = sao[0]
                if orb_index not in actorb_indices:
                    actorb_indices.append(orb_index)
                    is_match = True
                    break
            if not is_match:
                raise ValueError(f"No suitable orbital found for active atom pair {pair}. Consider adjusting the active space.")

        return actorb_indices
    
    def get_as_from_aoi(self, aoi: List[int]) -> Tuple[int, int]:
        '''
        从 aoi 参数获取活性空间的电子数和轨道数
        Args:
            aoi (List[int]): 活性轨道索引列表（从1开始计数）
        Returns:
            Tuple[int, int]: 活性电子数, 活性轨道数
        '''
        print(f"Deriving active space from AOI: provided orbital indices (1-based) = {aoi}")
        active_indices = [idx - 1 for idx in aoi]  # 转换成0-based索引
        nao = 0
        nae = 0
        nbo_data = self.nbo_parser.nbo_data
        for idx in active_indices:
            occupancy = nbo_data[idx].occupancy
            if occupancy > 1.0:
                nae += 2
            elif 0.5 < occupancy <= 1.0:
                nae += 1
            else:
                pass
            nao += len(nbo_data[idx].connection)
            print(f"AOI orbital index {idx+1}: occupancy={nbo_data[idx].occupancy}, connected atoms={nbo_data[idx].connection}, cumulative NAE={nae}, NAO={nao}")
        print(f"Derived active space from AOI: {nae} electrons / {nao} orbitals.")
        return nae, nao

    #####  #####

    def get_atom_sliced_orbital(self, orbital: np.ndarray, atom_index: int) -> np.ndarray:
        '''
        将轨道向量中除指定原子外的其他基函数系数清零（制作单原子 VB 定域初猜）
        Args:
            orbital (np.ndarray): 一维轨道向量
            atom_index (int): 原子序号 (从1开始计)
        Returns:
            ndarray (np.ndarray): 切片截断后的新轨道向量
        '''
        slices = self.mol.aoslice_by_atom()
        new_orb = np.zeros_like(orbital)
        # atom_index 是从 1 开始的，所以对应 slices 需要减 1 获取其起止基函数索引 a0, a1
        a0, a1 = slices[atom_index - 1][2], slices[atom_index - 1][3] 
        new_orb[a0:a1] = orbital[a0:a1]
        return new_orb

    def get_orb_init_guess_mapping(self, orbital_matrix: np.ndarray) -> Dict[int, List[int]]:
        '''
        获取轨道矩阵中每条轨道对应的原子索引列表，构建轨道到原子的映射关系
        Args:
            orbital_matrix (np.ndarray): 二维轨道矩阵
        Returns:
            dict (Dict[int, List[int]]): 轨道到原子索引列表的映射关系，键为轨道索引（从0开始），值为对应的原子索引列表（从1开始）
        '''
        orb_text_o = get_orbital_atom_contribution(orbital_matrix, self.mol)
        mapping = {}
        for i, orb_atom in enumerate(orb_text_o):
            mapping[i] = orb_atom
        return mapping

    def get_rumer_order(self, active_atoms: List[int]) -> List[int]:
        '''
        获得Rumer图的原子顺序，输入为活性原子列表，输出为按照Rumer图顺序排列的活性原子列表
        Args:
            active_atoms (List[int]): 活性原子索引列表（从1开始计数）
        Returns:
            List[int]: 按照Rumer图顺序排列的活性原子索引列表（从1开始计数）
        '''
        from .rumer_active_graph import (
            infer_active_atom_order,
            print_order_process,
            write_active_graph_topology_svg,
        )
        print_subroutine("Entry Rumer Active Graph")

        num_atoms = self.mol.natm
        xyz_block = f"{num_atoms}\n\n{self.geometry_text}"
        CHARGE = 0
        HIDE_HYDROGENS = False

        result = infer_active_atom_order(
            xyz_block,
            active_atoms,
            charge=CHARGE,
            hide_hydrogens=HIDE_HYDROGENS,
        )
        if self.input_data.vbsettings.draw_rumer:
            svg_file = write_active_graph_topology_svg(result, Path.cwd(), f'{self.origin_filename}_rumer_graph.svg')
            print(f"Active graph topology SVG: {svg_file.resolve()}")
        print(f"Rumer Active Graph: Final order: {result.final_order}")
        if self.input_data.debug:
            print_order_process(result)
        return result.final_order

    def get_orb_section_inactive(self, orbital_matrix: np.ndarray) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，非活性部分
        Args:
            orbital_matrix (np.ndarray): 轨道矩阵
        Returns:
            tuple (Tuple[str, str]): $orb 部分文本，前端为头文本，后端为轨道文本
        '''
        orb_text_o = get_orbital_atom_contribution(orbital_matrix, self.mol)
        # 将orb部分文本格式化为XMVB需要的格式
        # 获得非活性部分文本
        head_text = ' '.join(str(len(i)) for i in orb_text_o)
        orb_text = ''
        for orb_atom in orb_text_o:
            orb_text += f'{" ".join(str(i) for i in orb_atom)}\n'

        orb_tuple = (head_text,orb_text)
        return orb_tuple

    def get_orb_section_active(self, orbital_matrix: np.ndarray) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，活性部分
        Args:
            orbital_matrix (np.ndarray): 轨道矩阵
        Returns:
            tuple (Tuple[str, str]): $orb 部分文本，前端为头文本，后端为轨道文本
        '''
        orb_text_o = get_orbital_atom_contribution(orbital_matrix, self.mol)
        # 将orb部分文本格式化为XMVB需要的格式
        # 获得活性部分文本
        head_text = f'1*{self.active_orbital}'
        orb_atom_list = []
        for orb_atom in orb_text_o:
            orb_atom_list += orb_atom

        # 重新排序部分
        # 根据aoa设置的原子顺序重新排序轨道
        if self.input_data.vbsettings.aoa and self.input_data.vbsettings.active_order == 'aoa':
            orb_atom_list_reordered = []
            for atom in self.input_data.vbsettings.aoa:
                if atom not in orb_atom_list:
                    raise ValueError(f"Active atom {atom} specified in aoa is not found in the active orbitals. Check active space settings.")
                orb_atom_list_reordered.append(atom)
            orb_atom_list = orb_atom_list_reordered[::1]

        # 重新排序
        if self.input_data.vbsettings.active_order == 'seq':
            orb_atom_list.sort()

        # 默认按照Rumer图的顺序重新排序
        elif self.input_data.vbsettings.active_order == 'rumer':
            orb_atom_list = self.get_rumer_order(orb_atom_list)

        # 在活性轨道的第一行添加注释，标明活性轨道开始
        orb_atom_list[0] = f"{orb_atom_list[0]}   # active orbital start here"

        orb_text = '\n'.join(str(i) for i in orb_atom_list) + '\n'
        orb_tuple = (head_text,orb_text)
        return orb_tuple

    def get_orb_section_total(self, inactive_orbital_matrix: np.ndarray, active_orbital_matrix: np.ndarray) -> str:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，包含非活性和活性轨道
        Args:
            inactive_orbital_matrix (np.ndarray): 非活性轨道矩阵
            active_orbital_matrix (np.ndarray): 活性轨道矩阵
        Returns:
            str (str): $orb 部分文本
        '''
        inactive_head, inactive_text = self.get_orb_section_inactive(inactive_orbital_matrix)
        active_head, active_text = self.get_orb_section_active(active_orbital_matrix)
        orb_number_text = f'{inactive_head} {active_head}'
        orb_text = f'{orb_number_text}\n{inactive_text}{active_text}'
        return orb_text

    def get_init_guess_inactive(self, orbital_matrix: np.ndarray) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $gus 文本，适合非活性轨道
        Args:
            orbital_matrix (np.ndarray): 二维轨道矩阵
        Returns:
            tuple (Tuple[str, str]): $gus 部分文本，前端为头文本，后端为轨道文本
        '''
        head_text = (' ' + str(orbital_matrix.shape[1])) * orbital_matrix.shape[0]
        orb_text = ''
        for i, orb in enumerate(orbital_matrix):
            orb_text += f'# ORBITAL        {i+1}  NAO =    {len(orb)}\n'
            orb_text += make_xmvb_format_text(orb, per_line=4)
            orb_text += '\n'
        return (head_text, orb_text)
    
    def get_init_guess_active(self, orbital_matrix: np.ndarray) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $gus 文本，适合活性轨道
        Args:
            orbital_matrix (np.ndarray): 二维轨道矩阵
        Returns:
            tuple (Tuple[str, str]): $gus 部分文本，前端为头文本，后端为轨道文本
        '''
        # 生成活性原子：活性轨道的映射关系
        oac = get_orbital_atom_contribution(orbital_matrix, self.mol)
        atoms_num = [j for i in oac for j in i]

        # 使用列表而不是字典，允许同一个原子对应多个活性轨道
        atom_orb_items: List[Tuple[int, np.ndarray]] = []
        for orb, oac_item in zip(orbital_matrix, oac):
            # 根据轨道数量看需要复制多少份，同时计算活性轨道数量
            for j in oac_item:
                if self.input_data.vbsettings.atom_slice:
                    new_orb = self.get_atom_sliced_orbital(orb, j)
                    atom_orb_items.append((j, new_orb))
                else:
                    atom_orb_items.append((j, orb))
        
        if len(atom_orb_items) != self.active_orbital:
            raise ValueError(f"Calculated active orbital count ({len(atom_orb_items)}) does not match expected active orbital count ({self.active_orbital}). Check active space settings.")
        
        # 根据aoa设置的原子顺序重新排序轨道
        if self.input_data.vbsettings.aoa and self.input_data.vbsettings.active_order == 'aoa':
            atom_orb_items_remaining = atom_orb_items[::1]
            atom_orb_items_reordered: List[Tuple[int, np.ndarray]] = []
            for atom in self.input_data.vbsettings.aoa:
                # 逐个消耗匹配项，支持重复原子
                matched_idx = None
                for idx, item in enumerate(atom_orb_items_remaining):
                    if item[0] == atom:
                        matched_idx = idx
                        break
                if matched_idx is None:
                    raise ValueError(f"Active atom {atom} specified in aoa is not found in the active orbitals. Check active space settings.")
                atom_orb_items_reordered.append(atom_orb_items_remaining.pop(matched_idx))
            atom_orb_items = atom_orb_items_reordered

        # 重新排序
        if self.input_data.vbsettings.active_order == 'seq':
            atom_orb_items = sorted(atom_orb_items, key=lambda x: x[0])
        
        # 默认按照Rumer图的顺序重新排序
        elif self.input_data.vbsettings.active_order == 'rumer':
            atom_order = self.get_rumer_order(atoms_num)
            atom_orb_items.sort(key=lambda x: atom_order.index(x[0]))

        head_text = (' ' + str(orbital_matrix.shape[1])) * len(atom_orb_items)
        orb_text = ''
        # 26/3/19 已经完成支持（7，8）这样的活性空间了
        for i, (atom, orb) in enumerate(atom_orb_items):
            orb_text += f'# ACTIVE ORBITAL        {i+1}  NAO =    {len(orb)} Localization in atom {atom}{self.mol.atom_pure_symbol(atom-1)}\n'
            orb_text += make_xmvb_format_text(orb, per_line=4)
            orb_text += '\n'
        return (head_text, orb_text)

    def get_selected_atom_orbital(self, atom_list: List[int]) -> List[Tuple[int, int, np.ndarray]]:
        '''
        获取选定原子的轨道，将会选择最有可能的与这些原子相关的轨道（即在这些原子上有较大贡献的轨道）。
        Args:
            atom_list (List[int]): 选定的原子索引列表（从1开始计数）
        Returns:
            list (List[Tuple[int, int, np.ndarray]]): [选定的轨道索引，轨道占据数，对应的轨道矩阵]组成的列表
        '''
        oac = get_orbital_atom_contribution(self.occupation_orbital_matrix, self.mol)
        target_set = set(atom_list)
        matching_indices = []
        return_list = []
        
        # 查找完全匹配原子列表的分子轨道索引
        for i, atoms in enumerate(oac):
            # 将列表转为集合，忽略顺序（例如 [1, 2] 和 [2, 1] 视为相同）
            if set(atoms) == target_set:
                matching_indices.append(i)
                
        if not matching_indices:
            raise ValueError(f"No orbital found matching atoms: {atom_list}")
        
        for idx in matching_indices:
            orb = self.occupation_orbital_matrix[idx]
            occ = float(self.occupation_numbers[idx])
            return_list.append((idx, occ, orb))

        return return_list

    def split_inactive_active_orbitals(self, active_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        根据活性轨道数和电子数，将轨道矩阵分为非活性轨道矩阵和活性轨道矩阵
        Returns:
            tuple (Tuple[np.ndarray, np.ndarray]): (非活性轨道矩阵, 活性轨道矩阵)
        '''
        # 切片出选中的项
        active_orbital_matrix = self.occupation_orbital_matrix[active_indices]
        # 获取剩下的项
        inactive_orbital_matrix = np.delete(self.occupation_orbital_matrix, active_indices, axis=0)

        return inactive_orbital_matrix, active_orbital_matrix

    def write_xmi(self, 
                  inactive_orbital_matrix: np.ndarray, 
                  active_orbital_matrix: np.ndarray,
                  xmi_path: Path = None,
                  ) -> None:
        self._check_active_space()
        vbsetting = self.input_data.vbsettings
        if not xmi_path:
            xmi_path = Path(f'{self.filename}.xmi')

        # 获取orb部分
        inactive_head, inactive_text = self.get_orb_section_inactive(inactive_orbital_matrix)
        active_head, active_text = self.get_orb_section_active(active_orbital_matrix)
        active_text = active_text.rstrip('\n')
        orb_number_text = f'{inactive_head} {active_head}'
        orb_section = f'{orb_number_text}\n{inactive_text}{active_text}'

        inact_head, inact_guess = self.get_init_guess_inactive(inactive_orbital_matrix)
        act_head, act_guess = self.get_init_guess_active(active_orbital_matrix)
        # 拼装初猜文本
        init_guess_text = (
            inact_head + act_head + '\n' +
            inact_guess +
            act_guess.strip("\n")
        )

        stru_type = vbsetting.stru
        if not stru_type:
            if self.active_orbital > 8:
                stru_type = 'cov'
            else:
                stru_type = 'full'

        xmidata = XMIData(
            molecule_name=self.filename,
            method=self.input_data.method,
            stru_type=stru_type,
            int_type=vbsetting.inte,
            iscf=vbsetting.iscf,
            nae=self.active_electron,
            nao=self.active_orbital,
            basis_set=self.basis_set,
            sort=vbsetting.sort,
            orb_section=orb_section,
            geo_section=self.geometry_text,
            init_guess_section=init_guess_text,
        )
        passthrough = self.input_data.xmi_passthrough
        write_xmi_file(xmi_path, xmidata, passthrough)
        print(f"Wrote XMVB .xmi to {xmi_path}")
    
    def write_xmi_blw(self, 
                  orbital_matrix: np.ndarray,
                  xmi_path: Path = None,
                  ) -> None:
        vbsetting = self.input_data.vbsettings
        nae = 2
        nao = 1
        stru_type = 'full'
        method = 'vbscf'
        if not xmi_path:
            xmi_path = Path(f'{self.filename}.xmi')

        # 获取orb部分
        inactive_head, inactive_text = self.get_orb_section_inactive(orbital_matrix)
        inactive_text = inactive_text.strip("\n")
        orb_number_text = f'{inactive_head}'
        orb_section = f'{orb_number_text}\n{inactive_text}'

        inact_head, inact_guess = self.get_init_guess_inactive(orbital_matrix)
        # 拼装初猜文本
        init_guess_text = (
            inact_head + '\n' +
            inact_guess.strip("\n")
        )

        xmidata = XMIData(
            molecule_name=self.filename,
            method=method,
            stru_type=stru_type,
            int_type=vbsetting.inte,
            iscf=vbsetting.iscf,
            nae=nae,
            nao=nao,
            basis_set=self.basis_set,
            sort=vbsetting.sort,
            orb_section=orb_section,
            geo_section=self.geometry_text,
            init_guess_section=init_guess_text,
        )
        passthrough = self.input_data.xmi_passthrough
        write_xmi_file(xmi_path, xmidata, passthrough)
        print(f"Wrote XMVB .xmi to {xmi_path}")

class autoVBMain:
    """
    autoVBMain类负责整个流程，包括检查环境、生成Gaussian NBO输入文件、从NBO输出中提取轨道信息、选择活性空间、以及最终生成XMVB输入文件。
    """

    def __init__(self, input_data: autoVBInputData):
        self.input_data = input_data
        filename = self.input_data.filename
        self.nbo_gjf_name = f"{filename}_nbo"
        self.xmi_name = f"{filename}_vb"
        self.blw_name = f"{filename}_blw"
        self._check_gaussian_env()
        if not self.input_data.vbsettings.novb:
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
        write_gjf_nbo_file(mol, self.nbo_gjf_name, mem=self.input_data.mem, nproc=self.input_data.nproc)
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
        aoa_bond = self.input_data.vbsettings.aoa_bond
        threshold = self.input_data.vbsettings.threshold

        # 检查方法设置，如果是LAM-DFVB或BOVB，强制调整相关参数
        if self.input_data.method.lower() == 'lam-dfvb':
            print("LAM-DFVB method detected, currently only BLYP functional is available.")
            self.input_data.method = 'lam-dfvb=blyp'
        if self.input_data.method.lower() == 'bovb':
            print("BOVB method detected, only iscf=2 will be used regardless of user input.")
            self.input_data.vbsettings.iscf = 2

        wxp = XMVBNBO(self.nbo_gjf_name, mol, self.input_data)
        wxp.set_basis_set(basis)
        self.wxp = wxp

        # BLW方法，本质是2e1o的VBSCF
        if self.input_data.method.lower() == 'blw':
            print("BLW method detected, no active space will be set.")
            orb_matrix = wxp.occupation_orbital_matrix
            xmi_path = Path(f"{self.blw_name}.xmi")
            wxp.write_xmi_blw(
                orb_matrix,
                xmi_path=xmi_path,
            )

        else:
            nae, nao, active_indices = wxp.get_aoi(auto_set=True)

            inact, act = wxp.split_inactive_active_orbitals(active_indices)
            xmi_path = Path(f"{self.xmi_name}.xmi")
            wxp.write_xmi(
                inact, 
                act,
                xmi_path=xmi_path,
            )

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
        if self.input_data.method.lower() == 'blw':
            filename = self.blw_name
        else:
            filename = self.xmi_name
        xmvb_cmd = f"{self.xmvb_exe} -n {self.input_data.nproc} {filename}.xmi 1> {filename}.xmo  2> {filename}.err"
        self.run_subprocess_command(xmvb_cmd, f"XMVB execution completed successfully for {filename}.xmi.", f"XMVB execution failed for {filename}.xmi, check {filename}.xmo for details.")

    def draw_xmo(self, xmo_file: Path, weight_table: str = 'cc', max_str: int = 20):
        from .draw_xmo.molecule_bond_variant_drawer import MoleculeBondVariantDrawer
        from .draw_xmo.xmo_drawer_input_converter import XmoToDrawerInputConverter
        from .draw_xmo.xmo_output_parser import XmoParser

        WEIGHT = weight_table
        MAX_STR = max_str
        output_dir = Path.cwd()
        hide_hydrogens = True

        parsed_data = XmoParser(xmo_file).parse()
        converter = XmoToDrawerInputConverter(
            parsed_data,
            output_dir,
            hide_hydrogens=hide_hydrogens,
            max_structures=MAX_STR,
            baseline_index=1,
            weight_table=WEIGHT,
        )
        drawer_input = converter.convert()

        drawer = MoleculeBondVariantDrawer(
            xyz_file=drawer_input.xyz_file,
            output_dir=output_dir,
            charge=0,
            active_bond_atom=drawer_input.active_bond_atom,
            active_space=drawer_input.active_space,
            active_space_color="#B00000",
            active_space_width=3.0,
            color_active_space=True,
            show_atom_labels=True,
            hide_hydrogens=hide_hydrogens,
            show_lone_pairs=True,
            write_individual_svgs=False,
        )
        result = drawer.draw()

        print(f"Read XMO from: {parsed_data.source_file.resolve()}")
        print(f"Generated XYZ: {drawer_input.xyz_file.resolve()}")
        print(f"Active orbital -> atom: {drawer_input.orbital_to_atom}")
        print(f"Weight table: {drawer_input.weight_table}")
        print(f"active_bond_atom: {drawer_input.active_bond_atom}")
        print(f"Drawn structures: {len(drawer_input.active_space)}")
        print(f"Output directory: {result.output_dir.resolve()}")
        for out_file in result.written_files:
            print(f" - {out_file.name}")

    def timed_call(self, step_name: str, func, *args, **kwargs):
        step_start = datetime.datetime.now()
        print(f"Start: {step_name} @ {step_start.strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)
        step_elapsed = (datetime.datetime.now() - step_start).total_seconds()
        print(f"End:   {step_name} | elapsed = {step_elapsed:.2f} s \n")
        return result

    def main(self):
        workflow_start = datetime.datetime.now()

        # 进行 NBO 计算，生成 .fch 文件供后续提取轨道信息使用
        if self.input_data.vbsettings.nbo_file:
            self.nbo_gjf_name = self.input_data.vbsettings.nbo_file.stem
            print(f"User specified the NBO file directly, skipping Gaussian NBO calculation. NBO file: {self.input_data.vbsettings.nbo_file}")
        else:
            print_subroutine("Entry Gaussian NBO Calculation")
            self.timed_call("generate_gjf_from_geo", self.generate_gjf_from_geo)
            self.timed_call("run_gaussian", self.run_gaussian, self.nbo_gjf_name)
            self.timed_call("run_formchk", self.run_formchk, self.nbo_gjf_name)

        # 生成 .xmi 文件
        print_subroutine("Entry NBO to XMI Conversion")
        self.timed_call("generate_nbo_to_xmi", self.generate_nbo_to_xmi)

        # VB计算是可选的，如果novb设置为True，则跳过VB计算步骤，仅生成 .xmi 文件
        if self.input_data.vbsettings.novb:
            print("VB calculation is skipped due to novb setting.(only generate xmi file from NBO orbitals)")
        else:
            print_subroutine("Entry XMVB Calculation")
            self.timed_call("run_xmvb", self.run_xmvb)

        # draw_xmo 调用
        if self.input_data.vbsettings.draw_xmo:
            print_subroutine("Entry draw_xmo")
            if self.input_data.method.lower() == 'blw':
                filename = self.blw_name
            else:
                filename = self.xmi_name
            xmo_path = Path(f"{filename}.xmo")
            if not xmo_path.exists():
                print(f"XMVB output file {xmo_path} not found, cannot draw XMO. Make sure XMVB calculation completed successfully.")
            else:
                self.timed_call("draw_xmo", self.draw_xmo, xmo_path, 'cc')

        workflow_elapsed = (datetime.datetime.now() - workflow_start).total_seconds()

        print_subroutine(f"autoVB workflow completed successfully!\nTotal workflow elapsed = {workflow_elapsed:.2f} s")
