from pathlib import Path
import re
import os
from datetime import datetime
import numpy as np
from typing import List, Dict

from ..io.logging_config import get_logger
from ..utils.constants import BASIS_FUNCTION_DICT, FLOAT_RE, D_ORBITAL_3TO4, D_ORBITAL_4TO3, F_ORBITAL_3TO4, F_ORBITAL_4TO3
from ..utils.utils import (
    read_gvb_pair_information, 
    read_gamess_dat, 
    read_dat_orbital, 
    make_xmvb_format_text, 
    replace_col_orbital_numbers,
    array_to_orb,
    replace_row_orbital_numbers,
)

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
