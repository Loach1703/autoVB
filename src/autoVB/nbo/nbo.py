from __future__ import annotations

import io
import math
import numpy as np
from pathlib import Path

from collections import Counter
from contextlib import redirect_stdout
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from pyscf import gto

from ..utils.utils import (
    make_xmvb_format_text,
    replace_col_orbital_numbers,
    get_orbital_atom_contribution,
    pyscf_to_xyz,
)
from ..io.logging_config import get_logger

if TYPE_CHECKING:
    from ..main import autoVBInputData
    from ..io.readers import NBOOrbital
    from ..io.writers import XMIData


logger = get_logger(__name__)


def log_subroutine(message: str) -> None:
    """用 logger 输出 autoVB 子流程分隔信息。"""
    logger.info("=" * 40)
    logger.info(message)
    logger.info("=" * 40)

class XMVBNBO:
    """
    读取Pyscf的Mole对象和Gaussian NBO输出文件，提取轨道信息并转换为XMVB格式的写入器
    思路：
    1. 读取 Gaussian NBO输出文件，提取轨道信息
    2. 自动选择活性空间，或根据输入文件信息进行选择（如 AOA 参数等）
    3. 获得 AOI (Atomic Orbital Indices)，即需要作为活性空间的轨道
    4. 根据 AOI 拆分轨道，构造分块的轨道（可以分为occ，vir，act，ina），对应占据，虚，活性，非活性轨道
    5. 将非活性和活性轨道作为 XMVB 输入文件的依据，生成 .xmi 文件
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

    ##### 初始化的内部封装方法 #####

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

    def _change_orbital_order(self) -> None:
        '''
        内部方法，根据self.dxx_indices和self.fxxx_indices重排轨道矩阵的行顺序，存储在self.orbital_matrix中
        '''
        if self.dxx_indices:
            self.orbital_matrix = replace_col_orbital_numbers(self.orbital_matrix, self.dxx_indices)
        if self.fxxx_indices:
            self.orbital_matrix = replace_col_orbital_numbers(self.orbital_matrix, self.fxxx_indices, orbital_type='f')

    def _read_orbital_from_nbo(self) -> None:
        '''
        内部方法，从NBO输出文件中读取轨道矩阵和占据数，存储在self.orbital_matrix和self.occupation_numbers中
        '''
        from ..io.readers import GaussianNBOParser
        self.nbo_parser = GaussianNBOParser(self.nbo_out_file, self.nbo_orb_file, self.mol, debug=self.input_data.debug, spin="alpha")

        if self.input_data.vbsettings.guess == 'pnbo':
            # logger.info("Using PNBO orbitals as initial guess.")
            self.orbital_matrix = self.nbo_parser.pnbo_orbital_matrix
        else:
            # logger.info("Using NBO orbitals as initial guess.")
            self.orbital_matrix = self.nbo_parser.nbo_orbital_matrix

        # 如果是开壳层系统，除了读取alpha轨道的占据数
        if self.mol.spin > 0:
            # logger.info("Detected open-shell system, parsing both alpha and beta NBO occupation numbers...")
            # logger.info("Using alpha + beta NBO occupation numbers for open-shell active space selection.")
            self.nbo_parser_beta = GaussianNBOParser(self.nbo_out_file, self.nbo_orb_file, self.mol, debug=self.input_data.debug, spin="beta")
            # combined_occupation_numbers = self.nbo_parser.occupation_numbers * 2
            # self.nbo_parser.occupation_numbers = combined_occupation_numbers
            # for orbital, occupancy in zip(self.nbo_parser.nbo_data, combined_occupation_numbers):
            #     orbital.occupancy = float(occupancy)
            # self.nbo_parser.bond_antibond_pairs = self.nbo_parser.build_bond_antibond_pairs()
            # self.nbo_parser.bond_antibond_pair_by_bond_index = {
            #     pair.bond.index: pair for pair in self.nbo_parser.bond_antibond_pairs
            # }
            # self.nbo_parser.bond_antibond_pair_by_antibond_index = {
            #     pair.antibond.index: pair for pair in self.nbo_parser.bond_antibond_pairs
            # }
            # self.occupation_numbers = self.nbo_parser.occupation_numbers * 2
            # self.rectified_occupancy = self.nbo_parser.rectified_occupancy * 2

        else:
            logger.info("Detected closed-shell system.")

        self.occupation_numbers = self.nbo_parser.occupation_numbers
        self.rectified_occupancy = self.nbo_parser.rectified_occupancy
        self.orbital_atoms = self.nbo_parser.orbital_atoms
    
    ##### 处理轨道，构造分块（分为occ，vir，act，ina） #####

    def _split_occupied_virtual(self) -> None:
        '''
        内部方法，将轨道矩阵分为占据轨道和虚轨道，分别存储在self.occupation_orbital_matrix和self.virtual_orbital_matrix中，存储occ_indices和vir_indices用于后续切片，存储occ_orb_atom和vir_orb_atom用于后续分析
        '''
        total_elec = self.mol.nelectron
        # 向上取整轨道数
        orb_number = math.ceil(total_elec / 2)
        self.occupation_orbital_matrix = self.orbital_matrix[:orb_number]
        self.virtual_orbital_matrix = self.orbital_matrix[orb_number:]
        self.occ_orb_atom = self.orbital_atoms[:orb_number]
        self.vir_orb_atom = self.orbital_atoms[orb_number:]
        self.occ_indices = list(range(orb_number))
        self.vir_indices = list(range(orb_number, self.orbital_matrix.shape[0]))
        # print(self.occ_indices, self.vir_indices)
        # print(self.occ_orb_atom, self.vir_orb_atom)
        logger.info(f"Total electrons: {total_elec}, Occupied orbitals: {self.occupation_orbital_matrix.shape[0]}, Virtual orbitals: {self.virtual_orbital_matrix.shape[0]}")

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

    def split_inactive_active_orbitals(self, active_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        根据活性轨道索引，将占据轨道分为非活性轨道和活性轨道。
        同时保存两部分的轨道矩阵、原子贡献和原始占据轨道索引，便于后续分析。
        Returns:
            tuple (Tuple[np.ndarray, np.ndarray]): (非活性轨道矩阵, 活性轨道矩阵)
        '''
        active_index_set = set(active_indices)
        inactive_indices = [
            idx for idx in self.occ_indices
            if idx not in active_index_set
        ]

        self.active_indices = active_indices
        self.inactive_indices = inactive_indices
        # self.active_orbital_matrix = self.occupation_orbital_matrix[active_indices]
        self.active_orbital_matrix = self.orbital_matrix[active_indices]
        self.inactive_orbital_matrix = self.occupation_orbital_matrix[inactive_indices]
        self.active_orb_atom = [
            self.orbital_atoms[idx]
            for idx in active_indices
        ]
        self.inactive_orb_atom = [
            self.occ_orb_atom[idx]
            for idx in inactive_indices
        ]
        self.active_occupation_numbers = self.occupation_numbers[active_indices]
        self.inactive_occupation_numbers = self.occupation_numbers[inactive_indices]

        logger.info(
            f"Occupied orbitals: {self.occupation_orbital_matrix.shape[0]}, "
            f"Inactive orbitals: {self.inactive_orbital_matrix.shape[0]}, "
            f"Active orbitals: {self.active_orbital_matrix.shape[0]}"
        )
        if self.input_data.debug:
            logger.debug(f"Inactive orbital indices: {inactive_indices}, occupation numbers: {self.inactive_occupation_numbers}, atom contributions: {self.inactive_orb_atom}")
            logger.debug(f"Active orbital indices: {active_indices}, occupation numbers: {self.active_occupation_numbers}, atom contributions: {self.active_orb_atom}")

        return self.inactive_orbital_matrix, self.active_orbital_matrix

    ##### 自动选择活性空间 #####

    def auto_select_active_space_thre(self, threshold: float=1.8, auto_set=False) -> tuple[int, int, List[int]]:
        '''
        根据NBO占据数自动选择活性空间，返回活性电子数, 活性轨道数
        Args:
            threshold (float): 活性空间选择的占据数阈值，选择占据数小于等于该阈值的轨道作为活性轨道
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            tuple: (tuple[int, int, List[int]]): 活性电子数, 活性轨道数, 活性轨道索引列表
        '''
        active_orbital = 0
        active_electron = 0
        active_orbital_indices: list[int] = []
        # 筛选占据数大于1且小于等于threshold的轨道
        TOP = 0.02 if self.mol.spin == 0 else 0.01
        valid_occupied_indices = np.where((self.occupation_numbers > TOP) & (self.occupation_numbers <= threshold))[0]
        # 填充到 aoi 中
        active_orbital_indices = valid_occupied_indices.tolist()
        nbo_data = self.nbo_parser.nbo_data

        # 只是打印选出的活性轨道的信息，并没有业务逻辑上的作用
        for i in active_orbital_indices:
            atom_name_list = []
            for a in nbo_data[i].atoms:
                atom_name_list.append(f'{a[0]}{a[1]}')
            atom_name = ','.join(atom_name_list)
            logger.info(f"Selected NBO orbital {i+1}, Occupation number: {nbo_data[i].occupancy:.4f}, Atom(s): {atom_name}")

        # 计算出活性轨道数量，活性电子数量
        active_electron, active_orbital = self.get_as_from_aoi(active_orbital_indices)
        if auto_set == True:
            if active_orbital == 0 or active_electron == 0:
                raise ValueError("No active orbitals selected based on the given threshold. Please adjust the threshold or check the occupation numbers.")
            else:
                # 自动将该属性应用到类中
                logger.info(f"Automatically setting active space: {active_electron} electrons / {active_orbital} orbitals")
                self.set_active_space(active_electron, active_orbital)
        
        return active_electron, active_orbital, active_orbital_indices

    def auto_select_active_space_as(self):
        pass

    def auto_select_active_space_by_rethre(self, threshold: float=1.95, auto_set=False) -> tuple[int, int, List[int]]:
        '''
        根据NBO占据数自动选择活性空间，返回活性电子数, 活性轨道数
        Args:
            threshold (float): 活性空间选择的占据数阈值，选择占据数小于等于该阈值的轨道作为活性轨道
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            tuple: (tuple[int, int, List[int]]): 活性电子数, 活性轨道数, 活性轨道索引列表
        '''
        active_orbital = 0
        active_electron = 0
        active_orbital_indices: list[int] = []
        logger.info(f"Selecting active space based on rectified occupancy with threshold {threshold:.2f}...")
        # 筛选占据数大于1且小于等于threshold的轨道
        valid_occupied_indices = np.where((self.rectified_occupancy > 1.0) & (self.rectified_occupancy <= threshold))[0]
        # 填充到 aoi 中
        active_orbital_indices = valid_occupied_indices.tolist()
        nbo_data = self.nbo_parser.nbo_data

        # 只是打印选出的活性轨道的信息，并没有业务逻辑上的作用
        for i in active_orbital_indices:
            atom_name_list = []
            for a in nbo_data[i].atoms:
                atom_name_list.append(f'{a[0]}{a[1]}')
            atom_name = ','.join(atom_name_list)
            logger.info(f"Selected NBO orbital {i+1}, Occupation number: {nbo_data[i].occupancy:.5f}, Rectified occupancy: {self.rectified_occupancy[i]:.5f}, Atom(s): {atom_name}")

        # 计算出活性轨道数量，活性电子数量
        active_electron, active_orbital = self.get_as_from_aoi(active_orbital_indices)
        if auto_set == True:
            if active_orbital == 0 or active_electron == 0:
                raise ValueError("No active orbitals selected based on the given threshold. Please adjust the threshold or check the occupation numbers.")
            else:
                # 自动将该属性应用到类中
                logger.info(f"Automatically setting active space: {active_electron} electrons / {active_orbital} orbitals")
                self.set_active_space(active_electron, active_orbital)
        
        return active_electron, active_orbital, active_orbital_indices

    def auto_select_active_space_iter(self, auto_set=False, max_threshold=1.96) -> tuple[int, int, List[int]]:
        '''
        通过最小大于1的占据数+0.005的方式选择活性轨道，返回活性电子数, 活性轨道数
        Args:
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
            max_threshold (float): 最大阈值
        Returns:
            tuple: (tuple[int, int, List[int]]): 活性电子数, 活性轨道数, 活性轨道索引列表
        '''
        valid_occupations = self.occupation_numbers[self.occ_indices]
        # 默认阈值不能超过1.96，基本可以排除大多数sigma键
        threshold = min(float(valid_occupations.min()) + 0.002, max_threshold)
        nae, nao, aoi = self.auto_select_active_space_thre(threshold=threshold, auto_set=False)
        # 检查选出的活性轨道数量
        logger.info(f"Automatically selected active space with threshold {threshold:.3f}: {nae} electrons / {nao} orbitals.")
        # 如果活性轨道过多，则降低活性空间的选择阈值
        if nao >= 14 or nae >= 14:
            logger.warning(f"Automatically selected active space has  {nae} electrons / {nao} orbitals, trying to reduce the threshold to select fewer active orbitals......")
            for _ in range(1000):
                if nao < 14:
                    break
                threshold -= 0.002
                nae, nao, aoi = self.auto_select_active_space_thre(threshold=threshold, auto_set=False)
                logger.info(f"Trying threshold {threshold:.2f}: {nae} electrons / {nao} orbitals")
        # 最终检查选出的活性空间是否合理，如果仍然过大则给出警告提示用户手动选择
        if nao >= 14 or nae >= 14:
            logger.warning(f"Automatically selected active space has  {nae} electrons / {nao} orbitals, which may be too large for VB calculations. Consider manually selecting the active space.")
        if auto_set:
            logger.info(f"Automatically setting active space: {nae} electrons / {nao} orbitals")
            self.set_active_space(nae, nao)
        return nae, nao, aoi
    
    def auto_select_active_space_default(self, auto_set=False) -> tuple[int, int, List[int]]:
        '''
        默认选择活性空间的方式，目标轨道为：
        1. 如果BD轨道数小于等于5，则选入所有BD轨道，LP轨道仍按阈值判断
        2. BD轨道，成键占据数小于1.96的轨道
        3. BD轨道，成键小于1.99，同时反键大于0.06，（这个类型可以称为BD-BD*）
        4. BD反键占据数大于0.12时，选入其对应的BD轨道
        5. LP轨道，占据数小于1.96的轨道
        6. 如果按照上述规则无法选到任何活性空间（这代表所有轨道都接近双占），则转入auto_select_active_space_iter方法，通过最小大于1的占据数+0.002的方式选择活性轨道
        Args:
            auto_set (bool): 是否自动将选择的活性空间设置，默认False
        Returns:
            tuple: (tuple[int, int, List[int]]): 活性电子数, 活性轨道数, 活性轨道索引列表
        '''
        threshold_bd_bonding = 1.96
        threshold_bd_bonding_min = 1.9
        threshold_bd_bonding_star = 1.99
        threshold_bd_bonding_star_min = 1.9
        threshold_bd_antibonding = 0.06
        threshold_bd_antibonding_min = 0.15
        threshold_strong_antibonding = 0.12
        threshold_strong_antibonding_max = 0.3
        threshold_lp = 1.96
        threshold_lp_min = 1.9
        if self.mol.spin > 0:
            threshold_bd_bonding = 0.98
            threshold_bd_bonding_min = 0.95
            threshold_bd_bonding_star = 0.995
            threshold_bd_bonding_star_min = 0.95
            threshold_bd_antibonding = 0.03
            threshold_bd_antibonding_min = 0.075
            threshold_strong_antibonding = 0.06
            threshold_strong_antibonding_max = 0.15
            threshold_lp = 0.98
            threshold_lp_min = 0.95
        debug = self.input_data.debug
        nolp = self.input_data.vbsettings.nolp

        # 开始挑轨道，计算nae和nao，从self.nbo_parser中读取 NBO 轨道信息

        # 子函数：添加满足条件的轨道到选中列表中，并记录筛选原因
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
            strong_antibonding_threshold: float,
            lp_threshold: float,
        ) -> tuple[int, int, List[int], List[Dict]]:
            selected_orbitals: Dict[int, Dict] = {}
            rule_hits = {
                "BD-small": [],
                "BD": [],
                "BD-BD*": [],
                "BD*-strong": [],
                "LP": [],
            }

            # 如果 BD 轨道数小于等于5，则选入所有 BD 轨道，LP 轨道仍按阈值判断
            bd_orbitals = [orbital for orbital in self.nbo_parser.nbo_data if orbital.orbital_type == "BD"]
            if len(bd_orbitals) <= 5:
                for orbital in bd_orbitals:
                    add_orbital(selected_orbitals, orbital, "BD-small: all BD orbitals because BD count<=5")
                    rule_hits["BD-small"].append(orbital.index)

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

            # 反键占据数较大时，不再要求成键轨道同时满足占据数上限。
            for pair in self.nbo_parser.bond_antibond_pairs:
                if pair.antibond.occupancy > strong_antibonding_threshold:
                    reason = f"BD*>{strong_antibonding_threshold:.3f}"
                    add_orbital(selected_orbitals, pair.bond, reason)
                    rule_hits["BD*-strong"].append(pair.bond.index)

            # 挑选满足 LP/LP* 轨道占据数小于 lp_threshold 的轨道
            if not nolp:
                for orbital in self.nbo_parser.nbo_data:
                    if orbital.orbital_type in ("LP", "LP*") and 0.0 < orbital.occupancy < lp_threshold:
                        add_orbital(selected_orbitals, orbital, f"LP/LP*<{lp_threshold:.3f}")
                        rule_hits["LP"].append(orbital.index)
            else:
                if debug:
                    logger.debug("NOLP option is set, skipping LP orbital selection.")

            selected_items = sorted(selected_orbitals.values(), key=lambda item: item["orbital"].index)

            # NBO输出中的轨道编号从1开始；矩阵切片需要0-based索引，所以这里统一减一
            active_indices = [item["orbital"].index - 1 for item in selected_items]
            if not active_indices:
                return 0, 0, active_indices, selected_items
            nae, nao = self.get_as_from_aoi(active_indices)
            if debug:
                logger.debug(
                    f"thresholds: BD<{bd_bonding_threshold:.3f}, "
                    f"BD-BD*: BD<{bd_bonding_star_threshold:.3f} and BD*>{bd_antibonding_threshold:.3f}, "
                    f"BD*-strong>{strong_antibonding_threshold:.3f}, "
                    f"LP<{lp_threshold:.3f}"
                )
                logger.debug(f"rule hits before dedupe: {rule_hits}")
            return nae, nao, active_indices, selected_items

        # 子函数：检查选出的活性空间是否在合理范围内，这里以14为一个经验阈值，认为超过14就可能过大了
        def under_limit(selected_nae: int, selected_nao: int) -> bool:
            return selected_nae < 14 or selected_nao < 14

        nae, nao, active_indices, selected_items = select_orbitals(
            threshold_bd_bonding,
            threshold_bd_bonding_star,
            threshold_bd_antibonding,
            threshold_strong_antibonding,
            threshold_lp,
        )

        # 打印选出的活性轨道数量，以及对应几号轨道
        logger.info(
            f"Automatically selected active space by default thresholds: "
            f"{nae} electrons / {nao} orbitals."
        )
        # logger.info(
        #     f"Default thresholds: BD<{threshold_bd_bonding:.3f}, "
        #     f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
        #     f"BD*-strong>{threshold_strong_antibonding:.3f}, "
        #     f"LP<{threshold_lp:.3f}"
        # )
        # 如果选择轨道不为空，则打印每个被选中轨道的详细信息，包括轨道编号、类型、占据数、对应原子以及被选中的原因
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
                # logger.info(
                #     f"Selected NBO orbital {orbital.index}: "
                #     f"{orbital.orbital_type}({orbital.orbital_number}) "
                #     f"occ={orbital.occupancy:.5f}{antibond_occ_text}, atom(s): {atom_name}, reason: {reason_text}"
                # )
        else:
            logger.info("No active orbitals selected by default thresholds.")

        # 如果 nae/nao 过多（大于14），则降低活性空间的选择阈值，每次降低0.005，直到 nae/nao 其中之一小于14
        # 降低阈值的逻辑（1-4每一步都尝试重新选取活性空间）：
        # 1. LP轨道，占据数阈值降低0.005（下限为1.9）
        # 2. BD-BD*轨道，成键阈值降低0.001，反键阈值提高0.003（下限为1.9，上限为0.15）
        # 3. 强反键规则的占据数阈值提高0.1（上限为0.3）
        # 4. BD轨道，占据数阈值降低0.005（下限为1.9）
        # 1-4如此循环，直到达到合理的活性空间大小或者所有阈值都达到边界仍然过大则停止
        while not under_limit(nae, nao):
            changed = False

            if threshold_lp > threshold_lp_min:
                threshold_lp = max(threshold_lp_min, round(threshold_lp - 0.005, 10))
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_strong_antibonding,
                    threshold_lp,
                )
                # logger.info(
                #     f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                #     f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                #     f"BD*-strong>{threshold_strong_antibonding:.3f}, "
                #     f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                # )
                if under_limit(nae, nao):
                    break

            if threshold_bd_bonding_star > threshold_bd_bonding_star_min or threshold_bd_antibonding < threshold_bd_antibonding_min:
                if threshold_bd_bonding_star > threshold_bd_bonding_star_min:
                    threshold_bd_bonding_star = max(threshold_bd_bonding_star_min, round(threshold_bd_bonding_star - 0.003, 10))
                if threshold_bd_antibonding < threshold_bd_antibonding_min:
                    threshold_bd_antibonding = min(threshold_bd_antibonding_min, round(threshold_bd_antibonding + 0.001, 10))
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_strong_antibonding,
                    threshold_lp,
                )
                # logger.info(
                #     f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                #     f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                #     f"BD*-strong>{threshold_strong_antibonding:.3f}, "
                #     f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                # )
                if under_limit(nae, nao):
                    break

            if threshold_strong_antibonding < threshold_strong_antibonding_max:
                threshold_strong_antibonding = min(
                    threshold_strong_antibonding_max,
                    round(threshold_strong_antibonding + 0.1, 10),
                )
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_strong_antibonding,
                    threshold_lp,
                )
                # logger.info(
                #     f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                #     f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                #     f"BD*-strong>{threshold_strong_antibonding:.3f}, "
                #     f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                # )
                if under_limit(nae, nao):
                    break

            if threshold_bd_bonding > threshold_bd_bonding_min:
                threshold_bd_bonding = max(threshold_bd_bonding_min, round(threshold_bd_bonding - 0.005, 10))
                changed = True
                nae, nao, active_indices, selected_items = select_orbitals(
                    threshold_bd_bonding,
                    threshold_bd_bonding_star,
                    threshold_bd_antibonding,
                    threshold_strong_antibonding,
                    threshold_lp,
                )
                # logger.info(
                #     f"Trying default thresholds: BD<{threshold_bd_bonding:.3f}, "
                #     f"BD-BD*: BD<{threshold_bd_bonding_star:.3f} and BD*>{threshold_bd_antibonding:.3f}, "
                #     f"BD*-strong>{threshold_strong_antibonding:.3f}, "
                #     f"LP<{threshold_lp:.3f}: {nae} electrons / {nao} orbitals"
                # )
                if under_limit(nae, nao):
                    break

            if not changed:
                break
            
        # 最终检查选出的活性空间是否合理，如果仍然过大则给出警告提示用户手动选择
        if not under_limit(nae, nao):
            logger.warning(f"Automatically selected active space has {nae} electrons / {nao} orbitals, which may be too large for VB calculations. Consider manually selecting the active space.")
        
        # 如果最终选出的活性空间为空，则转入auto_select_active_space_iter方法，通过最小大于1的占据数+0.005的方式选择活性轨道
        if nae == 0 or nao == 0:
            logger.warning(f"No active orbitals selected by default rules. Trying use iterative method based on minimum occupation number to select active orbitals...")
            nae, nao, active_indices = self.auto_select_active_space_iter(auto_set=auto_set, max_threshold=2.00)
            # raise ValueError("No active orbitals selected by default rules. Please manually select the active space or check the NBO occupation numbers.")
            
        logger.info(f"Final default active space: {nae} electrons / {nao} orbitals")
        active_indices_1based = [idx + 1 for idx in active_indices]
        logger.info(f"Final default active orbital indices (1-based): {active_indices_1based}")
        self.set_active_indices(active_indices)
        if auto_set:
            if nae == 0 or nao == 0:
                raise ValueError("No active orbitals selected by default rules. Please manually select the active space or check the NBO occupation numbers.")
            logger.info(f"Automatically setting active space: {nae} electrons / {nao} orbitals")
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
        logger.info(f"Active space set: {self.active_electron} electrons / {self.active_orbital} orbitals")

    def set_basis_set(self, basis_set: str) -> None:
        '''
        设置基组名称，注意如果NBO计算的基组与这里设置的基组不一致，可能会导致生成的XMVB文件与实际计算不匹配
        Args:
            basis_set (str): 基组名称，例如 'cc-pVDZ'
        '''
        self.basis_set = basis_set

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
        rethre = self.input_data.vbsettings.rethre
        nae = self.active_electron
        nao = self.active_orbital
        # aoa参数，根据原子来判断活性轨道
        if active_orbital_atom:
            logger.info(f"AOA active orbital atom list provided: {active_orbital_atom}, selecting active orbitals based on these atoms...")
            active_indices = self.get_active_orbital_indices_from_active_atoms(active_orbital_atom)

        # 保留aoa_bond参数的兼容性，但不推荐使用
        elif aoa_bond:
            logger.info(f"AOA_BOND active orbital bond list provided: {aoa_bond}, selecting active orbitals based on these bonds...")
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
                    logger.warning(f"Active Orbital Indices index {idx} is out of range. Valid range is [1, {occ_norb}] for occupied orbitals.")
                    # raise ValueError(f"Active Orbital Indices index {idx} is out of range. Valid range is [1, {occ_norb}] for occupied orbitals.")
                active_indices.append(idx - 1)

        # 手动指定了活性空间，但没有aoa
        elif nae > 0 and nao > 0:
            active_indices = self.get_active_orbital_indices(nae, nao)

        # 手动设置了挑选阈值
        elif threshold > 1:
            nae, nao, active_indices = self.auto_select_active_space_thre(threshold=threshold)
            # active_indices = self.get_active_orbital_indices(nae, nao)

        # rethre，修正占据数的测试
        elif rethre > 1:
            nae, nao, active_indices = self.auto_select_active_space_by_rethre(threshold=rethre)

        # 没有任何设置，自动挑选
        else:
            nae, nao, active_indices = self.auto_select_active_space_default()

        derived_nae, derived_nao = self.get_as_from_aoi(active_indices)
        if nae != 0 and nao != 0:
            if nae != derived_nae or nao != derived_nao:
                raise ValueError(f"Active space derived from active orbital indices does not match expected values: derived ({derived_nae} electrons / {derived_nao} orbitals) vs expected ({nae} electrons / {nao} orbitals). Please check the selected active orbital indices and the corresponding occupation numbers.")
        else:
            nae, nao = derived_nae, derived_nao
        if nae > 15 or nao > 15:
            logger.warning(f"Selected active space has {nae} electrons / {nao} orbitals, which may be too large for VB calculations. Consider manually selecting the active space or adjusting the selection criteria.")

        if auto_set:
            self.set_active_space(nae, nao)
            self.set_active_indices(active_indices)

        return nae, nao, active_indices

    def get_active_orbital_indices(self, nae: int, nao: int) -> List[int]:
        '''
        自动获取活性轨道的索引（注意是从0开始计数），根据NBO占据数判断活性轨道，选择NBO占据数小的轨道，并且选择活性轨道数的一半的轨道。
        Args:
            nae (int): 活性电子数
            nao (int): 活性轨道数
        Returns:
            List[int]: 活性轨道的索引列表
        '''
        # 根据NBO占据数判断活性轨道，选择最小的half_orb个
        # 一半的轨道数量
        half_orb = int(nae / 2)

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
        logger.info(f"Automatically selected active orbital indices based on occupation numbers: {actorb_indices}, corresponding occupation numbers: {self.occupation_numbers[actorb_indices]} include atom(s): {get_orbital_atom_contribution(self.orbital_matrix[actorb_indices], self.mol)}")
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
            logger.debug(f"requested active atoms: {active_atom}")
            logger.debug(f"orbital atom contribution (occupation-sorted): {orb_text_o}")

        def consume_orbital_candidates(candidates: List[Tuple[int, int, List[int], float, str]], stage_name: str) -> None:
            if debug:
                logger.debug(f"{stage_name}: start scanning {len(candidates)} candidates, remaining atoms={active_atom_copy}")
            for sorted_orb_idx, orb_index, pair, occ, type_of in candidates:
                need_orb = all(atom in active_atom_copy for atom in pair)
                if debug:
                    logger.debug(f"{stage_name}: candidate sorted_idx={sorted_orb_idx}, orb_index={orb_index}, occ={occ:.6f}, pair={pair}, need_orb={need_orb}, remaining_before={active_atom_copy}")
                if not need_orb:
                    continue
                if orb_index not in actorb_indices:
                    actorb_indices.append(orb_index)
                for atom in pair:
                    if atom in active_atom_copy:
                        active_atom_copy.remove(atom)
                if debug:
                    logger.debug(f"{stage_name}: selected orb_index={orb_index}, selected_list={actorb_indices}, remaining_after={active_atom_copy}")
                if not active_atom_copy:
                    if debug:
                        logger.debug(f"{stage_name}: all active atoms are covered, stop scanning.")
                    break

        if bond_first or nolp:
            strategy = "bond-first" if bond_first else "NOLP"
            logger.info(f"Selecting active orbitals based on {strategy} strategy...")
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
                logger.debug(f"{strategy}: two_atom_orb_indices={[(i, idx, pair, type_of) for i, idx, pair, _, type_of in two_atom_orb_indices]}")
                logger.debug(f"{strategy}: one_atom_orb_indices={[(i, idx, pair, type_of) for i, idx, pair, _, type_of in one_atom_orb_indices]}")

            # 先搜索成键轨道（双原子/多原子）
            consume_orbital_candidates(two_atom_orb_indices, "two_atom_first")
            # 如果还有未覆盖活性原子，再补单原子轨道
            if active_atom_copy:
                if debug:
                    logger.debug(f"{strategy}: remaining atoms after two-atom scan: {active_atom_copy}, now scanning one-atom candidates.")
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
            logger.debug(f"final selected orbital indices={actorb_indices}, remaining active atoms={active_atom_copy}")
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

    def get_as_from_aoi(self, active_indices: List[int]) -> Tuple[int, int]:
        '''
        从 aoi 参数获取活性空间的电子数和轨道数
        Args:
            active_indices (List[int]): 活性轨道索引列表（从0开始计数）
        Returns:
            Tuple[int, int]: 活性电子数, 活性轨道数
        '''
        nao = 0
        nae = 0
        nbo_data = self.nbo_parser.nbo_data
        
        # 每个BD轨道占据数 > 1 对应 nae + 2， nao + 2
        # 每个LP轨道占据数 > 1 对应 nae + 2， nao + 1
        # LP轨道对应1个NAO，BD轨道对应2个NAO，对应的就是 connection 的长度

        nae = 0
        nao = 0

        two_electron_types = {"BD", "LP", "CR"}
        zero_electron_types = {"BD*", "LP*", "RY", "RY*"}

        for idx in active_indices:
            orbital = nbo_data[idx]
            if orbital.orbital_type in two_electron_types:
                nae += 2
            elif orbital.orbital_type in zero_electron_types:
                nae += 0
            else:
                logger.debug(
                    f"Unsupported NBO orbital type '{orbital.orbital_type}' "
                    f"for orbital {orbital.index}; cannot determine its active-electron count."
                )
            nao += len(orbital.connection)
            if self.input_data.debug:
                logger.debug(f"AOI orbital index {idx}(spin: {self.mol.spin}): occupancy={nbo_data[idx].occupancy}, connected atoms={nbo_data[idx].connection}, cumulative NAE={nae}, NAO={nao}")

        if self.mol.spin > 0:
            nae -= self.mol.spin

        # 判断活性空间是否合法
        if nae <= 0 or nao <= 0:
            raise ValueError(f"Derived active space has negative electrons or orbitals: {nae} electrons / {nao} orbitals. Please check the selected active orbital indices and the corresponding occupation numbers.")
        if nae % 2 != self.mol.spin % 2:
            raise ValueError(f"Molecule has spin {self.mol.spin}, but active electrons: {nae}. That is impossible.")

        if self.input_data.debug:
            logger.debug(f"Deriving active space from AOI: provided orbital indices (0-based) = {active_indices}")
            logger.debug(f"Derived active space from AOI: {nae} electrons / {nao} orbitals.")
        return nae, nao

    def set_active_indices(self, active_indices: List[int]) -> None:
        '''
        设置活性轨道的索引（注意是从0开始计数）
        Args:
            active_indices (List[int]): 活性轨道的索引列表
        '''
        self.active_indices = active_indices

    ##### 可以放到辅助函数那里去 #####

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
    
    ##### 生成XMVB文本的函数 #####

    def get_orb_section_inactive(self, atom_list: List[List[int]]) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，适合非活性部分
        Args:
            atom_list (List[List[int]]): 轨道原子贡献列表，例如 [[1,2], [2,3], [4]] 表示第一个轨道由1,2号原子贡献，第二个轨道由2,3号原子贡献，第三个轨道由4号原子贡献
        Returns:
            tuple (Tuple[str, str]): $orb 部分文本，前端为头文本，后端为轨道文本
        '''
        # 将orb部分文本格式化为XMVB需要的格式
        # 获得非活性部分文本
        head_text = ' '.join(str(len(i)) for i in atom_list)
        orb_text = ''
        for orb_atom in atom_list:
            orb_text += f'{" ".join(str(i) for i in orb_atom)}\n'
        orb_tuple = (head_text,orb_text)
        return orb_tuple

    def get_orb_section_active(self, atom_list: List[int]) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，活性部分
        Args:
            atom_list (List[int]): 轨道原子贡献列表，例如 [1,2,3,4] 表示轨道由1,2,3,4号原子贡献
        Returns:
            tuple (Tuple[str, str]): $orb 部分文本，前端为头文本，后端为轨道文本
        '''
        # 将orb部分文本格式化为XMVB需要的格式
        # 获得活性部分文本
        head_text = f'1*{self.active_orbital}'
        orb_atom_list = []
        for orb_atom in atom_list:
            orb_atom_list.append(orb_atom)
        
        # 在活性轨道的第一行添加注释，标明活性轨道开始
        orb_atom_list[0] = f"{orb_atom_list[0]}   # active orbital start here"

        orb_text = '\n'.join(str(i) for i in orb_atom_list) + '\n'
        orb_tuple = (head_text,orb_text)
        return orb_tuple

    def get_orb_section_total(self, active_order: List[int]) -> str:
        '''
        获取 XMVB .xmi 文件中的 $orb 部分文本，包含非活性和活性轨道
        Args:
            active_order (List[int]): 活性原子顺序列表，例如 [1,2,3,4] 表示活性轨道由1,2,3,4号原子贡献
        Returns:
            str (str): $orb 部分文本
        '''
        inactive_head, inactive_text = self.get_orb_section_inactive(self.inactive_orb_atom)
        active_head, active_text = self.get_orb_section_active(active_order)
        active_text = active_text.strip("\n")  # 去掉活性部分文本末尾的换行，避免和非活性部分之间多出空行
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
    
    def get_init_guess_active(self, orbital_matrix: np.ndarray, atom_list: List[List[int]], atom_order_list: Optional[List[int]]=None) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $gus 文本，适合活性轨道
        Args:
            orbital_matrix (np.ndarray): 二维轨道矩阵
            atom_list (List[int]): 活性原子列表
            atom_order_list (Optional[List[int]]): 原子顺序列表
        Returns:
            tuple (Tuple[str, str]): $gus 部分文本，前端为头文本，后端为轨道文本
        '''
        # 生成活性原子：活性轨道的映射关系
        # 使用列表而不是字典，允许同一个原子对应多个活性轨道
        # Tuple 的第一个元素是原子编号（1开始），第二个元素是对应的轨道矩阵
        atom_orb_items: List[Tuple[int, np.ndarray]] = []
        for orb, oac_item in zip(orbital_matrix, atom_list):
            # 根据轨道数量看需要复制多少份，同时计算活性轨道数量
            for j in oac_item:
                if self.input_data.vbsettings.atom_slice:
                    new_orb = self.get_atom_sliced_orbital(orb, j)
                    atom_orb_items.append((j, new_orb))
                else:
                    atom_orb_items.append((j, orb))
        
        if len(atom_orb_items) != self.active_orbital:
            raise ValueError(f"Calculated active orbital count ({len(atom_orb_items)}) does not match expected active orbital count ({self.active_orbital}). Check active space settings.")
        
        # 根据输入参数重新排序
        if atom_order_list:
            ordered_atom_orb_items: List[Tuple[int, np.ndarray]] = []
            for atom in atom_order_list:
                for i, item in enumerate(atom_orb_items):
                    if item[0] == atom:
                        ordered_atom_orb_items.append(atom_orb_items.pop(i))
                        break
            atom_orb_items = ordered_atom_orb_items

        head_text = (' ' + str(orbital_matrix.shape[1])) * len(atom_orb_items)
        orb_text = ''
        # 26/3/19 已经完成支持（7，8）这样的活性空间了
        for i, (atom, orb) in enumerate(atom_orb_items):
            orb_text += f'# ACTIVE ORBITAL        {i+1}  NAO =    {len(orb)} Localization in atom {atom}{self.mol.atom_pure_symbol(atom-1)}\n'
            orb_text += make_xmvb_format_text(orb, per_line=4)
            orb_text += '\n'
        return (head_text, orb_text)

    def get_init_guess_total(self, active_order: List[int]) -> Tuple[str, str]:
        '''
        获取 XMVB .xmi 文件中的 $gus 文本，包含非活性和活性轨道
        Args:
            active_order (List[int]): 活性原子顺序列表
        Returns:
            tuple (Tuple[str, str]): $gus 部分文本，前端为头文本，后端为轨道文本
        '''
        inact_head, inact_guess = self.get_init_guess_inactive(self.inactive_orbital_matrix)
        act_head, act_guess = self.get_init_guess_active(self.active_orbital_matrix, self.active_orb_atom, active_order)
        # 拼装初猜文本
        init_guess_text = (
            inact_head + act_head + '\n' +
            inact_guess +
            act_guess.strip("\n")
        )
        return init_guess_text

    def get_active_orb_atom_order(self, atom_list: List[List[int]]) -> List[int]:
        '''
        根据 active_order 设置生成活性 $orb 行使用的原子顺序。
        Args:
            atom_list (List[List[int]]): 活性轨道原子贡献列表。
        Returns:
            List[int]: 原子的顺序；None 表示不排序。
        '''
        orb_atom_list = [atom for orb_atom in atom_list for atom in orb_atom]
        active_order = self.input_data.vbsettings.active_order

        if self.input_data.vbsettings.aoa and active_order == 'aoa':
            return list(self.input_data.vbsettings.aoa)
        if active_order == 'seq':
            return sorted(orb_atom_list)
        if active_order == 'rumer':
            logger.info(f"Active order setting is 'rumer', determining active orbital atom order based on Rumer graph...")
            return self.get_rumer_order(orb_atom_list)
        return orb_atom_list
    
    def get_rumer_order(self, active_atoms: List[int]) -> List[int]:
        '''
        获得Rumer图的原子顺序，输入为活性原子列表，输出为按照Rumer图顺序排列的活性原子列表
        Args:
            active_atoms (List[int]): 活性原子索引列表（从1开始计数）
        Returns:
            List[int]: 按照Rumer图顺序排列的活性原子索引列表（从1开始计数）
        '''
        from ..utils.rumer_active_graph import (
            infer_active_atom_order,
            print_order_process_en,
            write_active_graph_topology_svg,
        )
        log_subroutine("Entry Rumer Active Graph")

        num_atoms = self.mol.natm
        xyz_block = f"{num_atoms}\n\n{self.geometry_text}"
        CHARGE = self.input_data.charge
        HIDE_HYDROGENS = False
        result = infer_active_atom_order(
            xyz_block,
            active_atoms,
            charge=CHARGE,
            hide_hydrogens=HIDE_HYDROGENS,
        )
        if self.input_data.vbsettings.draw_rumer:
            svg_file = write_active_graph_topology_svg(result, Path.cwd(), f'{self.origin_filename}_rumer_graph.svg')
            logger.info(f"Active graph topology SVG: {svg_file.resolve()}")
        logger.info(f"Rumer Active Graph: Final order(atom indices): {result.final_order}")
        if self.input_data.debug:
            logger.debug(f"Rumer Active Graph: Detailed order inference process:")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                print_order_process_en(result)
            order_process = buffer.getvalue().rstrip()
            if order_process:
                logger.debug(order_process)
        return result.final_order

    ##### 最终生成XMVB输入文件所需数据的函数 #####

    def get_xmidata(self) -> 'XMIData':
        vbsetting = self.input_data.vbsettings
        method = self.input_data.method.lower()
        iscf = vbsetting.iscf
        stru_type = vbsetting.stru
        nae, nao = self.active_electron, self.active_orbital

        logger.info(f"Preparing to generate XMVB input data with method={method}, nae={nae}, nao={nao}...")
        if method == 'blw':
            # BLW方法，本质是2e1o的VBSCF
            nae = 2
            nao = 1
            stru_type = 'full'
            method = 'vbscf'

            # 获取orb部分
            inactive_head, inactive_text = self.get_orb_section_inactive(self.occ_orb_atom)
            inactive_text = inactive_text.strip("\n")
            orb_number_text = f'{inactive_head}'
            orb_section = f'{orb_number_text}\n{inactive_text}'

            inact_head, inact_guess = self.get_init_guess_inactive(self.occupation_orbital_matrix)
            # 拼装初猜文本
            init_guess_section = (
                inact_head + '\n' +
                inact_guess.strip("\n")
            )

        else:
            # 检查方法设置，如果是LAM-DFVB或BOVB，强制调整相关参数
            if method == 'lam-dfvb':
                logger.info("LAM-DFVB method detected, currently only BLYP functional is available.")
                method = 'lam-dfvb=blyp'
            if method == 'bovb':
                logger.info("BOVB method detected, only iscf=2 will be used regardless of user input.")
                iscf = 2

            # 获得active_order
            active_order = self.get_active_orb_atom_order(self.active_orb_atom)
            # 获取orb部分
            orb_section = self.get_orb_section_total(active_order)
            # 获取初猜部分
            init_guess_section = self.get_init_guess_total(active_order)

            stru_type = vbsetting.stru
            if stru_type == 'default':
                if self.active_orbital > 8:
                    stru_type = 'cov'
                else:
                    stru_type = 'full'

        # 生成XMIData对象
        from ..io.writers import XMIData
        xmidata = XMIData(
            molecule_name=self.filename,
            method=method,
            stru_type=stru_type,
            int_type=vbsetting.inte,
            iscf=iscf,
            nae=nae,
            nao=nao,
            ncharge=self.input_data.charge,
            nmul=self.input_data.spin,
            basis_set=self.basis_set,
            sort=vbsetting.sort,
            orb_section=orb_section,
            geo_section=self.geometry_text,
            init_guess_section=init_guess_section,
        )
        return xmidata
