from pathlib import Path
import re
import os
import io
import shutil
import subprocess
from datetime import datetime
from collections import Counter
from contextlib import redirect_stdout
import numpy as np
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

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
    get_orbital_atom_contribution,
    pyscf_to_xyz,
)

if TYPE_CHECKING:
    from ..main import autoVBInputData
    from ..io.writers import XMIData
    from pyscf import gto

logger = get_logger(__name__)

# GVB GI orbital可以通过mokit的一个接口生成，但是很难用
# gen_cf_orb(datname='xxxx.dat',ndb=10,nopen=0)
# ndb为双占的离域轨道数量，nopen是开壳层的单占轨道数量
# 详情查看 https://doc.mokit.xyz/chap4-6.html#4634-gen_cf_orb


class XMVBGVB:
    """
    读取 MOKIT/GAMESS 的 GVB 结果。

    Attributes:
        fch_path: MOKIT/GVB 计算后用于读轨道的 fch 文件，通常是 `xxx_s.fch`。
        gms_path: GAMESS 输出文件，用来读取 `PAIR INFORMATION`。
        mo_coeff: MOKIT 读出的 MO 系数矩阵，形状是 AO x MO。
        orbital_matrix: 转置后的轨道矩阵，形状是 MO x AO。
        occupation_numbers: fch 中保存的轨道占据数。
        pair_information: GAMESS `PAIR INFORMATION` 表格，列含义见 `read_gvb_pair_information`。
    """

    def __init__(
        self,
        fch_path: Path | str,
        input_data: 'autoVBInputData',
        gms_path: Optional[Path | str] = None,
        filename: Optional[str] = None,
        orbital_atoms: Optional[List[List[int]]] = None,
    ) -> None:
        """
        初始化 GVB 输出读取器。

        Args:
            fch_path: MOKIT 输出的 fch 文件路径，优先传入排序后的 `xxx_s.fch`。
            gms_path: GAMESS `.gms` 输出文件路径；不传时会根据 `fch_path` 自动猜。
            input_data: autoVB 的输入数据对象，暂时只先保存，后续生成 XMIData 时会用到。
            filename: 输出对象的基准文件名；不传时会从 `fch_path` 推断。
            orbital_atoms: 可选的轨道原子贡献列表，格式是每个轨道对应一个原子索引列表。

        Raises:
            FileNotFoundError: 当 `fch_path` 不存在时抛出。

        Returns:
            None
        """
        # GVB 主要靠 fch 读轨道，靠 .gms 读 pair 表。
        self.fch_path = Path(fch_path)
        if not self.fch_path.is_file():
            raise FileNotFoundError(f"GVB fch file not found: {self.fch_path}")
        self.input_orbital_atoms = [list(atoms) for atoms in orbital_atoms] if orbital_atoms else None

        # MOKIT 常见命名是 `xxx_s.fch` 搭配 `xxx.gms`，所以这里允许不手动传 gms。
        self.original_fch_path = self.fch_path
        self.gms_path = Path(gms_path) if gms_path is not None else self._infer_gms_path()
        self.dat_path = self._infer_dat_path()
        self.input_data = input_data
        self.filename = filename or self._default_filename()
        self.origin_filename = self.filename
        self.mol = self._load_mol()
        self.cf_dat_path: Optional[Path] = None
        self.cf_fch_path: Optional[Path] = None

        self.basis_set = input_data.basis if input_data is not None else ""
        self.geometry_text = pyscf_to_xyz(self.mol)
        symbols = [self.mol.atom_pure_symbol(i) for i in range(self.mol.natm)]
        counts = Counter(symbols)
        self.formula = "".join(f"{sym}{cnt if cnt > 1 else ''}" for sym, cnt in counts.items())
        self.dxx_indices: list[int] = []
        self.fxxx_indices: list[int] = []
        self._df_indices()

        # 这些属性先占位，read() 之后就会变成真实数据。
        self.nbf: Optional[int] = None
        self.nif: Optional[int] = None
        self.mo_coeff: Optional[np.ndarray] = None
        self.raw_orbital_matrix: Optional[np.ndarray] = None
        self.orbital_matrix: Optional[np.ndarray] = None
        self.occupation_numbers: Optional[np.ndarray] = None
        self.gvb_orbital_atoms: Optional[list[list[int]]] = None
        self.orbital_atoms: Optional[list[list[int]]] = None
        self.pair_information: Optional[np.ndarray] = None
        self.gvb_pair_count: int = 0

        # 读完原始轨道之后，把后续 XMVB 会用到的几个分块也准备好。
        self.read()
        logger.debug(self.occupation_numbers)
        logger.debug(f"GVB pair count: {self.gvb_pair_count}")

        # 先用 GVB 轨道自己算一份标签；如果外面传了 orbital_atoms，后面会用一一匹配把它们贴到 GVB 轨道上。
        self.gvb_orbital_atoms = get_orbital_atom_contribution(self.raw_orbital_matrix, self.mol)
        self.orbital_atoms = self.gvb_orbital_atoms
        self.split_occ_vir()
        if self.input_orbital_atoms:
            self.match_orbital_atoms()
        self.split_act_ina()

    ##### 一些mokit默认文件名 #####
    def _default_filename(self) -> str:
        """
        根据 fch 文件名推断默认文件名。

        MOKIT 的 GVB 结果默认是 `xxx_s.fch`，这里会把结尾的 `_s` 去掉，
        这样后续输出文件名会更接近原始 GVB 任务名。

        Returns:
            str: 推断出的默认文件名。
        """
        stem = self.fch_path.stem
        if stem.endswith("_s"):
            stem = stem[:-2]
        return stem

    def _infer_gms_path(self) -> Optional[Path]:
        """
        根据 fch 文件路径自动寻找对应的 GAMESS `.gms` 文件。

        现在支持两种常见命名：
        - `xxx.fch` -> `xxx.gms`
        - `xxx_s.fch` -> `xxx.gms`

        Returns:
            Optional[Path]: 找到时返回 `.gms` 路径；找不到就返回 None。
        """
        candidates = [self.fch_path.with_suffix(".gms")]
        if self.fch_path.stem.endswith("_s"):
            candidates.append(self.fch_path.with_name(f"{self.fch_path.stem[:-2]}.gms"))

        # 按最可能的顺序试一遍，找到第一个存在的就用。
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _infer_dat_path(self) -> Path:
        """
        根据 MOKIT/GAMESS 的默认命名寻找对应的 `.dat` 文件。

        MOKIT 最常见的组合是 `xxx_s.fch` 搭配 `xxx.dat`，所以这里把
        `_s` 后缀去掉再换成 `.dat`。

        Returns:
            Path: 推断出的 `.dat` 文件路径。
        """
        stem = self.original_fch_path.stem
        if stem.endswith("_s"):
            stem = stem[:-2]
        return self.original_fch_path.with_name(f"{stem}.dat")

    def _load_mol(self) -> 'gto.Mole':
        """
        获取用于分析轨道原子贡献的 PySCF mol 对象。

        Returns:
            pyscf.gto.Mole: PySCF 分子对象。
        """
        from mokit.lib.gaussian import load_mol_from_fch
        return load_mol_from_fch(self.fch_path)

    def _df_indices(self) -> None:
        """
        找出 DXX 和 FXXX 基函数的位置，后面按 XMVB 的顺序调整列顺序。

        Returns:
            None
        """
        atom_bf_labels = self.mol.ao_labels(fmt=False)
        for i, bf in enumerate(atom_bf_labels):
            if 'd' in bf[2] and bf[3] == 'xx':
                self.dxx_indices.append(i + 1)
            elif 'f' in bf[2] and bf[3] == 'xxx':
                self.fxxx_indices.append(i + 1)

    def _change_orbital_order(self) -> None:
        """
        调整轨道矩阵列顺序。
        MOKIT 读出的矩阵列是 AO/基函数顺序；这里把 D/F 壳的列顺序换成 autoVB 后续生成 XMVB 初猜时使用的顺序。

        Returns:
            None
        """
        if self.dxx_indices:
            self.orbital_matrix = replace_col_orbital_numbers(self.orbital_matrix, self.dxx_indices)
        if self.fxxx_indices:
            self.orbital_matrix = replace_col_orbital_numbers(self.orbital_matrix, self.fxxx_indices, orbital_type='f')

    ##### 读取函数 #####

    def read(self) -> None:
        """
        一次性读取当前阶段需要的 GVB 数据。

        这只是个薄封装：先读 fch 中的轨道和占据数，再读 `.gms` 中的
        `PAIR INFORMATION`。拆成两个函数是为了后面调试或单独重读更方便。

        Returns:
            None
        """
        self.read_fch_orbitals(self.fch_path)
        self.read_pair_information()
        self.cf_fch_path = self.prepare_cf_orbitals()

    def prepare_cf_orbitals(self) -> Path:
        """
        尝试生成 Coulson-Fischer 轨道并写入新的 fch 文件。

        MOKIT 的 `gen_cf_orb` 只能处理 GAMESS `.dat`，会生成 `xxx_new.dat`；
        如果后面想像普通 fch 一样读取或可视化这些 CF 轨道，还需要再调用
        `dat2fch xxx_new.dat xxx_cf.fch` 把轨道写进 fch。这个函数成功时返回
        新的 `xxx_cf.fch`，失败时保留原始 GVB natural orbital fch。

        Returns:
            Path: 实际用于后续读取轨道的 fch 文件路径。
        """
        from mokit.lib.lo import gen_cf_orb
        npair = self.pair_information.shape[0]
        nopen = int(getattr(self.mol, "spin", 0))
        ndb = int(self.mol.nelec[0] - npair - nopen)
        dat2fch_exe = shutil.which("dat2fch")

        gen_cf_orb(datname=str(self.dat_path), ndb=ndb, nopen=nopen)

        self.cf_dat_path = self.dat_path.with_name(f"{self.dat_path.stem}_new.dat")
        self.cf_fch_path = self.original_fch_path.with_name(f"{self.original_fch_path.stem}_cf.fch")
        shutil.copy2(self.original_fch_path, self.cf_fch_path)
        proc = subprocess.run(
            [dat2fch_exe, str(self.cf_dat_path), str(self.cf_fch_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info(
            "Generated Coulson-Fischer orbital fch %s from %s.",
            self.cf_fch_path,
            self.cf_dat_path,
        )

        return self.cf_fch_path

    def read_fch_orbitals(self, fchname:str) -> tuple[np.ndarray, np.ndarray]:
        """
        从 MOKIT 生成的 fch 文件读取 GVB 轨道和占据数。

        MOKIT 的 `fch2py` 返回的是 AO x MO 的矩阵；XMVB 初猜生成时一行一个轨道，所以这里保存一份转置后的 `orbital_matrix`。

        Args:
            fchname: fch 文件路径。

        Returns:
            tuple[np.ndarray, np.ndarray]: `(mo_coeff, occupation_numbers)`。
                `mo_coeff` 的形状是 AO x MO，`occupation_numbers` 是一维占据数数组。
        """
        from mokit.lib.fch2py import fch2py
        from mokit.lib.rwwfn import read_eigenvalues_from_fch, read_nbf_and_nif_from_fch

        # nbf/nif 是 fch2py 必须要的尺寸信息，先读出来再取 MO 系数。
        self.nbf, self.nif = read_nbf_and_nif_from_fch(fchname)
        self.mo_coeff: np.ndarray = fch2py(fchname, self.nbf, self.nif, "a")
        # raw_orbital_matrix 保留 PySCF/MOKIT 原始 AO 顺序，适合做原子贡献分析。
        self.raw_orbital_matrix: np.ndarray = self.mo_coeff.T.copy()
        # orbital_matrix 会按 XMVB 的 D/F 轨道顺序重排，适合后面写初猜。
        self.orbital_matrix: np.ndarray = self.raw_orbital_matrix.copy()
        self._change_orbital_order()
        self.occupation_numbers: np.ndarray = read_eigenvalues_from_fch(fchname, self.nif, "a")
        logger.info(
            "Read GVB orbitals from %s: nbf=%s, nif=%s",
            self.fch_path,
            self.nbf,
            self.nif,
        )
        return self.mo_coeff, self.occupation_numbers

    def read_pair_information(self) -> Optional[np.ndarray]:
        """
        从 GAMESS `.gms` 输出中读取 `PAIR INFORMATION` 表。

        用函数 `read_gvb_pair_information`，返回的矩阵每一行
        对应一个 GVB pair。当前列顺序是：
        `pair_id, orb1, orb2, ci1, ci2, occ1, occ2, overlap, energy_lowering`。

        Raises:
            FileNotFoundError: 当显式给出的 `.gms` 路径不存在时抛出。

        Returns:
            Optional[np.ndarray]: 读取成功时返回 pair 信息矩阵；没有 `.gms` 或没找到
            pair 表时返回 None。
        """
        if self.gms_path is None:
            logger.warning("No GAMESS .gms file found for %s; PAIR INFORMATION was not read.", self.fch_path)
            self.pair_information = None
            self.gvb_pair_count = 0
            return None
        if not self.gms_path.is_file():
            raise FileNotFoundError(f"GVB GAMESS output file not found: {self.gms_path}")

        self.pair_information = read_gvb_pair_information(self.gms_path)
        if self.pair_information is None:
            self.gvb_pair_count = 0
            logger.warning("No PAIR INFORMATION block found in %s.", self.gms_path)
        else:
            self.gvb_pair_count = int(self.pair_information.shape[0])
            logger.info("Read %s GVB pairs from %s.", self.gvb_pair_count, self.gms_path)
        return self.pair_information

    ##### 处理轨道，构造分块（分为occ，vir，act，ina） #####

    def split_occ_vir(self) -> tuple[np.ndarray, np.ndarray]:
        """
        按电子数把 GVB 轨道拆成占据轨道和虚轨道。

        占据轨道数 `nocc = ceil(nelectron / 2)`。
        拆完后会保存 occ/vir 的矩阵、索引、占据数和原子贡献。

        Returns:
            tuple[np.ndarray, np.ndarray]: `(occupation_orbital_matrix, virtual_orbital_matrix)`。
        """

        # 按总电子数向上取整，得到占据轨道数量。
        self.nocc = int((self.mol.nelectron + 1) // 2)
        self.occupation_orbital_matrix = self.orbital_matrix[:self.nocc]
        self.virtual_orbital_matrix = self.orbital_matrix[self.nocc:]
        self.occ_occupation_numbers = self.occupation_numbers[:self.nocc]
        self.virtual_occupation_numbers = self.occupation_numbers[self.nocc:]
        self.occ_orb_atom = self.orbital_atoms[:self.nocc]
        self.vir_orb_atom = self.orbital_atoms[self.nocc:]
        self.occ_indices = list(range(self.nocc))
        self.vir_indices = list(range(self.nocc, self.orbital_matrix.shape[0]))

        logger.info(
            "Split GVB orbitals: nocc=%s, occupied=%s, virtual=%s",
            self.nocc,
            self.occupation_orbital_matrix.shape[0],
            self.virtual_orbital_matrix.shape[0],
        )
        return self.occupation_orbital_matrix, self.virtual_orbital_matrix

    def get_atom_contribution_matrix(self, orbital_matrix: np.ndarray) -> np.ndarray:
        """
        计算每条轨道在每个原子上的贡献。

        这里用的是很直观的系数平方和：某个原子上的 AO 系数平方和 / 整条轨道的 AO 系数平方和。
        这个矩阵专门用来给 GVB 轨道和传入的 orbital_atoms 标签打分。

        Args:
            orbital_matrix: 轨道矩阵，形状是 MO x AO；AO 顺序需要和 PySCF mol 保持一致。

        Returns:
            np.ndarray: 贡献矩阵，形状是 轨道数 x 原子数。
        """
        atom_slices = self.mol.aoslice_by_atom()
        coeff_square = np.abs(orbital_matrix) ** 2
        total_square = coeff_square.sum(axis=1)
        total_square[total_square == 0] = 1.0

        contribution_matrix = np.zeros((orbital_matrix.shape[0], self.mol.natm))
        for atom_index, (_, _, start_bf, end_bf) in enumerate(atom_slices):
            contribution_matrix[:, atom_index] = coeff_square[:, start_bf:end_bf].sum(axis=1)
        contribution_matrix /= total_square[:, None]
        return contribution_matrix

    def match_orbital_atoms(self) -> list[list[int]]:
        """
        将传入的 orbital_atoms 原子标签一一匹配到 GVB 占据轨道上。

        匹配分数来自 GVB 轨道在标签原子上的贡献总和。例如标签是 [1, 2]，
        就看某条 GVB 轨道落在 1 号和 2 号原子上的贡献加起来有多大。
        最后用 Hungarian 算法做全局一一匹配，保证每个标签都刚好用一次。

        Raises:
            ValueError: 当传入标签数量和 GVB 占据轨道数量不一致时抛出。

        Returns:
            list[list[int]]: 按 GVB 占据轨道顺序排列后的原子标签。
        """
        input_orbital_atoms = self.input_orbital_atoms
        if input_orbital_atoms is None:
            return self.occ_orb_atom

        if len(input_orbital_atoms) != self.nocc:
            raise ValueError(
                "orbital_atoms must have the same length as occupied GVB orbitals: "
                f"{len(input_orbital_atoms)} != {self.nocc}"
            )

        from scipy.optimize import linear_sum_assignment

        # 用 raw_orbital_matrix 是因为它还没有做 XMVB 的 D/F 列顺序变换，和 PySCF 的原子 AO 切片正好对得上。
        atom_contribution_matrix = self.get_atom_contribution_matrix(self.raw_orbital_matrix[:self.nocc])
        score_matrix = np.zeros((self.nocc, self.nocc))
        for label_index, atoms in enumerate(input_orbital_atoms):
            atom_indices = [atom - 1 for atom in atoms]
            score_matrix[:, label_index] = atom_contribution_matrix[:, atom_indices].sum(axis=1)

        gvb_indices, label_indices = linear_sum_assignment(-score_matrix)

        matched_orbital_atoms: list[list[int]] = [[] for _ in range(self.nocc)]
        self.gvb_to_orbital_atom_indices: list[int] = [0] * self.nocc
        self.orbital_atom_to_gvb_indices: list[int] = [0] * self.nocc
        self.gvb_to_orbital_atom_scores = np.zeros(self.nocc)
        self.orbital_atom_match_score_matrix = score_matrix

        for gvb_index, label_index in zip(gvb_indices, label_indices):
            matched_orbital_atoms[int(gvb_index)] = list(input_orbital_atoms[int(label_index)])
            self.gvb_to_orbital_atom_indices[int(gvb_index)] = int(label_index)
            self.orbital_atom_to_gvb_indices[int(label_index)] = int(gvb_index)
            self.gvb_to_orbital_atom_scores[int(gvb_index)] = score_matrix[int(gvb_index), int(label_index)]

        self.occ_orb_atom = matched_orbital_atoms
        self.orbital_atoms = self.occ_orb_atom + self.gvb_orbital_atoms[self.nocc:]
        self.vir_orb_atom = self.orbital_atoms[self.nocc:]

        logger.info(
            "Matched %s orbital atom labels to occupied GVB orbitals one-to-one.",
            self.nocc,
        )
        return self.occ_orb_atom

    def split_act_ina(self, threshold: float = 1.98) -> tuple[np.ndarray, np.ndarray]:
        """
        只在占据轨道里按 GVB 占据数拆分活性轨道和非活性轨道。

        规则很直接：占据轨道中 `occupation < threshold` 的轨道为活性轨道，
        其余占据轨道为非活性轨道。默认阈值是 1.98。需要先拆分出占据轨道后才能做这个拆分。

        Args:
            threshold: 判断活性轨道的 GVB 占据数阈值。

        Returns:
            tuple[np.ndarray, np.ndarray]: `(active_orbital_matrix, inactive_orbital_matrix)`。
        """
        occ_numbers = self.occupation_numbers[:self.nocc]
        self.active_indices = np.where(occ_numbers < threshold)[0].tolist()
        self.inactive_indices = np.where(occ_numbers >= threshold)[0].tolist()

        self.active_orbital_matrix = self.occupation_orbital_matrix[self.active_indices]
        self.inactive_orbital_matrix = self.occupation_orbital_matrix[self.inactive_indices]
        self.active_occupation_numbers = occ_numbers[self.active_indices]
        self.inactive_occupation_numbers = occ_numbers[self.inactive_indices]
        self.active_orb_atom = [self.occ_orb_atom[i] for i in self.active_indices]
        self.inactive_orb_atom = [self.occ_orb_atom[i] for i in self.inactive_indices]
        self.active_gvb_orbital = len(self.active_indices)
        self.active_orbital = sum(len(atoms) for atoms in self.active_orb_atom)
        self.active_electron = int(round(float(np.sum(self.active_occupation_numbers))))

        logger.info(
            "Split occupied GVB orbitals: active_gvb=%s, active_xmvb=%s, inactive=%s, threshold=%.4f",
            self.active_gvb_orbital,
            self.active_orbital,
            self.inactive_orbital_matrix.shape[0],
            threshold,
        )
        return self.active_orbital_matrix, self.inactive_orbital_matrix

    ##### 生成XMVB文本的函数 #####

    def set_basis_set(self, basis_set: str) -> None:
        """
        设置 XMVB 输入文件中的基组名称。

        Args:
            basis_set: 基组名称，例如 `cc-pVDZ`。

        Returns:
            None
        """
        self.basis_set = basis_set

    def get_atom_sliced_orbital(self, orbital: np.ndarray, atom_index: int) -> np.ndarray:
        """
        将轨道向量中除指定原子外的其他基函数系数清零。

        Args:
            orbital: 一维轨道向量。
            atom_index: 原子序号，从 1 开始。

        Returns:
            np.ndarray: 只保留指定原子 AO 系数的新轨道。
        """
        slices = self.mol.aoslice_by_atom()
        new_orb = np.zeros_like(orbital)
        a0, a1 = slices[atom_index - 1][2], slices[atom_index - 1][3]
        new_orb[a0:a1] = orbital[a0:a1]
        return new_orb

    def get_orb_section_inactive(self, atom_list: List[List[int]]) -> Tuple[str, str]:
        """
        获取 XMVB `.xmi` 文件中非活性轨道的 `$orb` 文本。

        Args:
            atom_list: 非活性轨道原子贡献列表。

        Returns:
            Tuple[str, str]: `(头文本, 轨道原子文本)`。
        """
        head_text = ' '.join(str(len(i)) for i in atom_list)
        orb_text = ''
        for orb_atom in atom_list:
            orb_text += f'{" ".join(str(i) for i in orb_atom)}\n'
        return head_text, orb_text

    def get_orb_section_active(self, atom_list: List[int]) -> Tuple[str, str]:
        """
        获取 XMVB `.xmi` 文件中活性轨道的 `$orb` 文本。

        Args:
            atom_list: 活性轨道展开后的原子顺序。

        Returns:
            Tuple[str, str]: `(头文本, 活性轨道原子文本)`。
        """
        head_text = f'1*{self.active_orbital}'
        orb_atom_list = list(atom_list)
        if orb_atom_list:
            orb_atom_list[0] = f"{orb_atom_list[0]}   # active orbital start here"
        orb_text = '\n'.join(str(i) for i in orb_atom_list) + '\n'
        return head_text, orb_text

    def get_orb_section_total(self, active_order: List[int]) -> str:
        """
        获取完整 `$orb` 文本，顺序是非活性轨道在前、活性轨道在后。

        Args:
            active_order: 活性原子顺序。

        Returns:
            str: `$orb` 区块正文。
        """
        inactive_head, inactive_text = self.get_orb_section_inactive(self.inactive_orb_atom)
        active_head, active_text = self.get_orb_section_active(active_order)
        active_text = active_text.strip("\n")
        orb_number_text = f'{inactive_head} {active_head}'
        return f'{orb_number_text}\n{inactive_text}{active_text}'

    def get_init_guess_inactive(self, orbital_matrix: np.ndarray) -> Tuple[str, str]:
        """
        获取 XMVB `.xmi` 文件中非活性轨道的 `$gus` 初猜文本。

        Args:
            orbital_matrix: 二维轨道矩阵，每行一条轨道。

        Returns:
            Tuple[str, str]: `(头文本, 轨道系数文本)`。
        """
        head_text = (' ' + str(orbital_matrix.shape[1])) * orbital_matrix.shape[0]
        orb_text = ''
        for i, orb in enumerate(orbital_matrix):
            orb_text += f'# ORBITAL        {i+1}  NAO =    {len(orb)}\n'
            orb_text += make_xmvb_format_text(orb, per_line=4)
            orb_text += '\n'
        return head_text, orb_text

    def get_init_guess_active(
        self,
        orbital_matrix: np.ndarray,
        atom_list: List[List[int]],
        atom_order_list: Optional[List[int]] = None,
    ) -> Tuple[str, str]:
        """
        获取 XMVB `.xmi` 文件中活性轨道的 `$gus` 初猜文本。

        一条 GVB 活性轨道通常对应一个 NBO 标签，例如 `[1, 2]`。这里会按标签
        展开成两个原子局域初猜，所以 XMVB 里的活性轨道数是这些标签长度之和。

        Args:
            orbital_matrix: 二维轨道矩阵，每行一条 GVB 活性轨道。
            atom_list: 每条活性 GVB 轨道对应的原子标签。
            atom_order_list: 可选的活性原子输出顺序。

        Raises:
            ValueError: 当展开出的活性轨道数量和 `self.active_orbital` 不一致时抛出。

        Returns:
            Tuple[str, str]: `(头文本, 活性轨道系数文本)`。
        """
        atom_orb_items: List[Tuple[int, np.ndarray]] = []
        for orb, oac_item in zip(orbital_matrix, atom_list):
            for atom in oac_item:
                if self.input_data is not None and self.input_data.vbsettings.atom_slice:
                    atom_orb_items.append((atom, self.get_atom_sliced_orbital(orb, atom)))
                else:
                    atom_orb_items.append((atom, orb))

        if len(atom_orb_items) != self.active_orbital:
            raise ValueError(
                f"Calculated active orbital count ({len(atom_orb_items)}) does not match "
                f"expected active orbital count ({self.active_orbital}). Check GVB orbital labels."
            )

        if atom_order_list:
            ordered_atom_orb_items: List[Tuple[int, np.ndarray]] = []
            for atom in atom_order_list:
                for i, item in enumerate(atom_orb_items):
                    if item[0] == atom:
                        ordered_atom_orb_items.append(atom_orb_items.pop(i))
                        break
            atom_orb_items = ordered_atom_orb_items + atom_orb_items

        head_text = (' ' + str(orbital_matrix.shape[1])) * len(atom_orb_items)
        orb_text = ''
        for i, (atom, orb) in enumerate(atom_orb_items):
            orb_text += f'# ACTIVE ORBITAL        {i+1}  NAO =    {len(orb)} Localization in atom {atom}{self.mol.atom_pure_symbol(atom-1)}\n'
            orb_text += make_xmvb_format_text(orb, per_line=4)
            orb_text += '\n'
        return head_text, orb_text

    def get_init_guess_total(self, active_order: List[int]) -> str:
        """
        获取完整 `$gus` 初猜文本，顺序是非活性轨道在前、活性轨道在后。

        Args:
            active_order: 活性原子顺序。

        Returns:
            str: `$gus` 区块正文。
        """
        inact_head, inact_guess = self.get_init_guess_inactive(self.inactive_orbital_matrix)
        act_head, act_guess = self.get_init_guess_active(
            self.active_orbital_matrix,
            self.active_orb_atom,
            active_order,
        )
        return inact_head + act_head + '\n' + inact_guess + act_guess.strip("\n")

    def get_active_orb_atom_order(self, atom_list: List[List[int]]) -> List[int]:
        """
        根据 `active_order` 设置生成活性 `$orb` 行使用的原子顺序。

        Args:
            atom_list: 活性轨道原子贡献列表。

        Returns:
            List[int]: 活性原子顺序。
        """
        orb_atom_list = [atom for orb_atom in atom_list for atom in orb_atom]
        active_order = self.input_data.vbsettings.active_order if self.input_data is not None else "seq"

        if self.input_data is not None and self.input_data.vbsettings.aoa and active_order == 'aoa':
            return self._complete_active_order(list(self.input_data.vbsettings.aoa), orb_atom_list)
        if active_order == 'seq':
            return sorted(orb_atom_list)
        if active_order == 'rumer':
            logger.info("Active order setting is 'rumer', determining active orbital atom order based on Rumer graph...")
            return self._complete_active_order(self.get_rumer_order(orb_atom_list), orb_atom_list)
        return orb_atom_list

    def _complete_active_order(self, preferred_order: List[int], atom_list: List[int]) -> List[int]:
        """
        用推荐顺序重排活性原子，同时保留重复原子和没有被推荐顺序覆盖的原子。

        Args:
            preferred_order: 推荐的活性原子顺序。
            atom_list: 实际需要输出的活性原子列表，允许重复。

        Returns:
            List[int]: 长度和 `atom_list` 一致的活性原子顺序。
        """
        remaining = list(atom_list)
        completed_order: List[int] = []
        for atom in preferred_order:
            if atom in remaining:
                completed_order.append(atom)
                remaining.remove(atom)
        completed_order.extend(remaining)
        return completed_order

    def get_rumer_order(self, active_atoms: List[int]) -> List[int]:
        """
        按 Rumer 图推断活性原子顺序。

        Args:
            active_atoms: 活性原子索引列表，从 1 开始。

        Returns:
            List[int]: Rumer 图推断后的原子顺序。
        """
        from ..utils.rumer_active_graph import (
            infer_active_atom_order,
            print_order_process_en,
            write_active_graph_topology_svg,
        )

        num_atoms = self.mol.natm
        xyz_block = f"{num_atoms}\n\n{self.geometry_text}"
        charge = self.input_data.charge if self.input_data is not None else self.mol.charge
        result = infer_active_atom_order(
            xyz_block,
            active_atoms,
            charge=charge,
            hide_hydrogens=False,
        )
        if self.input_data is not None and self.input_data.vbsettings.draw_rumer:
            svg_file = write_active_graph_topology_svg(result, Path.cwd(), f'{self.origin_filename}_rumer_graph.svg')
            logger.info(f"Active graph topology SVG: {svg_file.resolve()}")
        logger.info(f"Rumer Active Graph: Final order(atom indices): {result.final_order}")
        if self.input_data is not None and self.input_data.debug:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                print_order_process_en(result)
            order_process = buffer.getvalue().rstrip()
            if order_process:
                logger.debug(order_process)
        return result.final_order

    def get_xmidata(self) -> 'XMIData':
        """
        将已读取的 GVB 数据转换成 XMIData。

        Raises:
            ValueError: 当初始化时没有传入 autoVB input_data 时抛出。

        Returns:
            XMIData: 可直接交给 `write_xmi_file` 的 XMVB 输入数据对象。
        """
        vbsetting = self.input_data.vbsettings
        method = self.input_data.method.lower()
        iscf = vbsetting.iscf
        stru_type = vbsetting.stru
        nae, nao = self.active_electron, self.active_orbital

        logger.info(f"Preparing to generate XMVB input data from GVB with method={method}, nae={nae}, nao={nao}...")
        if method == 'blw':
            nae = 2
            nao = 1
            stru_type = 'full'
            method = 'vbscf'

            inactive_head, inactive_text = self.get_orb_section_inactive(self.occ_orb_atom)
            inactive_text = inactive_text.strip("\n")
            orb_section = f'{inactive_head}\n{inactive_text}'

            inact_head, inact_guess = self.get_init_guess_inactive(self.occupation_orbital_matrix)
            init_guess_section = inact_head + '\n' + inact_guess.strip("\n")
        else:
            if method == 'lam-dfvb':
                logger.info("LAM-DFVB method detected, currently only BLYP functional is available.")
                method = 'lam-dfvb=blyp'
            if method == 'bovb':
                logger.info("BOVB method detected, only iscf=2 will be used regardless of user input.")
                iscf = 2

            active_order = self.get_active_orb_atom_order(self.active_orb_atom)
            orb_section = self.get_orb_section_total(active_order)
            init_guess_section = self.get_init_guess_total(active_order)

            if stru_type == 'default':
                if self.active_orbital > 8:
                    stru_type = 'cov'
                else:
                    stru_type = 'full'

        from ..io.writers import XMIData
        return XMIData(
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
