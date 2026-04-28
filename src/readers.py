from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, get_args, get_origin, get_type_hints
import re
from typing import TYPE_CHECKING
from pyssian import GaussianInFile
import numpy as np
from collections import Counter

from .constants import SUPPORTED_METHODS
from .main import autoVBInputData, VBSettings, XMIPassthrough
from .utils import print_warning, print_subroutine

if TYPE_CHECKING:
    from pyscf import gto

class autoVBInputParser:
    '''
    解析输入文件，提取必要的信息，如分子结构、基组、计算参数等
    '''
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.text = self.input_path.read_text(errors='ignore')
        # vbscf(2,1)会覆盖VBSettings中的nae/nao值，因此需要单独存储
        self.cmd_nae: int | None = None
        self.cmd_nao: int | None = None
        suffix = self.input_path.suffix.lower()

        # 检查是否使用 .xmi 格式输入
        if ('$ctrl' in self.text.lower() and '$geo' in self.text.lower()) or suffix == ".xmi":
            print('Detected .xmi input file format...')
            is_xmi = True
        else:
            is_xmi = False

        print(f"Content of autoVB input file {self.input_path}:")
        print_subroutine(self.text)

        if is_xmi:
            self.input_data = self.parse_xmi()
        else:
            self.input_data = self.parse_gaussian()
            settings = self.parse_autovb_options(self.input_data.title)

            # 如果 method 中通过 vbscf(nae,nao) 提供了显式值，则覆盖 settings
            if self.cmd_nae is not None or self.cmd_nao is not None:
                # warn if settings already specified nao/nae
                if getattr(settings, 'nae', 0) and getattr(settings, 'nao', 0):
                    print_warning(f"VBSettings in input file contains 'nae' and 'nao' but method provided overrides; using method values {self.cmd_nae},{self.cmd_nao} and ignoring commandline values.")
                if self.cmd_nae is not None:
                    settings.nae = int(self.cmd_nae)
                if self.cmd_nao is not None:
                    settings.nao = int(self.cmd_nao)

            self.input_data.vbsettings = settings

    def parse_xmi(self) -> autoVBInputData:

        lines = self.text.splitlines()
        title = self.input_path.stem
        for ln in lines:
            if ln.strip():
                title = ln.strip()
                break

        # 将 xmi 拆分为各个 section（$ctrl/$geo/$actorb/$str 等）
        sections: Dict[str, List[List[str]]] = {}
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped.startswith("$"):
                i += 1
                continue
            sec_name = stripped[1:].strip().lower()
            if sec_name == "end":
                i += 1
                continue
            i += 1
            body_lines: List[str] = []
            while i < len(lines):
                cur = lines[i].strip().lower()
                if cur == "$end":
                    break
                body_lines.append(lines[i])
                i += 1
            sections.setdefault(sec_name, []).append(body_lines)
            # 跳过 $end
            while i < len(lines) and lines[i].strip().lower() != "$end":
                i += 1
            if i < len(lines) and lines[i].strip().lower() == "$end":
                i += 1

        settings = VBSettings()
        type_hints = get_type_hints(VBSettings)
        alias_map = {"int": "inte", "str": "stru"}
        passthrough = XMIPassthrough()

        # 默认方法
        method: Optional[str] = None
        basis: Optional[str] = None
        debug = False

        ctrl_lines = sections.get("ctrl", [[]])[0] if sections.get("ctrl") else []
        for raw_line in ctrl_lines:
            s = raw_line.strip()
            if not s:
                continue

            if "=" in s:
                key, value = s.split("=", 1)
                key = key.strip()
                value = value.strip()
                key_lower = key.lower()
                mapped_key = alias_map.get(key_lower, key_lower)

                if mapped_key == "basis":
                    basis = value
                    continue
                if mapped_key == "debug":
                    debug = self.parse_value_by_type(value, bool, mapped_key)
                    continue

                if hasattr(settings, mapped_key):
                    target_type = type_hints.get(mapped_key, str)
                    try:
                        parsed_value = self.parse_value_by_type(value, target_type, mapped_key)
                        setattr(settings, mapped_key, parsed_value)
                    except Exception as e:
                        print(f"Warning: failed to parse $ctrl key '{key}' with value '{value}': {e}")
                    continue

                passthrough.ctrl_extra_lines.append(raw_line)
                continue

            # 无等号的开关型字段，例如 sort
            key_lower = s.lower()
            if key_lower in SUPPORTED_METHODS:
                if method is None:
                    method = key_lower
                continue
            mapped_key = alias_map.get(key_lower, key_lower)
            if mapped_key == "debug":
                debug = True
                continue
            if hasattr(settings, mapped_key):
                target_type = type_hints.get(mapped_key, bool)
                try:
                    parsed_value = self.parse_value_by_type(True, target_type, mapped_key)
                    setattr(settings, mapped_key, parsed_value)
                except Exception as e:
                    print(f"Warning: failed to parse flag field '{s}' in $ctrl: {e}")
            else:
                passthrough.ctrl_extra_lines.append(raw_line)

        if method is None:
            raise ValueError(f"Failed to parse method from $ctrl section in {self.input_path}")
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}. Supported methods are: {SUPPORTED_METHODS}")
        if not basis:
            raise ValueError(f"Failed to parse basis from $ctrl section in {self.input_path}")

        # $actorb 等价于 aoa
        actorb_blocks = sections.get("actorb", [])
        if actorb_blocks:
            actorb_text = "\n".join(actorb_blocks[0])
            actorb_numbers = re.findall(r"[+-]?\d+", actorb_text)
            settings.aoa = [int(x) for x in actorb_numbers]

        # $str 原样透传
        str_blocks = sections.get("str", [])
        if str_blocks:
            passthrough.str_section_text = "\n".join(str_blocks[0])

        # 解析几何
        geo_lines = sections.get("geo", [[]])[0] if sections.get("geo") else []
        if not geo_lines:
            raise ValueError(f"Failed to parse $geo section from {self.input_path}")
        geometry = "\n".join(geo_lines).strip()

        settings.validate()

        atvb_input = autoVBInputData(
            title=title,
            filepath=self.input_path,
            filename=self.input_path.stem,
            method=method,
            basis=basis,
            geometry=geometry,
            debug=debug,
            vbsettings=settings,
            xmi_passthrough=passthrough,
        )
        print(f"Parsed XMI input file {self.input_path} successfully with method {method} and basis {basis}")
        return atvb_input

    def parse_gaussian(self) -> autoVBInputData:
        with GaussianInFile(self.input_path) as input_file:
            input_file.read()
        # method和basis不会自动读取，原因是GaussianInFile不支持VB方法的读取，它会识别成一整个参数
        # 识别包含VB或/的行，提取method和basis
        cmd_line = input_file.commandline
        method = None
        basis = None
        for i in cmd_line.items():
            key: str = i[0]
            if "/" in key:
                method_basis = key.split('/')
                method_raw = method_basis[0].strip()
                basis = method_basis[1]

                # 解析形如 vbscf(2,1) 的结构
                m = re.fullmatch(r"([A-Za-z0-9_+-]+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", method_raw)
                if m:
                    method = m.group(1).lower()
                    try:
                        self.cmd_nae = int(m.group(2))
                        self.cmd_nao = int(m.group(3))
                    except Exception:
                        raise ValueError(f"Failed to parse nae/nao from method specification: {method_raw}")
                else:
                    method = method_raw.lower()

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
        if isinstance(raw, str):
            raw = raw.strip()
        origin = get_origin(target_type)
        # print(f"Parsing value for key '{key}': raw='{raw}', target_type={target_type}, origin={origin}")
        if get_origin(target_type) == Union:
            actual_types = get_args(target_type)
            target_type = actual_types[0]

        if isinstance(raw, bool) and target_type is bool:
            return raw
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
        if origin is list or target_type is list:
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
            if key == "aoa_bond":
                if all(isinstance(x, int) for x in parsed):
                    return [parsed[i : i + 2] for i in range(0, len(parsed), 2)]
            return parsed
        if target_type is Path:
            return Path(raw)
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
        pair_list: list[str] = [p.strip() for p in re.findall(pattern, inner) if p.strip()]
        settings = VBSettings()
        type_hints = get_type_hints(VBSettings)
        alias_map = {"int": "inte", "str": "stru"}

        for pair in pair_list:
            if "=" not in pair:
                key = pair.strip()
                value = True
            else:
                key, value = pair.split("=", 1)
                key = key.strip()
                value = value.strip(' ')

            key_lower = key.lower()
            key = alias_map.get(key_lower, key_lower)
            if key == "debug":
                if value is True:
                    self.input_data.debug = True
                continue
            if not hasattr(settings, key):
                continue
            target_type = type_hints.get(key, str)
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


@dataclass
class NBOHybridInfo:
    """
    NBO 杂化信息，如 s(42.15%) p1.37(57.72%) d0.00(0.13%)
    """
    raw_text: str = ""
    s_percent: Optional[float] = None
    p_ratio: Optional[float] = None
    p_percent: Optional[float] = None
    d_ratio: Optional[float] = None
    d_percent: Optional[float] = None
    f_ratio: Optional[float] = None
    f_percent: Optional[float] = None


@dataclass
class NBOContribution:
    """
    单个贡献项，例如 (66.28%) 0.8141* O 1 ...
    """
    contribution_percent: float
    coefficient: float
    atom_symbol: str
    atom_index: int
    hybrid: NBOHybridInfo
    coefficient_vector: List[float] = field(default_factory=list)


@dataclass
class NBOOrbital:
    """
    一个完整 NBO 轨道块的信息。
    """
    index: int
    occupancy: float
    orbital_type: str
    orbital_number: int
    atoms: List[tuple[str, int]]
    connection: List[int]
    raw_header: str
    raw_text: str
    orbital_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    contributions: List[NBOContribution] = field(default_factory=list)


@dataclass
class NBOBondAntibondPair:
    """
    成键 NBO 与对应反键 NBO 的配对信息。
    """
    bond: NBOOrbital
    antibond: NBOOrbital
    key: tuple[int, tuple[int, ...]]

    def __str__(self):
        return f"Bond: {self.bond.index} {self.bond.occupancy} Antibond: {self.antibond.index} {self.antibond.occupancy}\n"
    
    __repr__ = __str__


class GaussianNBOParser:
    '''
    解析 Gaussian 的 NBO 输出文件，提取必要的信息，如 NBO 分析结果等
    '''
    # NBO 轨道段落起始标记
    _SECTION_HEADER_RE = re.compile(r"\(Occupancy\)\s+Bond orbital/\s+Coefficients/\s+Hybrids")
    # 每个轨道块首行，例如：24. (1.99983) CR ( 1) O   1 ...
    _BLOCK_START_RE = re.compile(
        r"^\s*(?P<idx>\d+)\.\s+\(\s*(?P<occ>[-+]?\d+(?:\.\d+)?)\)\s+"
        r"(?P<orb_type>[A-Z0-9]+\*?)\s*\(\s*(?P<orb_no>\d+)\)\s+(?P<label>.+?)\s*$"
    )
    # 多中心贡献行，例如：(66.28%) 0.8141* O 1 ...
    _CONTRIB_RE = re.compile(
        r"^\s*\(\s*(?P<pct>[-+]?\d+(?:\.\d+)?)%\)\s+"
        r"(?P<coef>[-+]?\d+(?:\.\d+)?)\*\s+(?P<atom>[A-Za-z]{1,2})\s+"
        r"(?P<atom_idx>\d+)\s*(?P<hybrid>.*)$"
    )
    # 单原子头行标签部分：X  10  s(...)p...d...
    _SINGLE_ATOM_LABEL_RE = re.compile(
        r"^\s*(?P<atom>[A-Za-z]{1,2})\s+(?P<atom_idx>\d+)\b(?P<rest>.*)$"
    )
    _FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
    _COEFF_LINE_RE = re.compile(
        r"^\s*[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?"
        r"(?:\s+[-+]?(?:\d+\.\d*|\.\d+)(?:[Ee][-+]?\d+)?)+\s*$"
    )
    _END_TOKENS = (
        "NHO Directionality",
        "Second Order Perturbation Theory Analysis",
        "Natural Bond Orbitals (Summary):",
    )

    def __init__(self, nbo_output_path: Path, nbo_orb_path: Path, mol: 'gto.Mole', debug: bool = False):
        self.nbo_output_path = nbo_output_path
        self.nbo_orb_path = nbo_orb_path
        self.mol = mol
        self.debug = debug
        self.basis_function_number = self.mol.nao_nr()
        self.nbo_data = self.parse_out()
        self.origin_nbo_orbital_matrix = self.parse_nbo_orbital()
        self.nbo_orbital_matrix: np.ndarray = self.origin_nbo_orbital_matrix[:-1]
        self.occupation_numbers: np.ndarray = self.origin_nbo_orbital_matrix[-1]
        self.pnbo_orbital_matrix = self.parse_pnbo_orbital()

        # 填充nbo_data中的 orbital_vector
        for orbital in self.nbo_data:
            orbital.orbital_vector = self.nbo_orbital_matrix[orbital.index - 1]
        # 构建每个轨道对应的原子列表
        self.orbital_atoms = [orbital.connection for orbital in self.nbo_data]
        # 获得每种轨道类型对应的dict，键为轨道类型字符串，值为该类型轨道的列表
        self.orbitals_by_type: Dict[str, List[int]] = {}
        for orbital in self.nbo_data:
            self.orbitals_by_type.setdefault(orbital.orbital_type, []).append(orbital.index)
        self.bond_antibond_pairs = self.build_bond_antibond_pairs()
        self.bond_antibond_pair_by_bond_index = {
            pair.bond.index: pair for pair in self.bond_antibond_pairs
        }
        self.bond_antibond_pair_by_antibond_index = {
            pair.antibond.index: pair for pair in self.bond_antibond_pairs
        }
        if self.debug:
            print("[DEBUG][nbo_parser] Finished parsing NBO output. Summary of parsed data:")
            print(self.bond_antibond_pairs)
            orbital_type_counter = Counter(orbital.orbital_type for orbital in self.nbo_data)
            print(f"[DEBUG][nbo_parser] NBO orbital type counts: {dict(orbital_type_counter)}")
            print(f"[DEBUG][nbo_parser] BD/BD* pair count: {len(self.bond_antibond_pairs)}")

    def _parse_hybrid_info(self, text: str) -> NBOHybridInfo:
        """解析 s/p/d/f 杂化文本。"""
        info = NBOHybridInfo(raw_text=text.strip())
        ms = re.search(r"s\(\s*([-+]?\d+(?:\.\d+)?)%\)", text)
        if ms:
            info.s_percent = float(ms.group(1))

        mp = re.search(r"p\s*([-+]?\d+(?:\.\d+)?)\(\s*([-+]?\d+(?:\.\d+)?)%\)", text)
        if mp:
            info.p_ratio = float(mp.group(1))
            info.p_percent = float(mp.group(2))

        md = re.search(r"d\s*([-+]?\d+(?:\.\d+)?)\(\s*([-+]?\d+(?:\.\d+)?)%\)", text)
        if md:
            info.d_ratio = float(md.group(1))
            info.d_percent = float(md.group(2))

        mf = re.search(r"f\s*([-+]?\d+(?:\.\d+)?)\(\s*([-+]?\d+(?:\.\d+)?)%\)", text)
        if mf:
            info.f_ratio = float(mf.group(1))
            info.f_percent = float(mf.group(2))

        return info

    def _parse_single_atom_orbital_label(self, label: str) -> tuple[List[tuple[str, int]], List[int], NBOContribution]:
        """
        解析单原子轨道头行（例如 CR/LR/RY*/LP）。
        这类行中常带有 p 0.00/d 0.00 等杂化信息，不能把其误识别为原子编号。
        """
        m = self._SINGLE_ATOM_LABEL_RE.match(label)
        if not m:
            raise ValueError(f"Invalid single-atom label: {label}")

        atom_symbol = m.group("atom")
        atom_index = int(m.group("atom_idx"))
        hybrid_text = m.group("rest").strip()

        atoms = [(atom_symbol, atom_index)]
        connection = [atom_index]
        contrib = NBOContribution(
            contribution_percent=100.0,
            coefficient=1.0,
            atom_symbol=atom_symbol,
            atom_index=atom_index,
            hybrid=self._parse_hybrid_info(hybrid_text),
        )
        return atoms, connection, contrib

    def _parse_multi_atom_orbital_label(self, label: str) -> tuple[List[tuple[str, int]], List[int]]:
        """解析多原子轨道头行（例如 BD/BD*，通常包含 '-' 连接符）。"""
        atoms = [(x[0], int(x[1])) for x in re.findall(r"([A-Za-z]{1,2})\s+(\d+)", label)]
        connection = [atom_idx for _, atom_idx in atoms]
        return atoms, connection

    def _bond_pair_key(self, orbital: NBOOrbital) -> tuple[int, tuple[int, ...]]:
        """
        BD 与 BD* 的对应关系由 NBO 轨道序号和连接原子确定。
        例如 BD (2) C 2 - C 4 对应 BD*(2) C 2 - C 4。
        """
        return (orbital.orbital_number, tuple(sorted(orbital.connection)))

    def read_bond_orbital_blocks(self) -> List[str]:
        """
        读取 Gaussian NBO 输出文件中 Bond orbital 段落，按“每个轨道块”切分。
        返回值示例为用户期望的字符串列表。
        """
        lines = self.nbo_output_path.read_text(errors='ignore').splitlines()

        section_start = None
        for i, line in enumerate(lines):
            if self._SECTION_HEADER_RE.search(line):
                section_start = i
                break

        if section_start is None:
            raise ValueError(f"Failed to find NBO Bond orbital section in file: {self.nbo_output_path}")

        blocks: List[str] = []
        current_block: List[str] = []

        for line in lines[section_start + 1:]:
            stripped = line.strip()
            if any(stripped.startswith(tok) for tok in self._END_TOKENS):
                break

            if self._BLOCK_START_RE.match(line):
                if current_block:
                    blocks.append("\n".join(current_block).rstrip())
                current_block = [line.rstrip()]
                continue

            if not current_block:
                continue

            current_block.append(line.rstrip())

        if current_block:
            blocks.append("\n".join(current_block).rstrip())

        if not blocks:
            raise ValueError(f"Found NBO Bond orbital section but no orbital block in file: {self.nbo_output_path}")

        return blocks

    def parse_bond_orbital_block(self, block_text: str) -> NBOOrbital:
        """
        将单个轨道块字符串解析为结构化数据类。
        """
        lines = [ln.rstrip("\n") for ln in block_text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("Empty block_text cannot be parsed.")

        header = lines[0]
        m = self._BLOCK_START_RE.match(header)
        if not m:
            raise ValueError(f"Invalid bond orbital header: {header}")

        index = int(m.group("idx"))
        occupancy = float(m.group("occ"))
        orbital_type = m.group("orb_type")
        orbital_number = int(m.group("orb_no"))
        label = m.group("label")

        # 单原子轨道（CR/LR/RY*/LP 等）与多原子轨道（BD/BD*）分开解析，避免把 p 0.00 识别为原子。
        is_single_atom = "-" not in label
        seed_single_contrib: Optional[NBOContribution] = None
        if is_single_atom:
            atoms, connection, seed_single_contrib = self._parse_single_atom_orbital_label(label)
        else:
            atoms, connection = self._parse_multi_atom_orbital_label(label)

        parsed = NBOOrbital(
            index=index,
            occupancy=occupancy,
            orbital_type=orbital_type,
            orbital_number=orbital_number,
            atoms=atoms,
            connection=connection,
            raw_header=header,
            raw_text=block_text,
        )
        if seed_single_contrib is not None:
            parsed.contributions.append(seed_single_contrib)

        current_contrib: Optional[NBOContribution] = None
        for line in lines[1:]:
            contrib_match = self._CONTRIB_RE.match(line)
            if contrib_match:
                hybrid_text = contrib_match.group("hybrid").strip()
                current_contrib = NBOContribution(
                    contribution_percent=float(contrib_match.group("pct")),
                    coefficient=float(contrib_match.group("coef")),
                    atom_symbol=contrib_match.group("atom"),
                    atom_index=int(contrib_match.group("atom_idx")),
                    hybrid=self._parse_hybrid_info(hybrid_text),
                )
                parsed.contributions.append(current_contrib)
                continue

            if current_contrib is None:
                # 单原子轨道时，系数行紧跟在头行后，归入预置的 100% 贡献项。
                if seed_single_contrib is not None and self._COEFF_LINE_RE.match(line):
                    coeffs = [float(x) for x in self._FLOAT_RE.findall(line)]
                    seed_single_contrib.coefficient_vector.extend(coeffs)
                continue

            if self._COEFF_LINE_RE.match(line):
                coeffs = [float(x) for x in self._FLOAT_RE.findall(line)]
                current_contrib.coefficient_vector.extend(coeffs)

        return parsed

    def parse_out(self) -> List[NBOOrbital]:
        """
        解析整个 out 文件，返回结构化 NBO 轨道块数据。
        Returns:
            list (List[NBOOrbital]): 解析得到的 NBO 轨道块列表，每个元素包含轨道类型、占据数、连接原子等信息。
        """
        blocks = self.read_bond_orbital_blocks()
        nbo_data = [self.parse_bond_orbital_block(block) for block in blocks]
        return nbo_data

    def build_bond_antibond_pairs(self) -> List[NBOBondAntibondPair]:
        """
        构建成键 BD 与反键 BD* 的对应关系。
        Returns:
            List[NBOBondAntibondPair]: 每个元素包含一个 BD 轨道及其对应 BD* 轨道。
        """
        bonds: Dict[tuple[int, tuple[int, ...]], NBOOrbital] = {}
        antibonds: Dict[tuple[int, tuple[int, ...]], NBOOrbital] = {}

        for orbital in self.nbo_data:
            if orbital.orbital_type == "BD":
                bonds[self._bond_pair_key(orbital)] = orbital
            elif orbital.orbital_type == "BD*":
                antibonds[self._bond_pair_key(orbital)] = orbital

        pairs: List[NBOBondAntibondPair] = []
        for key, bond in bonds.items():
            antibond = antibonds.get(key)
            if antibond is None:
                continue
            pairs.append(NBOBondAntibondPair(bond=bond, antibond=antibond, key=key))

        pairs.sort(key=lambda pair: pair.bond.index)
        return pairs
    
    def parse_nbo_orbital(self) -> np.ndarray:
        """
        读取并解析 NBO .37 文件中的轨道数据，返回二维系数矩阵。
        Returns:
            array (np.ndarray): 形状为 (basis_functions + 1, basis_functions) 的二维数组，最后一行是占据数
        """
        nbo_path = self.nbo_orb_path
        basis_functions = self.basis_function_number
        headfile_path = nbo_path.with_suffix('.31')
        text = nbo_path.read_text(errors='ignore')
        lines = text.splitlines()
        data = []
        for line in lines[3:]:
            # i==3开始
            if '1  1' in line:
                break
            data_line = line.strip().split()
            data.extend(data_line)
        # 将一维数据转换为二维矩阵 (NAO, NAO)
        # 假设 basis_functions 是 N，则读取的数据应该是 N*(N+1) 个
        # 最后一行是占据数
        arr = np.array(data, dtype=float)
        
        # 简单的完整性检查
        expected = basis_functions * (basis_functions + 1)
        if arr.size != expected:
            print(f"Warning: Read {arr.size} coefficients in NBO file, expected {basis_functions+1}*{basis_functions}={expected}.")
            # 如果读取过多，进行截断；如果过少，reshape会抛出异常
            if arr.size > expected:
                arr = arr[:expected]
        
        return arr.reshape((basis_functions + 1, basis_functions))

    def parse_pnbo_orbital(self) -> np.ndarray:
        """
        读取并解析 NBO .36 文件中的轨道数据（PNBO），返回二维系数矩阵。
        Returns:
            array (np.ndarray): 形状为 (basis_functions , basis_functions) 的二维数组
        """
        nbo_path = self.nbo_orb_path
        basis_functions = self.basis_function_number
        pnbo_path = nbo_path.with_suffix('.36')
        text = pnbo_path.read_text(errors='ignore')
        lines = text.splitlines()
        data = []
        for line in lines[3:]:
            # i==3开始
            if '1  1' in line:
                break
            data_line = line.strip().split()
            data.extend(data_line)
        # 将一维数据转换为二维矩阵 (NAO, NAO)
        # 假设 basis_functions 是 N，则读取的数据应该是 N*N 个
        # 最后一行是占据数
        arr = np.array(data, dtype=float)
        
        # 简单的完整性检查
        expected = basis_functions * basis_functions
        if arr.size != expected:
            print(f"Warning: Read {arr.size} coefficients in PNBO file, expected {basis_functions}*{basis_functions}={expected}.")
            # 如果读取过多，进行截断；如果过少，reshape会抛出异常
            if arr.size > expected:
                arr = arr[:expected]
        
        return arr.reshape((basis_functions, basis_functions))
