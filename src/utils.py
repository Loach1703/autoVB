import numpy as np
import re
import os
import shutil
import subprocess
import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from .constants import D_ORBITAL_3TO4, D_ORBITAL_4TO3, F_ORBITAL_3TO4, F_ORBITAL_4TO3
if TYPE_CHECKING:
    from pyscf import gto
    from .main import XMIData

def build_shift_map(offset: int, base: dict[int, int]) -> dict[int, int]:
    """
    将 base 的键值整体平移 offset。
    例：offset=10, base={1:1,2:4,3:6,4:2,5:3,6:5}
    -> {11:11,12:14,13:16,14:12,15:13,16:15}
    Args:
        offset (int): 偏移量
        base (dict[int, int]): 基础映射字典
    Returns:
        dict (dict[int, int]): 平移后的映射字典
    """
    return {offset + k: offset + v for k, v in base.items()}

def make_xmvb_format_text(arr: np.ndarray, per_line: int = 4, ignore_zero: bool = False) -> str:
    """
    将一维浮点数组格式化为固定小数位输出（XMVB .orb文件格式）
    每个数字后跟其序号（从1开始），每行输出per_line个。
    Args:
        arr: np.ndarray 一维浮点数组
        per_line: int 每行输出的项数，默认4
        ignore_zero: bool 是否忽略值为0的元素
    Returns:
        str: 格式化后的字符串
    """
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        line = ' '.join(f"{v:15.10f} {i+j+1:3d}" for j, v in enumerate(chunk))
        lines.append(line)
    return '\n'.join(lines)

def replace_xmvb_orbital_numbers(replace_text: str, orbital_index: list[int], replace_type: int = 3, orbital_type: str = 'd') -> str:
    """
    根据替换规则字典，将XMVB轨道文本中的编号进行替换。
    Args:
        replace_text (str): 输入的轨道文本内容
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

def replace_col_orbital_numbers(replace_array: np.ndarray, orbital_index: list[int], replace_type: int = 3, orbital_type: str = 'd') -> np.ndarray:
    """
    根据替换规则字典，交换读取后的GAMESS轨道文件，以得到符合XMVB格式的基函数排序，交换列
    Args:
        replace_array: str 输入的轨道数组
        orbital_index: list[int] 首个轨道的编号，例如需要替换D轨道，DXX基函数位于第11号与22号，则传入[11, 22]
        replace_type: int 替换类型，3表示XMVB3.0->XMVB4.0，4表示4.0->3.0，默认3
        orbital_type: str 轨道类型，默认'd'
    Returns:
        np.ndarray: 交换后的轨道数组
    """
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

    for bf_index in orbital_index:
        replace_dict.update(build_shift_map(bf_index-1, need_dict))
    # 重排轨道列
    # replace_array的列数量
    n_cols = replace_array.shape[1]
    index = np.arange(n_cols)

    # 替换字典指定的列
    for old_col, new_col in replace_dict.items():
        index[new_col] = old_col
    # 按 index 重排列
    replace_array_new = replace_array[:, index]

    return replace_array_new

def replace_row_orbital_numbers(replace_array: np.ndarray, orbital_index: list[int], replace_type: int = 3, orbital_type: str = 'd') -> np.ndarray:
    """
    根据替换规则字典，交换读取后的GAMESS轨道文件，以得到符合XMVB格式的基函数排序，交换行
    Args:
        replace_array: str 输入的轨道数组
        orbital_index: list[int] 首个轨道的编号，例如需要替换D轨道，DXX基函数位于第11号与22号，则传入[11, 22]
        replace_type: int 替换类型，3表示XMVB3.0->XMVB4.0，4表示4.0->3.0，默认3
        orbital_type: str 轨道类型，默认'd'
    Returns:
        np.ndarray: 交换后的轨道数组
    """
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

    for bf_index in orbital_index:
        replace_dict.update(build_shift_map(bf_index-1, need_dict))

    # replace_array的行数量 
    n_rows = replace_array.shape[0]
    index = np.arange(n_rows)

    # 按映射替换
    for old_row, new_row in replace_dict.items():
        index[new_row] = old_row

    # 重排
    replace_array_new = replace_array[index]

    return replace_array_new

def read_gamess_dat(path: Path) -> str:
    """
    读取 .dat 文件中 $VEC 到 $END 之间的内容，返回为原始多行字符串。
    仅截取首个 $VEC 段。
    Args:
        path (Path): 输入文件路径
    Returns:
        str: $VEC 段内容的多行字符串
    """
    p = Path(path)
    text = p.read_text(errors='ignore')
    lines = text.splitlines()

    in_vec = False
    buf: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not in_vec:
            if s.upper().startswith("$VEC"):
                in_vec = True
            continue
        # in_vec == True
        if s.upper().startswith("$END"):
            break
        if s:  # 跳过空行
            buf.append(ln.rstrip())
    return "\n".join(buf)

def read_gvb_pair_information(file_path: Path):
    """
    从文件中提取 PAIR INFORMATION 部分并转换为 np.ndarray。
    
    参数:
    file_path (str): 文件的路径。
    
    返回:
    np.ndarray: 包含提取数据的二维数组 (float类型)。
                如果没有找到数据，返回 None。
    """
    data_list = []
    start_reading = False
    
    # 定义正则表达式来匹配数据行
    # 这一行通常以数字开头 (PAIR ID)，后面跟着一系列数字
    # 例如: "  1  15  16    0.999926 -0.012197 ..."
    # 简单的逻辑是：如果一行以数字开头，并且我们在 "PAIR INFORMATION" 块之后，就是数据
    
    try:
        text = file_path.read_text(errors='ignore')
        lines = text.splitlines()
            
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # 1. 检测开始标志
            if "PAIR INFORMATION" in line:
                start_reading = True
                continue # 跳过这一行
            
            if not start_reading:
                continue

            # 2. 跳过表头部分
            # 表头通常包含 "ORBITAL", "CI COEFFICIENTS", "PAIR", "ORB 1" 等字样
            # 或者全是破折号 "-------"
            if "ORBITAL" in line or "COEFFICIENTS" in line or "---" in line or "PAIR" in line:
                continue
            
            # 3. 检测结束标志 (可选)
            # 如果遇到空行或者非数字开头且非表头的行，可能意味着块结束了
            # 这里我们假设数据行必须以数字开头
            if not stripped_line:
                if len(data_list) > 0: # 如果已经读到了数据又遇到空行，说明结束了
                    break 
                continue

            # 4. 解析数据行
            # 尝试将行分割并转换为浮点数
            # 使用 split() 默认按任意空白字符分割
            parts = stripped_line.split()
            
            # 检查第一个元素是否是数字，以确保是数据行
            if parts[0].isdigit():
                try:
                    # 将所有部分转换为 float
                    row_data = [float(x) for x in parts]
                    data_list.append(row_data)
                except ValueError:
                    # 如果转换失败，说明可能不是数据行，停止或跳过
                    continue
            else:
                # 如果在开始读取后遇到非数字开头的行（且不是被忽略的表头），通常意味着部分结束
                if len(data_list) > 0:
                    break
                    
        if not data_list:
            return None

        return np.array(data_list)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def split_dat_line(line: str) -> list[float]:
    """
    解析 $VEC 段的一行，提取后续所有浮点数（忽略前两个整数 orb_id 与 line_id）
    例：'1  2-2.1863E-03-1.8207E-04-2.6384E-04 1.5399E-06-9.5735E-04'
    返回: [float, float, ...]
    Args:
        line(str): 输入行字符串
    Returns:
        list[float]: 提取的浮点数列表
    """
    # 捕获第二个整数之后的整个余串（允许第二个整数与负号直接相连）
    m = re.match(r'^\s*\d+\s*\d+(.*)$', line)
    if not m:
        return []
    rest = m.group(1)
    # 全局匹配浮点数，支持可选符号与科学计数法，不依赖空格
    nums = re.findall(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?', rest)
    try:
        return [float(x) for x in nums]
    except ValueError:
        return []

def read_dat_orbital(vec_text: str) -> np.ndarray:
    """
    解析 $VEC 段文本为二维系数矩阵:
    - 每一行形如: <orb_id> <line_id> <c1> <c2> <c3> <c4> <c5>
    - 将相同 <orb_id> 的各行，按 <line_id> 升序拼接其5个系数，得到该轨道的完整系数向量
    - 处理编号回卷规则：
        · 第二次出现的 '1' 代表 101，第三次出现的 '1' 代表 201，依此类推
        · 第一次出现的 '0' 代表 100，第二次出现的 '0' 代表 200，依此类推
    - 返回二维数组 shape = (n_orb, n_coeff)
    Args:
        vec_text(str): $VEC 段的多行字符串
    Returns:
        np.ndarray: 形状为 (轨道数, 基函数数) 的二维数组
    """
    buckets: dict[int, list[tuple[int, list[float]]]] = {}

    # 记录每个原始 orb_id 的“回卷次数”与最近 line_id
    wrap_count: dict[int, int] = {}
    last_line_id: dict[int, int] = {}

    for raw in vec_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # 先解析前两个整数（orb_id, line_id），允许第二个整数与后续负号相连
        m = re.match(r'^\s*(\d+)\s*(\d+)', s)
        if not m:
            continue
        try:
            raw_orb = int(m.group(1))
            line_id = int(m.group(2))
        except ValueError:
            continue

        # 判断是否发生“回卷”（同一 raw_orb 的 line_id 重置/不递增）
        if raw_orb in last_line_id and line_id <= last_line_id[raw_orb]:
            wrap_count[raw_orb] = wrap_count.get(raw_orb, 0) + 1
            last_line_id[raw_orb] = 0  # 重置以便后续递增判断
        # 计算逻辑轨道编号
        cnt = wrap_count.get(raw_orb, 0)
        if raw_orb == 0:
            adjusted_orb = (cnt + 1) * 100  # 0 -> 100, 下一次 200, ...
        else:
            adjusted_orb = raw_orb + 100 * cnt  # 1 -> 1/101/201, ...

        last_line_id[raw_orb] = line_id

        # 提取该行的全部浮点数（忽略前两个整数）
        coeffs = split_dat_line(s)
        if not coeffs:
            continue
        if adjusted_orb not in buckets:
            buckets[adjusted_orb] = []
        buckets[adjusted_orb].append((line_id, coeffs))

    if not buckets:
        raise ValueError("未在 $VEC 文本中解析到任何轨道数据")

    orb_ids = sorted(buckets.keys())
    vectors: list[np.ndarray] = []
    expected_len: int | None = None

    for oid in orb_ids:
        rows = sorted(buckets[oid], key=lambda t: t[0])
        flat: list[float] = []
        for _, cs in rows:
            flat.extend(cs)
        vec = np.asarray(flat, dtype=float)
        if expected_len is None:
            expected_len = vec.size
        elif vec.size != expected_len:
            raise ValueError(f"轨道 {oid} 的系数长度({vec.size})与期望({expected_len})不一致")
        vectors.append(vec)

    return np.vstack(vectors)

def read_fch_orbital(path: Path) -> np.ndarray:
    """
    读取并解析 Gaussian .fch 文件中的轨道数据，返回二维系数矩阵。
    Args:
        path (Path): 输入文件路径
    Returns:
        np.ndarray: 形状为 (轨道数, 基函数数) 的二维数组
    """
    p = Path(path)
    text = p.read_text(errors='ignore')
    lines = text.splitlines()

    in_vec = False
    buf: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not in_vec:
            if 'Alpha MO coefficients' in ln:
                in_vec = True
            continue
        # in_vec == True
        if s.upper().startswith("$END"):
            break
        if s:  # 跳过空行
            buf.append(ln.rstrip())
    return "\n".join(buf)

def read_nbo_orbital(nbo_path: Path, basis_functions: int) -> np.ndarray:
    """
    读取并解析 NBO .37 文件中的轨道数据，返回二维系数矩阵。
    Args:
        nbo_path (Path): NBO 文件路径
        basis_functions (int): 基函数数量
    Returns:
        array (np.ndarray): 形状为 (basis_functions + 1, basis_functions) 的二维数组，最后一行是占据数
    """
    nbo_path = Path(nbo_path).with_suffix('.37')
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

def array_to_orb(dat_orbital_matrix: np.ndarray, reorder_d: bool=True, reorder_f: bool=True, dxx_indices: list[int]=[], fxxx_indices: list[int]=[]) -> str:
    """
    将一个二维数组转换为XMVB的orb格式
    对应GVBGI类中的get_all_dat_text方法
    Args:
        dat_orbital_matrix: np.ndarray 形状为 (轨道数, 基函数数) 的二维数组
        reorder_d: bool 是否重排D轨道次序，默认True
        reorder_f: bool 是否重排F轨道次序，默认True
        dxx_indices: list[int] DXX基函数的索引列表
        fxxx_indices: list[int] FXXX基函数的索引列表
    Returns:
        str: 所有轨道的文本信息
    """
    all_pair = []
    for i, orb_matrix in enumerate(dat_orbital_matrix):
        title_line = f"# ORBITAL        {i+1}  NAO =    {len(orb_matrix)}"
        xmvb_text = make_xmvb_format_text(orb_matrix)
        all_pair.append(f"{title_line}\n{xmvb_text}\n")
    all_pair_text = ''.join(all_pair)
    if reorder_d:
        if not dxx_indices:
            print("No DXX basis functions found; skipping D orbital reordering.")
        else:
            all_pair_text = replace_xmvb_orbital_numbers(all_pair_text, dxx_indices, 3, 'd')
            print("D orbitals reordered.")
    if reorder_f:
        if not fxxx_indices:
            print("No FXXX basis functions found; skipping F orbital reordering.")
        else:
            all_pair_text = replace_xmvb_orbital_numbers(all_pair_text, fxxx_indices, 3, 'f')
            print("F orbitals reordered.")
    return all_pair_text

def main_read_gamess_dat(path: Path, all_orbital_number: int=None, d_index: list=[]) -> np.ndarray:
    """
    读取并解析 GAMESS .dat 文件中的轨道数据，返回二维系数矩阵。
    Args:
        path (Path): 输入文件路径
        all_orbital_number (optional[int]): 活性轨道最大编号
        d_index (list[int]): 需要重排的D轨道起始基函数编号列表
    Returns:
        np.ndarray: 形状为 (轨道数, 基函数数) 的二维数组
    """
    vec_text = read_gamess_dat(path)
    orbital_matrix = read_dat_orbital(vec_text)
    # 对活性轨道最大编号进行截断
    if all_orbital_number:
        orbital_matrix = orbital_matrix[:all_orbital_number, :]
        print(f"Truncated dat_orbital_matrix to first {all_orbital_number} orbitals.")
    if d_index:
        orbital_matrix = replace_col_orbital_numbers(orbital_matrix, d_index, 3, 'd')
    return orbital_matrix

def print_localized_orbitals_info(number: int, orbital: np.ndarray, atom_labels: dict, need_print: bool = False, need_atom_number:int = 0):
    """
    打印轨道的部分信息
    """
    square_sum_coeff = sum(abs(orbital)**2)
    orbital_atom = []
    for j in range(len(atom_labels)):
        # 找到第j个原子的起始和结束索引
        atom_info: dict = atom_labels[j+1]
        all_basis_index = list(atom_info['basis_functions'].keys())
        # print(all_basis_index)
        start_index = min(all_basis_index) - 1
        end_index = max(all_basis_index) - 1
        contribution = sum(abs(orbital[start_index:end_index])**2) # sum of the square of the coefficients of the basis functions of the atom
        # 计算该原子的基函数系数平方和
        if contribution / square_sum_coeff >= 0.1:
            # 如果原子的贡献大于 10%，则打印
            orbital_atom.append((j+1, atom_info['atom'], contribution / square_sum_coeff))
            # print("系数:", orbital[start_index:end_index]) # 打印该原子的基函数系数
    # print("="*80,"\n")
    orbital_atom.sort(key=lambda x: x[2], reverse=True)
    if need_atom_number > 0:
        orbital_atom = orbital_atom[:need_atom_number]
    if need_print:
        print(f"轨道 {number}:")
        print("系数平方和：", square_sum_coeff)
        print("="*80)
        for o in orbital_atom:
            print(f"原子 {o[0]} ({o[1]}) 贡献: {round(o[2]*100,1)}%")
    return orbital_atom

def print_localized_orbitals_info_pyscf(number: int, orbital: np.ndarray, mol: 'gto.Mole', need_print: bool = False, need_atom_number:int = 0) -> list[tuple[int, str, float]]:
    """
    打印轨道信息，采用pyscf的gto类型
    Args:
        number (int): 轨道编号
        orbital (np.ndarray): 轨道系数数组，一维的，每个对应一个基函数
        mol (gto): pyscf的gto对象，包含分子信息
        need_print (bool): 是否打印轨道信息，默认False
        need_atom_number (int): 需要的原子数量，默认0表示打印所有贡献大于10%的原子
    Returns:
        list[tuple[int, str, float]]: 包含原子编号、原子符号和贡献度的列表，按贡献度从高到低排序
    """
    square_sum_coeff = sum(abs(orbital)**2)
    orbital_atom = []
    # 所有基函数的列表：int 原子序号（从0开始），str 原子符号+编号，str 基函数类型, str 基函数磁量子数，长度为基函数总数
    atom_labels: list[tuple[int,str,str,str]] = mol.ao_labels(fmt=False)
    num_atoms = mol.natm
    # [壳层起始, 壳层结束, 起始基函数索引, 结束基函数索引]
    slices: list[list[int,int,int,int]] = mol.aoslice_by_atom()
    for j in range(num_atoms):
        # 第j个原子的起始和结束基函数索引
        start_index = slices[j][2]
        end_index = slices[j][3]
        # print(orbital[start_index:end_index])
        contribution = sum(abs(orbital[start_index:end_index])**2)
        # 计算该原子的基函数系数平方和
        if contribution / square_sum_coeff >= 0.1:
            # 如果原子的贡献大于 10%，则打印
            orbital_atom.append((j+1, atom_labels[start_index][1], float(contribution / square_sum_coeff)))
            # print("系数:", orbital[start_index:end_index]) # 打印该原子的基函数系数
    orbital_atom.sort(key=lambda x: x[2], reverse=True)
    if need_atom_number > 0:
        orbital_atom = orbital_atom[:need_atom_number]
    if need_print:
        print(f"轨道 {number}:")
        print("系数平方和：", square_sum_coeff)
        print("="*80)
        for o in orbital_atom:
            print(f"原子 {o[0]} ({o[1]}) 贡献: {round(o[2]*100,1)}%")
    return orbital_atom

def print_basis_function_info_pyscf(number: int, atom_num: int, orbital: np.ndarray, mol: 'gto.Mole', need_print: bool = False) -> list[tuple[int, str, float]]:
    """
    打印基函数信息，采用pyscf的gto类型，SCPA布居法（平方和贡献度分析）
    Args:
        number (int): 轨道编号
        atom_num (int): 原子编号（从1开始）
        orbital (np.ndarray): 轨道系数数组，一维的，每个对应一个基函数
        mol (gto): pyscf的gto对象，包含分子信息
        need_print (bool): 是否打印轨道信息，默认False
        need_atom_number (int): 需要的原子数量，默认0表示打印所有贡献大于10%的原子
    Returns:
        list[tuple[int, str, float]]: 包含原子编号、原子符号和贡献度的列表，按贡献度从高到低排序
    """
    # 所有基函数的列表：int 原子序号（从0开始），str 原子符号+编号，str 基函数类型, str 基函数磁量子数，长度为基函数总数
    atom_labels: list[tuple[int,str,str,str]] = mol.ao_labels(fmt=False)
    num_atoms = mol.natm
    # [壳层起始, 壳层结束, 起始基函数索引, 结束基函数索引]
    slices: list[list[int,int,int,int]] = mol.aoslice_by_atom()
    ao_basis_slice_index = slices[atom_num-1]
    start_bf = ao_basis_slice_index[2]
    end_bf = ao_basis_slice_index[3]
    ao_basis_function = atom_labels[start_bf:end_bf]

    # 将基函数归类，包括s,p,d,f等类型，得到一个字典，键为基函数类型，值为该类型的基函数索引列表
    ao_basis_function_type: dict[str, list[int]] = {}
    for i, aobf in enumerate(ao_basis_function):
        bf_type = aobf[2].strip("0123456789")
        if bf_type not in ao_basis_function_type:
            ao_basis_function_type[bf_type] = []
        ao_basis_function_type[bf_type].append(i+1)

    need_orbital = orbital[start_bf:end_bf]
    square_sum_coeff = sum(abs(need_orbital)**2)
    orbital_atom = []
    for i, abft in enumerate(ao_basis_function_type.items()):
        # 第i个基函数类型的起始和结束基函数索引
        start_index = abft[1][0]
        end_index = abft[1][-1]
        # print(start_index, end_index)
        # print(orbital[start_index:end_index])
        contribution = sum(abs(need_orbital[start_index:end_index])**2)
        # 计算该原子的基函数系数平方和
        orbital_atom.append((i+1, abft[0], float(contribution / square_sum_coeff)))
        # print("系数:", orbital[start_index:end_index]) # 打印基函数系数
    orbital_atom.sort(key=lambda x: x[2], reverse=True)
    if need_print:
        print(f"轨道 {number} 原子{atom_num}({mol.atom_pure_symbol(atom_num-1)}):")
        print("系数平方和：", square_sum_coeff)
        print("="*80)
        for o in orbital_atom:
            print(f"{o[1]}基函数 贡献: {round(o[2]*100,1)}%")
    return orbital_atom

def get_orbital_atom_contribution(orbital: np.ndarray, mol: 'gto.Mole') -> list[list[int]]:
    """
    获取一组轨道的原子贡献度信息，返回每个轨道最高贡献度的两个原子组成的list
    Args:
        orbital (np.ndarray): 轨道系数数组，二维的，每行对应一个轨道
        mol (gto): pyscf的gto对象，包含分子信息
    Returns:
        list[list[int]]: 每个轨道最高贡献度的两个原子
    """
    orbital_atom_list = []
    for i,o in enumerate(orbital):
        orbital_atom = print_localized_orbitals_info_pyscf(i+1, o, mol, need_atom_number=2)
        orbital_atom_list.append([k[0] for k in orbital_atom])
    return orbital_atom_list
    
def export_raw_molden(mol_obj: 'gto.Mole', filename, coefficient_matrix: np.ndarray):
    """
    导出 Molden 文件，不进行任何排序和系数缩放转换，原样输出 NumPy 的数值
    """
    from pyscf.tools import molden
    with open(filename, 'w') as f:
        # 1. 利用 pyscf 输出 [Atoms] 和 [GTO] 等结构信息
        molden.header(mol_obj, f)
        
        # 2. 手动写入 [MO] 部分的内容
        f.write('[MO]\n')
        num_ao, num_mo = coefficient_matrix.shape
        for i in range(num_mo):
            f.write(f" Sym= 1A\n")       # 对称性占位
            f.write(f" Ene= 0.0000\n")   # 轨道能量占位
            f.write(f" Spin= Alpha\n")   # 自旋占位
            f.write(f" Occup= 1.0000\n") # 占据数占位
            # 原样遍历输出每一列(轨道)的所有行(基函数)数值
            for j in range(num_ao):
                f.write(f" {j+1:4d} {coefficient_matrix[j, i]:20.10e}\n")

def generate_fch_from_chk(chkname: str, fchname: str, formchk_path: str = 'formchk'):
    '''
    使用 Gaussian 的 formchk 工具将 .chk 文件转换为 .fch 文件。
    '''
    print(f"Running formchk to generate {fchname}...")

    result = subprocess.run(
        f"{formchk_path} {chkname} {fchname}", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    # 获取并打印标准输出 (stdout)
    print(result.stdout)
    
    # 获取并打印错误输出 (stderr)
    if result.stderr:
        print(result.stderr)

def pyscf_to_xyz(mol: 'gto.Mole') -> str:
    """
    将 pyscf 的 gto.Mole 对象转换为 XYZ 格式的字符串
    Args:
        mol (gto.Mole): pyscf 的 gto.Mole 对象，包含分子信息
    Returns:
        str: XYZ 格式的字符串
    """
    # 获取几何坐标文本 (XYZ 格式)
    # mol.atom_coords(unit='ANG') 提取坐标，默认是Bohr，传入 'ANG' 转换为埃(Angstrom)
    coords = mol.atom_coords(unit='ANG')
    geo_lines = []
    for i in range(mol.natm):
        # 获取纯元素符号（去除数字）
        sym = mol.atom_pure_symbol(i)
        x, y, z = coords[i]
        # 格式化为 XYZ 文本
        geo_lines.append(f"{sym:2s}  {x:12.9f}  {y:12.9f}  {z:12.9f}")

    geometry_text = "\n".join(geo_lines)
    return geometry_text

def find_executable_in_env() -> str | None:
    """
    尝试查找 Gaussian 可执行文件的路径。
    检查顺序：
      1. 环境变量 GAUSS_EXEDIR 下是否存在可执行文件
      2. PATH（shutil.which）
    返回第一个找到的可执行文件绝对路径，找不到返回 None。
    """
    name = "g16"
    gauss_dir = os.environ.get("GAUSS_EXEDIR")
    if gauss_dir:
        p = Path(gauss_dir) / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    p = shutil.which(name)
    if p:
        return p
    return None

def find_tool(name: str) -> str | None:
    """通用查找工具（例如 formchk），优先 PATH。"""
    return shutil.which(name)