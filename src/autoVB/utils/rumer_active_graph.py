from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from html import escape
from math import cos, inf, pi, sin
from pathlib import Path
from typing import Sequence

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


@dataclass(slots=True)
class ComponentInfo:
    """保存一个活性子图连通分量的排序候选。

    Args:
        atoms: 该连通分量内的活性原子编号。
        kind: 分量类型，例如 single、path、ring、path-like-topology 或 branch-topology。
        degrees: 活性子图内每个原子的度数。
        orientations: 该分量可能的方向候选。
    """

    atoms: list[int]
    kind: str
    degrees: dict[int, int]
    orientations: list[list[int]]


@dataclass(slots=True)
class JoinStep:
    """记录一次分量拼接选择。

    Args:
        current_end: 当前已排序序列的末端原子。
        chosen_order: 本次选中的分量方向。
        graph_distance: 末端原子到候选起点的分子图最短距离。
    """

    current_end: int
    chosen_order: list[int]
    graph_distance: int | float


@dataclass(slots=True)
class ActiveAtomOrderResult:
    """保存活性原子排序的完整判断结果。

    Args:
        xyz_file: 输入 XYZ 文件路径；当输入为 XYZ block 时为虚拟路径。
        input_active_atoms: 用户原始给定的活性原子编号，保留重复项。
        active_atoms: 用于拓扑分析的活性原子编号，已去重并排序。
        hide_hydrogens: 是否隐藏氢后再解释原子编号。
        atom_symbols: 可见原子编号到元素符号的映射。
        bonds: 可见分子中的键列表。
        active_edges: 活性子图内的直接连接边。
        components: 活性子图连通分量信息。
        start_order: 起始分量选中的方向。
        join_steps: 后续分量拼接步骤。
        final_topology_order: 去重后的拓扑顺序。
        final_order: 按原始输入重复次数展开后的最终 Rumer 顺序。
        active_edge_crossing_count: 去重拓扑顺序下活性子图直接连接边的交叉数。
    """

    xyz_file: Path
    input_active_atoms: list[int]
    active_atoms: list[int]
    hide_hydrogens: bool
    atom_symbols: dict[int, str]
    bonds: list[tuple[int, int, str]]
    active_edges: list[tuple[int, int]]
    components: list[ComponentInfo]
    start_order: list[int]
    join_steps: list[JoinStep]
    final_topology_order: list[int]
    final_order: list[int]
    active_edge_crossing_count: int


def infer_active_atom_order(
    xyz_file: str | Path,
    active_atoms: Sequence[int],
    *,
    charge: int = 0,
    hide_hydrogens: bool = True,
) -> ActiveAtomOrderResult:
    """根据 XYZ 推断出的键连拓扑和活性原子集合自动推断 Rumer 用原子顺序。

    排序策略是：先用 RDKit 从 XYZ 推断分子图，再把活性原子诱导出的
    active graph 拆成 single/path/ring/branch 分量。path 和 ring 使用
    明确的图遍历方向；branch/path-like 分量使用纯拓扑评分优化方向。
    若有多个分量，则从包含最小原子的分量开始，之后按完整分子图最短
    路距离和原子编号选择下一个分量的方向。

    Args:
        xyz_file: 输入 XYZ 文件路径，或直接传入 XYZ block 文本。
        active_atoms: 1-based 活性原子编号；默认按隐藏氢后的可见原子编号解释。
        charge: RDKit 推断键连时使用的总电荷。
        hide_hydrogens: 是否在排序前隐藏氢原子。

    Returns:
        包含最终顺序和中文打印所需中间信息的结果对象。

    Raises:
        FileNotFoundError: XYZ 文件不存在。
        ValueError: XYZ 解析、键连推断或活性原子校验失败。
    """
    xyz_path: Path
    if isinstance(xyz_file, Path):
        xyz_path = xyz_file
        mol = build_molecule_from_xyz(
            xyz_path,
            charge=charge,
            hide_hydrogens=hide_hydrogens,
        )
    elif is_xyz_block(xyz_file):
        xyz_path = Path("xyz_block")
        mol = build_molecule_from_xyz_block(
            xyz_file,
            charge=charge,
            hide_hydrogens=hide_hydrogens,
        )
    else:
        xyz_path = Path(xyz_file)
        mol = build_molecule_from_xyz(
            xyz_path,
            charge=charge,
            hide_hydrogens=hide_hydrogens,
        )

    input_active_atoms = list(active_atoms)

    # 1. 用 RDKit 从 XYZ 文件或 XYZ block 恢复“完整分子拓扑图”。
    #    后续 active graph 只从这里推断出的键连中筛选，不再使用空间方位判断。

    # 2. active graph 的拓扑推断把 active_atoms 当作“原子集合”处理；
    #    因此这里先去重建图，最后再按原始输入中的重复次数展开 Rumer 顺序。
    normalized_active_atoms = normalize_active_atoms(input_active_atoms)
    validate_active_atoms(mol, normalized_active_atoms)

    atom_symbols = {
        atom.GetIdx() + 1: atom.GetSymbol()
        for atom in mol.GetAtoms()
    }
    # 3. adjacency 是完整可见分子的邻接表；它包含所有可见原子的直接成键关系。
    adjacency = build_adjacency(mol)
    bonds = molecule_bonds(mol)

    # 4. active_adjacency 是 active graph 的拓扑结构：
    #    只保留“两个端点都是 active atom 且在完整分子图中直接成键”的边。
    #    例如 active atoms 为 [1, 2, 4]，完整图有 1-2-3-4，
    #    active graph 只会有 1-2，不会补一条 2-4。
    active_adjacency = build_active_adjacency(adjacency, normalized_active_atoms)
    active_edges = active_subgraph_edges(active_adjacency)

    # 5. active graph 可能不是单一路径/单一环，因此先拆连通分量；
    #    每个分量再单独生成可能的圆周方向候选。
    components = build_component_infos(
        normalized_active_atoms,
        active_adjacency,
    )

    # 6. 多个分量按完整分子图距离和编号规则拼接成一个 Rumer 顺序。
    start_order, join_steps, final_topology_order = join_components(
        components,
        adjacency,
    )

    # 7. 重复 active site 不改变 active graph 拓扑；它们在最终 Rumer 顺序中紧跟
    #    对应原子展开，例如 [1, 1, 2, 3] 的拓扑分析仍使用 [1, 2, 3]。
    final_order = expand_order_with_duplicate_sites(
        final_topology_order,
        input_active_atoms,
    )

    # 8. 交叉数只用于诊断：用去重后的 topology order 判断 active graph 边是否交叉。
    active_edge_crossing_count = count_edge_crossings(final_topology_order, active_edges)

    return ActiveAtomOrderResult(
        xyz_file=xyz_path,
        input_active_atoms=input_active_atoms,
        active_atoms=normalized_active_atoms,
        hide_hydrogens=hide_hydrogens,
        atom_symbols=atom_symbols,
        bonds=bonds,
        active_edges=active_edges,
        components=components,
        start_order=start_order,
        join_steps=join_steps,
        final_topology_order=final_topology_order,
        final_order=final_order,
        active_edge_crossing_count=active_edge_crossing_count,
    )


def is_xyz_block(xyz_input: str) -> bool:
    """判断字符串输入是否更像 XYZ block 而不是文件路径。"""
    if "\n" in xyz_input or "\r" in xyz_input:
        return True

    parts = xyz_input.split()
    if len(parts) < 4:
        return False

    try:
        float(parts[1])
        float(parts[2])
        float(parts[3])
    except ValueError:
        return False
    return parts[0].isalpha()


def build_molecule_from_xyz(
    xyz_file: Path,
    *,
    charge: int,
    hide_hydrogens: bool,
) -> Chem.Mol:
    """从 XYZ 文件读取分子并推断键连。

    Args:
        xyz_file: 输入 XYZ 文件路径。
        charge: RDKit 推断键连时使用的总电荷。
        hide_hydrogens: 是否移除氢原子。

    Returns:
        完成键连推断后的 RDKit 分子。

    Raises:
        FileNotFoundError: XYZ 文件不存在。
        ValueError: XYZ 解析或键连推断失败。
    """
    if not xyz_file.exists():
        raise FileNotFoundError(f"XYZ 文件不存在: {xyz_file}")

    mol = Chem.MolFromXYZFile(str(xyz_file))
    if mol is None:
        raise ValueError(f"无法解析 XYZ 文件: {xyz_file}")

    return prepare_molecule_topology(
        mol,
        charge=charge,
        hide_hydrogens=hide_hydrogens,
        source_name=xyz_file.name,
    )


def build_molecule_from_xyz_block(
    xyz_block: str,
    *,
    charge: int,
    hide_hydrogens: bool,
) -> Chem.Mol:
    """从 XYZ block 文本读取分子并推断键连。

    Args:
        xyz_block: XYZ 文本；可以是标准 XYZ block，也可以只有原子坐标行。
        charge: RDKit 推断键连时使用的总电荷。
        hide_hydrogens: 是否移除氢原子。

    Returns:
        完成键连推断后的 RDKit 分子。

    Raises:
        ValueError: XYZ 文本为空、解析失败或键连推断失败。
    """
    normalized_block = normalize_xyz_block(xyz_block)
    mol = Chem.MolFromXYZBlock(normalized_block)
    if mol is None:
        raise ValueError("无法解析 XYZ block。")

    return prepare_molecule_topology(
        mol,
        charge=charge,
        hide_hydrogens=hide_hydrogens,
        source_name="XYZ block",
    )


def normalize_xyz_block(xyz_block: str) -> str:
    """把裸坐标行补成 RDKit 可读取的标准 XYZ block。"""
    raw_lines = xyz_block.splitlines()
    first_line_index = next(
        (idx for idx, line in enumerate(raw_lines) if line.strip()),
        None,
    )
    if first_line_index is None:
        raise ValueError("XYZ block 不能为空。")

    first_fields = raw_lines[first_line_index].split()
    if len(first_fields) == 1 and first_fields[0].isdigit():
        atom_count = int(first_fields[0])
        remaining_lines = [line.rstrip() for line in raw_lines[first_line_index + 1:]]
        nonempty_remaining = [line for line in remaining_lines if line.strip()]
        if len(nonempty_remaining) == atom_count and (
            not remaining_lines or remaining_lines[0].strip()
        ):
            remaining_lines.insert(0, "")
        return "\n".join([str(atom_count), *remaining_lines]) + "\n"

    lines = [line.strip() for line in raw_lines[first_line_index:] if line.strip()]
    return f"{len(lines)}\n\n" + "\n".join(lines) + "\n"


def prepare_molecule_topology(
    mol: Chem.Mol,
    *,
    charge: int,
    hide_hydrogens: bool,
    source_name: str,
) -> Chem.Mol:
    """推断键连、按需隐藏氢，并返回后续排序使用的可见分子。"""
    try:
        # rdDetermineBonds.DetermineBonds(mol, charge=charge)
        rdDetermineBonds.DetermineConnectivity(mol, charge=charge)
    except Exception as exc:
        raise

    visible_mol = (
        Chem.RemoveHs(mol, sanitize=False)
        if hide_hydrogens
        else Chem.Mol(mol)
    )
    try:
        Chem.Kekulize(visible_mol, clearAromaticFlags=True)
    except Exception:
        # 排序只依赖连接图，Kekulize 失败时仍可继续使用已推断的键连。
        pass

    for atom in visible_mol.GetAtoms():
        atom.SetNoImplicit(True)
    visible_mol.UpdatePropertyCache(strict=False)
    return visible_mol


def normalize_active_atoms(active_atoms: Sequence[int]) -> list[int]:
    """标准化活性原子列表。

    Args:
        active_atoms: 用户给定的活性原子编号。

    Returns:
        去重并排序后的活性原子编号列表。

    Raises:
        ValueError: 活性原子列表为空。
    """
    normalized_atoms = sorted(set(active_atoms))
    if not normalized_atoms:
        raise ValueError("active_atoms 不能为空。")
    return normalized_atoms


def expand_order_with_duplicate_sites(
    topology_order: Sequence[int],
    input_active_atoms: Sequence[int],
) -> list[int]:
    """按原始输入中的重复次数展开去重后的拓扑顺序。

    拓扑图只能表示“原子之间是否成键”，因此同一个原子出现多次时不会生成
    多个拓扑节点。最终 Rumer 顺序需要恢复这些重复 active site，并让它们
    紧跟对应原子出现。

    Args:
        topology_order: 去重后的 active graph 顺序。
        input_active_atoms: 原始活性原子输入，保留重复项。

    Returns:
        按重复次数展开后的 Rumer 顺序。
    """
    site_counts = Counter(input_active_atoms)
    expanded_order: list[int] = []
    for atom_number in topology_order:
        expanded_order.extend([atom_number] * site_counts[atom_number])
    return expanded_order


def validate_active_atoms(mol: Chem.Mol, active_atoms: Sequence[int]) -> None:
    """检查活性原子编号是否在可见原子范围内。

    Args:
        mol: 用于确定可见原子数量的分子。
        active_atoms: 1-based 活性原子编号。

    Returns:
        None.

    Raises:
        ValueError: 任一活性原子编号超出范围。
    """
    atom_count = mol.GetNumAtoms()
    for atom_number in active_atoms:
        if atom_number < 1 or atom_number > atom_count:
            raise ValueError(
                f"活性原子编号必须在 1~{atom_count} 之间，但得到 {atom_number}。"
            )


def build_adjacency(mol: Chem.Mol) -> dict[int, list[int]]:
    """构建分子图邻接表。

    Args:
        mol: RDKit 分子。

    Returns:
        1-based 原子编号邻接表。
    """
    # 完整分子拓扑图的基础表示：
    # key 是 1-based 原子编号，value 是与该原子“直接成键”的邻居编号。
    adjacency: dict[int, list[int]] = {
        atom.GetIdx() + 1: []
        for atom in mol.GetAtoms()
    }
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx() + 1
        end_atom = bond.GetEndAtomIdx() + 1
        adjacency[begin_atom].append(end_atom)
        adjacency[end_atom].append(begin_atom)

    return {
        atom_number: sorted(neighbors)
        for atom_number, neighbors in adjacency.items()
    }


def molecule_bonds(mol: Chem.Mol) -> list[tuple[int, int, str]]:
    """读取分子中的键。

    Args:
        mol: RDKit 分子。

    Returns:
        `(起点, 终点, 键类型)` 形式的键列表。
    """
    bonds: list[tuple[int, int, str]] = []
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx() + 1
        end_atom = bond.GetEndAtomIdx() + 1
        bond_type = str(bond.GetBondType()).split(".")[-1]
        bonds.append((*sorted((begin_atom, end_atom)), bond_type))
    return sorted(bonds)


def build_active_adjacency(
    adjacency: dict[int, list[int]],
    active_atoms: Sequence[int],
) -> dict[int, list[int]]:
    """构建只包含活性原子直接连接的诱导子图。

    Args:
        adjacency: 完整分子图邻接表。
        active_atoms: 活性原子编号。

    Returns:
        活性子图邻接表。
    """
    active_set = set(active_atoms)

    # active graph 是完整分子图在 active atom 集合上的“诱导子图”：
    # 只筛选已经存在的直接键连，不根据最短路或空间距离额外加边。
    return {
        atom_number: [
            neighbor
            for neighbor in adjacency[atom_number]
            if neighbor in active_set
        ]
        for atom_number in active_atoms
    }


def active_subgraph_edges(
    active_adjacency: dict[int, list[int]],
) -> list[tuple[int, int]]:
    """读取活性子图中的直接连接边。

    Args:
        active_adjacency: 活性子图邻接表。

    Returns:
        去重后的活性子图边列表。
    """
    # 邻接表中每条无向边会出现两次，例如 1->2 和 2->1；
    # 排序成 tuple 后放入 set，可以得到去重后的 active graph 边集合。
    edges = {
        tuple(sorted((atom_number, neighbor)))
        for atom_number, neighbors in active_adjacency.items()
        for neighbor in neighbors
    }
    return sorted(edges)


def build_component_infos(
    active_atoms: Sequence[int],
    active_adjacency: dict[int, list[int]],
) -> list[ComponentInfo]:
    """拆分活性子图并生成每个分量的方向候选。

    Args:
        active_atoms: 活性原子编号。
        active_adjacency: 活性子图邻接表。

    Returns:
        连通分量信息列表。
    """
    # active graph 可能是多个互不相连的片段；
    # 每个片段需要独立判断它是 path、ring、single 还是 branch。
    components = connected_components(active_atoms, active_adjacency)
    component_infos = [
        build_component_info(component, active_adjacency)
        for component in components
    ]
    return sorted(component_infos, key=lambda info: min(info.atoms))


def connected_components(
    active_atoms: Sequence[int],
    active_adjacency: dict[int, list[int]],
) -> list[list[int]]:
    """计算活性子图连通分量。

    Args:
        active_atoms: 活性原子编号。
        active_adjacency: 活性子图邻接表。

    Returns:
        每个连通分量包含的原子编号。
    """
    seen: set[int] = set()
    components: list[list[int]] = []

    for atom_number in sorted(active_atoms):
        if atom_number in seen:
            continue

        # BFS 只沿 active graph 的边走，因此得到的是 active graph 的连通分量，
        # 不是完整分子图的连通分量。
        component: list[int] = []
        queue: deque[int] = deque([atom_number])
        seen.add(atom_number)
        while queue:
            current_atom = queue.popleft()
            component.append(current_atom)
            for neighbor in active_adjacency[current_atom]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        components.append(sorted(component))

    return components


def build_component_info(
    component: list[int],
    active_adjacency: dict[int, list[int]],
) -> ComponentInfo:
    """为一个活性子图分量生成方向候选。

    Args:
        component: 连通分量原子编号。
        active_adjacency: 活性子图邻接表。

    Returns:
        连通分量信息。
    """
    component_set = set(component)

    # 度数只在 active graph 内计算：
    # path 的端点度数为 1，ring 中每个点度数通常为 2，branch 会出现度数大于 2 的点。
    degrees = {
        atom_number: len(
            [
                neighbor
                for neighbor in active_adjacency[atom_number]
                if neighbor in component_set
            ]
        )
        for atom_number in component
    }

    if len(component) == 1:
        # 单点分量没有方向选择。
        orientations = [component[:]]
        kind = "single"
    elif all(degree <= 2 for degree in degrees.values()):
        endpoints = [
            atom_number
            for atom_number, degree in degrees.items()
            if degree <= 1
        ]
        if len(endpoints) == 2:
            # path 分量从两个端点分别遍历，得到正反两个候选方向。
            orientations = [
                traverse_path(endpoints[0], component_set, active_adjacency),
                traverse_path(endpoints[1], component_set, active_adjacency),
            ]
            kind = "path"
        elif len(endpoints) == 0:
            # ring 分量没有端点；固定从最小编号出发，生成顺/逆两个候选方向。
            orientations = cycle_orientations(component, active_adjacency)
            kind = "ring"
        else:
            # 少数异常 path-like 结构无法只靠端点判断方向，使用拓扑评分优化。
            orientations = topology_optimized_orientations(component, active_adjacency)
            kind = "path-like-topology"
    else:
        # branch 分量有分叉点，不存在单一图遍历路径；使用拓扑评分优化。
        orientations = topology_optimized_orientations(component, active_adjacency)
        kind = "branch-topology"

    return ComponentInfo(
        atoms=component,
        kind=kind,
        degrees=degrees,
        orientations=dedupe_orders(orientations),
    )


def topology_optimized_orientations(
    component: list[int],
    active_adjacency: dict[int, list[int]],
) -> list[list[int]]:
    """为非 path/ring 分量生成纯拓扑排序候选。

    该函数只读取 active graph 邻接表。它先从编号顺序、DFS、BFS 等多个
    拓扑遍历结果出发，再用 2-opt 片段翻转减少 active graph 边交叉。

    Args:
        component: 当前 active graph 连通分量的原子编号。
        active_adjacency: active graph 邻接表。

    Returns:
        拓扑评分最好的候选顺序及其反向等价顺序。
    """
    if len(component) <= 1:
        return [component[:]]

    component_edges = component_subgraph_edges(component, active_adjacency)
    initial_orders = initial_topology_orders(component, active_adjacency)
    optimized_orders = [
        optimize_order_by_topology(order, component_edges, active_adjacency)
        for order in initial_orders
    ]
    best_score = min(
        topology_order_score(order, component_edges, active_adjacency)
        for order in optimized_orders
    )
    best_orders = [
        rotate_order_to_min_atom(order)
        for order in optimized_orders
        if topology_order_score(order, component_edges, active_adjacency) == best_score
    ]

    orientations: list[list[int]] = []
    for order in best_orders:
        orientations.append(order)
        orientations.append(rotate_order_to_min_atom(list(reversed(order))))
    return dedupe_orders(orientations)


def component_subgraph_edges(
    component: Sequence[int],
    active_adjacency: dict[int, list[int]],
) -> list[tuple[int, int]]:
    """读取一个 active graph 连通分量内部的无向边。

    Args:
        component: 连通分量原子编号。
        active_adjacency: active graph 邻接表。

    Returns:
        分量内部去重后的无向边。
    """
    component_set = set(component)
    edges = {
        tuple(sorted((atom_number, neighbor)))
        for atom_number in component
        for neighbor in active_adjacency[atom_number]
        if neighbor in component_set
    }
    return sorted(edges)


def initial_topology_orders(
    component: Sequence[int],
    active_adjacency: dict[int, list[int]],
) -> list[list[int]]:
    """生成拓扑优化的初始顺序集合。

    Args:
        component: 连通分量原子编号。
        active_adjacency: active graph 邻接表。

    Returns:
        编号顺序、DFS 顺序、BFS 顺序等候选。
    """
    orders: list[list[int]] = [sorted(component), sorted(component, reverse=True)]
    for start_atom in sorted(component):
        orders.append(graph_dfs_order(start_atom, component, active_adjacency))
        orders.append(graph_dfs_order(start_atom, component, active_adjacency, reverse=True))
        orders.append(graph_bfs_order(start_atom, component, active_adjacency))
        orders.append(graph_bfs_order(start_atom, component, active_adjacency, reverse=True))

    component_key = sorted(component)
    valid_orders = [
        order
        for order in orders
        if sorted(order) == component_key
    ]
    return dedupe_orders(valid_orders)


def graph_dfs_order(
    start_atom: int,
    component: Sequence[int],
    active_adjacency: dict[int, list[int]],
    *,
    reverse: bool = False,
) -> list[int]:
    """从指定起点生成 DFS 拓扑顺序。

    Args:
        start_atom: DFS 起点。
        component: 连通分量原子编号。
        active_adjacency: active graph 邻接表。
        reverse: 是否按编号降序访问邻居。

    Returns:
        DFS 访问顺序。
    """
    component_set = set(component)
    seen: set[int] = set()
    order: list[int] = []

    def visit(atom_number: int) -> None:
        seen.add(atom_number)
        order.append(atom_number)
        neighbors = [
            neighbor
            for neighbor in active_adjacency[atom_number]
            if neighbor in component_set and neighbor not in seen
        ]
        for neighbor in sorted(neighbors, reverse=reverse):
            if neighbor not in seen:
                visit(neighbor)

    visit(start_atom)
    return order


def graph_bfs_order(
    start_atom: int,
    component: Sequence[int],
    active_adjacency: dict[int, list[int]],
    *,
    reverse: bool = False,
) -> list[int]:
    """从指定起点生成 BFS 拓扑顺序。

    Args:
        start_atom: BFS 起点。
        component: 连通分量原子编号。
        active_adjacency: active graph 邻接表。
        reverse: 是否按编号降序访问邻居。

    Returns:
        BFS 访问顺序。
    """
    component_set = set(component)
    seen = {start_atom}
    order: list[int] = []
    queue: deque[int] = deque([start_atom])

    while queue:
        atom_number = queue.popleft()
        order.append(atom_number)
        neighbors = [
            neighbor
            for neighbor in active_adjacency[atom_number]
            if neighbor in component_set and neighbor not in seen
        ]
        for neighbor in sorted(neighbors, reverse=reverse):
            seen.add(neighbor)
            queue.append(neighbor)

    return order


def optimize_order_by_topology(
    order: Sequence[int],
    edges: Sequence[tuple[int, int]],
    active_adjacency: dict[int, list[int]],
) -> list[int]:
    """用 2-opt 片段翻转优化圆周顺序。

    Args:
        order: 初始圆周顺序。
        edges: 当前分量内部的 active graph 边。
        active_adjacency: active graph 邻接表。

    Returns:
        拓扑评分局部最优的圆周顺序。
    """
    best_order = rotate_order_to_min_atom(order)
    best_score = topology_order_score(best_order, edges, active_adjacency)
    improved = True

    while improved:
        improved = False
        atom_count = len(best_order)
        for begin_index in range(1, atom_count - 1):
            for end_index in range(begin_index + 1, atom_count):
                candidate = (
                    best_order[:begin_index]
                    + list(reversed(best_order[begin_index : end_index + 1]))
                    + best_order[end_index + 1 :]
                )
                candidate = rotate_order_to_min_atom(candidate)
                candidate_score = topology_order_score(candidate, edges, active_adjacency)
                if candidate_score < best_score:
                    best_order = candidate
                    best_score = candidate_score
                    improved = True
                    break
            if improved:
                break

    return best_order


def topology_order_score(
    order: Sequence[int],
    edges: Sequence[tuple[int, int]],
    active_adjacency: dict[int, list[int]],
) -> tuple[int, int, float, tuple[int, ...]]:
    """计算一个圆周顺序的纯拓扑评分。

    Args:
        order: 待评分的圆周顺序。
        edges: 当前分量内部的 active graph 边。
        active_adjacency: active graph 邻接表。

    Returns:
        评分元组，越小越好：交叉数、负的圆周相邻边数、圆周相邻图距离总和、规范顺序。
    """
    canonical_order = rotate_order_to_min_atom(order)
    return (
        count_edge_crossings(canonical_order, edges),
        -adjacent_edge_count(canonical_order, edges),
        circular_graph_distance_sum(canonical_order, active_adjacency),
        tuple(canonical_order),
    )


def adjacent_edge_count(
    order: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> int:
    """统计 active graph 边有多少条同时也是圆周相邻边。

    Args:
        order: 圆周顺序。
        edges: active graph 边。

    Returns:
        圆周相邻的 active graph 边数量。
    """
    edge_set = {tuple(sorted(edge)) for edge in edges}
    count = 0
    for index, atom_number in enumerate(order):
        next_atom = order[(index + 1) % len(order)]
        if tuple(sorted((atom_number, next_atom))) in edge_set:
            count += 1
    return count


def circular_graph_distance_sum(
    order: Sequence[int],
    active_adjacency: dict[int, list[int]],
) -> float:
    """计算圆周相邻点在 active graph 上的最短路距离总和。

    Args:
        order: 圆周顺序。
        active_adjacency: active graph 邻接表。

    Returns:
        相邻点图距离总和；不连通时返回无穷大。
    """
    total = 0.0
    for index, atom_number in enumerate(order):
        next_atom = order[(index + 1) % len(order)]
        distance = shortest_path_distance(active_adjacency, atom_number, next_atom)
        if distance == inf:
            return inf
        total += float(distance)
    return total


def rotate_order_to_min_atom(order: Sequence[int]) -> list[int]:
    """把环状顺序旋转到最小原子编号开头。

    Args:
        order: 原始顺序。

    Returns:
        以最小原子编号开头的顺序。
    """
    min_atom = min(order)
    min_index = list(order).index(min_atom)
    return list(order[min_index:]) + list(order[:min_index])


def traverse_path(
    start_atom: int,
    component_set: set[int],
    active_adjacency: dict[int, list[int]],
) -> list[int]:
    """从端点开始遍历 path 分量。

    Args:
        start_atom: 起始端点。
        component_set: 当前分量原子集合。
        active_adjacency: 活性子图邻接表。

    Returns:
        path 方向上的原子顺序。
    """
    order: list[int] = []
    previous_atom: int | None = None
    current_atom = start_atom

    while True:
        order.append(current_atom)
        next_atoms = [
            neighbor
            for neighbor in active_adjacency[current_atom]
            if neighbor in component_set and neighbor != previous_atom
        ]
        if not next_atoms:
            break
        previous_atom, current_atom = current_atom, sorted(next_atoms)[0]

    return order


def cycle_orientations(
    component: list[int],
    active_adjacency: dict[int, list[int]],
) -> list[list[int]]:
    """为 ring 分量生成两个方向候选。

    Args:
        component: ring 分量原子编号。
        active_adjacency: 活性子图邻接表。

    Returns:
        从最小原子开始的顺/逆两个方向。
    """
    start_atom = min(component)
    neighbors = sorted(
        neighbor
        for neighbor in active_adjacency[start_atom]
        if neighbor in set(component)
    )
    return [
        traverse_cycle(start_atom, neighbor, active_adjacency)
        for neighbor in neighbors[:2]
    ]


def traverse_cycle(
    start_atom: int,
    next_atom: int,
    active_adjacency: dict[int, list[int]],
) -> list[int]:
    """沿 ring 的一个方向遍历。

    Args:
        start_atom: ring 起点。
        next_atom: 起点之后的第一个邻居。
        active_adjacency: 活性子图邻接表。

    Returns:
        ring 上一个方向的原子顺序，不重复写入起点。
    """
    order = [start_atom]
    previous_atom = start_atom
    current_atom = next_atom

    while current_atom != start_atom:
        order.append(current_atom)
        next_atoms = [
            neighbor
            for neighbor in active_adjacency[current_atom]
            if neighbor != previous_atom
        ]
        if not next_atoms:
            break
        previous_atom, current_atom = current_atom, sorted(next_atoms)[0]

    return order


def fallback_orientations(
    component: list[int],
    active_adjacency: dict[int, list[int]],
) -> list[list[int]]:
    """为非 path/ring 分量生成确定性的 DFS 兜底方向。

    Args:
        component: 连通分量原子编号。
        active_adjacency: 活性子图邻接表。

    Returns:
        DFS 顺序和其反向。
    """
    start_atom = min(component)
    component_set = set(component)
    order: list[int] = []
    seen: set[int] = set()

    def visit(atom_number: int) -> None:
        seen.add(atom_number)
        order.append(atom_number)
        for neighbor in active_adjacency[atom_number]:
            if neighbor in component_set and neighbor not in seen:
                visit(neighbor)

    visit(start_atom)
    return [order, list(reversed(order))]


def dedupe_orders(orders: Sequence[Sequence[int]]) -> list[list[int]]:
    """去除重复方向候选。

    Args:
        orders: 方向候选列表。

    Returns:
        去重后的方向候选列表。
    """
    seen: set[tuple[int, ...]] = set()
    unique_orders: list[list[int]] = []
    for order in orders:
        key = tuple(order)
        if key in seen:
            continue
        seen.add(key)
        unique_orders.append(list(order))
    return unique_orders


def join_components(
    components: Sequence[ComponentInfo],
    adjacency: dict[int, list[int]],
) -> tuple[list[int], list[JoinStep], list[int]]:
    """把多个活性分量拼接成一个 Rumer 顺序。

    Args:
        components: 活性子图分量信息。
        adjacency: 完整分子图邻接表。

    Returns:
        `(起始分量顺序, 拼接步骤, 最终顺序)`。
    """
    if not components:
        return [], [], []

    # 起始分量固定选包含最小 active atom 的分量，保证同一输入下输出稳定。
    min_active_atom = min(atom for info in components for atom in info.atoms)
    start_component = next(
        info
        for info in components
        if min_active_atom in info.atoms
    )
    start_candidates = [
        order
        for order in start_component.orientations
        if order and order[0] == min_active_atom
    ]
    if not start_candidates:
        start_candidates = start_component.orientations
    start_order = min(start_candidates, key=lambda order: tuple(order))

    final_order = list(start_order)
    remaining_components = [
        info
        for info in components
        if info is not start_component
    ]
    join_steps: list[JoinStep] = []

    while remaining_components:
        current_end = final_order[-1]
        best_key: tuple[float, int, tuple[int, ...]] | None = None
        best_component: ComponentInfo | None = None
        best_order: list[int] | None = None
        best_graph_distance: int | float = inf

        for component in remaining_components:
            for order in component.orientations:
                candidate_start = order[0]

                # 拼接多个 active graph 分量时，使用“完整分子图”的最短路距离，
                # 而不是 active graph 距离；这样非 active 原子也能作为连接桥梁参与判断。
                graph_distance = shortest_path_distance(
                    adjacency,
                    current_end,
                    candidate_start,
                )
                key = (
                    float(graph_distance),
                    candidate_start,
                    tuple(order),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_component = component
                    best_order = order
                    best_graph_distance = graph_distance

        if best_component is None or best_order is None:
            raise ValueError("无法拼接活性原子分量。")

        final_order.extend(best_order)
        remaining_components.remove(best_component)
        join_steps.append(
            JoinStep(
                current_end=current_end,
                chosen_order=best_order,
                graph_distance=best_graph_distance,
            )
        )

    return start_order, join_steps, final_order


def shortest_path_distance(
    adjacency: dict[int, list[int]],
    start_atom: int,
    end_atom: int,
) -> int | float:
    """计算完整分子图上两个原子的最短路径距离。

    Args:
        adjacency: 完整分子图邻接表。
        start_atom: 起始原子。
        end_atom: 目标原子。

    Returns:
        最短路径边数；若不连通则返回无穷大。
    """
    if start_atom == end_atom:
        return 0

    # BFS 在完整分子图上找最少键数，用于判断两个 active 分量在真实分子中谁更接近。
    seen = {start_atom}
    queue: deque[tuple[int, int]] = deque([(start_atom, 0)])
    while queue:
        atom_number, distance = queue.popleft()
        for neighbor in adjacency[atom_number]:
            if neighbor == end_atom:
                return distance + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))

    return inf


def count_edge_crossings(
    atom_order: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> int:
    """计算给定圆周顺序下边的交叉数量。

    共用端点的两条边不算交叉。这个数值用于辅助判断当前顺序是否适合
    作为 Rumer 圆周顺序；通常越小越好。

    Args:
        atom_order: 活性原子圆周顺序。
        edges: 活性子图中的直接连接边。

    Returns:
        边交叉数量。
    """
    # 把圆周顺序转换成线性位置。两条边 (a,b)、(c,d) 在圆内相交，
    # 等价于四个端点在圆周上交替出现：a < c < b < d 或 c < a < d < b。
    atom_positions = {
        atom_number: index
        for index, atom_number in enumerate(atom_order)
    }
    crossing_count = 0

    for edge_index, first_edge in enumerate(edges):
        if first_edge[0] not in atom_positions or first_edge[1] not in atom_positions:
            continue
        first_begin = atom_positions[first_edge[0]]
        first_end = atom_positions[first_edge[1]]
        if first_begin > first_end:
            first_begin, first_end = first_end, first_begin

        for second_edge in edges[edge_index + 1 :]:
            # 共享端点的两条边在拓扑图里相接，不算作 crossing。
            if len(set(first_edge + second_edge)) < 4:
                continue
            if (
                second_edge[0] not in atom_positions
                or second_edge[1] not in atom_positions
            ):
                continue

            second_begin = atom_positions[second_edge[0]]
            second_end = atom_positions[second_edge[1]]
            if second_begin > second_end:
                second_begin, second_end = second_end, second_begin

            if (
                first_begin < second_begin < first_end < second_end
                or second_begin < first_begin < second_end < first_end
            ):
                crossing_count += 1

    return crossing_count


def print_order_process(result: ActiveAtomOrderResult) -> None:
    """用中文打印活性原子顺序判断过程。

    Args:
        result: 活性原子排序结果。

    Returns:
        None.
    """
    print(f"读取 XYZ : {result.xyz_file.resolve()}")
    print(f"是否隐藏氢原子后编号: {result.hide_hydrogens}")
    print()

    print("可见原子:")
    for atom_number, symbol in result.atom_symbols.items():
        active_mark = " <- active" if atom_number in result.active_atoms else ""
        print(f"  {atom_number}: {symbol}{active_mark}")
    print()

    print(f"给定活性原子: {result.input_active_atoms}")
    if result.input_active_atoms != result.active_atoms:
        print(f"拓扑分析去重后活性原子: {result.active_atoms}")
    print("RDKit 推断的可见原子键连:")
    for begin_atom, end_atom, bond_type in result.bonds:
        print(f"  {begin_atom}-{end_atom}: {bond_type}")
    print()

    print("活性子图中的直接连接边:")
    if result.active_edges:
        for begin_atom, end_atom in result.active_edges:
            print(f"  {begin_atom}-{end_atom}")
    else:
        print("  无直接连接边。")
    print()

    print("活性子图连通分量:")
    has_fallback_component = False
    for index, component in enumerate(result.components, start=1):
        print(f"  分量 {index}: atoms={component.atoms}, 类型={component.kind}")
        print(f"    活性子图内度数: {component.degrees}")
        print(f"    方向候选: {component.orientations}")
        if component.kind in {
            "branch-topology",
            "path-like-topology",
        }:
            has_fallback_component = True
    if has_fallback_component:
        print(
            "  注意：存在 branch/path-like 分量，脚本使用 active graph 拓扑评分优化方向；"
            "这类体系可能有多个等价或近似等价顺序，建议结合结构图复核。"
        )
    print()

    print("拼接判断:")
    print(
        "  起始分量选择包含最小活性原子的方向: "
        f"{result.start_order}"
    )
    for step in result.join_steps:
        print(
            "  当前末端 "
            f"{step.current_end} -> 选择 {step.chosen_order}; "
            f"分子图距离={step.graph_distance}"
        )
    print()

    if result.final_topology_order != result.final_order:
        print(f"去重拓扑顺序: {result.final_topology_order}")
    print(f"最终推断顺序: {result.final_order}")
    print(
        "最终顺序下活性子图直接连接边的交叉数: "
        f"{result.active_edge_crossing_count}"
    )


def print_order_process_en(result: ActiveAtomOrderResult) -> None:
    """Print the active atom order inference process in English.

    Args:
        result: Active atom order inference result.

    Returns:
        None.
    """
    print(f"Read XYZ: {result.xyz_file.resolve()}")
    print(f"Atom numbering after hiding hydrogens: {result.hide_hydrogens}")
    print()

    print("Visible atoms:")
    for atom_number, symbol in result.atom_symbols.items():
        active_mark = " <- active" if atom_number in result.active_atoms else ""
        print(f"  {atom_number}: {symbol}{active_mark}")
    print()

    print(f"Input active atoms: {result.input_active_atoms}")
    if result.input_active_atoms != result.active_atoms:
        print(f"Deduplicated active atoms for topology analysis: {result.active_atoms}")
    print("RDKit-inferred bonds between visible atoms:")
    for begin_atom, end_atom, bond_type in result.bonds:
        print(f"  {begin_atom}-{end_atom}: {bond_type}")
    print()

    print("Direct edges in the active subgraph:")
    if result.active_edges:
        for begin_atom, end_atom in result.active_edges:
            print(f"  {begin_atom}-{end_atom}")
    else:
        print("  No direct edges.")
    print()

    print("Connected components in the active subgraph:")
    has_fallback_component = False
    for index, component in enumerate(result.components, start=1):
        print(f"  Component {index}: atoms={component.atoms}, type={component.kind}")
        print(f"    Degrees in the active subgraph: {component.degrees}")
        print(f"    Orientation candidates: {component.orientations}")
        if component.kind in {
            "branch-topology",
            "path-like-topology",
        }:
            has_fallback_component = True
    if has_fallback_component:
        print(
            "  Note: branch/path-like components are present. The script uses "
            "active-graph topology scoring to optimize orientations; such systems "
            "may have multiple equivalent or near-equivalent orders, so reviewing "
            "the structure graph is recommended."
        )
    print()

    print("Component joining decisions:")
    print(
        "  Initial component orientation chosen from the component containing "
        f"the smallest active atom: {result.start_order}"
    )
    for step in result.join_steps:
        print(
            "  Current end "
            f"{step.current_end} -> choose {step.chosen_order}; "
            f"molecular graph distance={step.graph_distance}"
        )
    print()

    if result.final_topology_order != result.final_order:
        print(f"Deduplicated topology order: {result.final_topology_order}")
    print(f"Final inferred order: {result.final_order}")
    print(
        "Crossing count of direct active-subgraph edges in the final order: "
        f"{result.active_edge_crossing_count}"
    )


def write_active_graph_topology_svg(
    result: ActiveAtomOrderResult,
    output_dir: str | Path,
    svg_filename: str | None = None,
) -> Path:
    """写出 active graph 的 Rumer 圆周拓扑 SVG。

    Args:
        result: `infer_active_atom_order` 返回的排序结果。
        output_dir: SVG 输出目录。

    Returns:
        写出的 SVG 文件路径。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not svg_filename:
        svg_file = out_dir / f"{result.xyz_file.stem}_active_graph_topology.svg"
    else:
        svg_file = out_dir / svg_filename
    svg_file.write_text(render_active_graph_topology_svg(result), encoding="utf-8")
    return svg_file


def render_active_graph_topology_svg(result: ActiveAtomOrderResult) -> str:
    """渲染 active graph 圆周拓扑图，不使用真实分子几何。

    Args:
        result: `infer_active_atom_order` 返回的排序结果。

    Returns:
        可直接写入 `.svg` 文件的 SVG 文本。
    """
    width = 760
    height = 760
    center_x = width / 2
    center_y = height / 2 + 10
    radius = 250
    label_radius = radius + 38
    topology_order = list(result.final_topology_order)
    final_order = list(result.final_order)
    edge_set = set(result.active_edges)
    crossing_edges = crossing_edge_set(topology_order, result.active_edges)

    positions: dict[int, tuple[float, float]] = {}
    label_positions: dict[int, tuple[float, float]] = {}
    atom_count = max(len(topology_order), 1)
    for index, atom_number in enumerate(topology_order):
        angle = -pi / 2 + 2 * pi * index / atom_count
        positions[atom_number] = (
            center_x + radius * cos(angle),
            center_y + radius * sin(angle),
        )
        label_positions[atom_number] = (
            center_x + label_radius * cos(angle),
            center_y + label_radius * sin(angle),
        )

    final_order_positions: dict[int, list[int]] = {}
    for order_index, atom_number in enumerate(final_order, start=1):
        final_order_positions.setdefault(atom_number, []).append(order_index)

    parts = [
        "<?xml version='1.0' encoding='UTF-8'?>\n",
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' "
        f"height='{height}' viewBox='0 0 {width} {height}'>\n",
        "<rect width='100%' height='100%' fill='#ffffff'/>\n",
        "<style>\n",
        "  text { font-family: Arial, Helvetica, sans-serif; }\n",
        "  .title { font-size: 22px; font-weight: 700; fill: #111827; }\n",
        "  .subtitle { font-size: 13px; fill: #4b5563; }\n",
        "  .atom { font-size: 14px; font-weight: 700; text-anchor: middle; dominant-baseline: central; }\n",
        "  .order { font-size: 11px; fill: #4b5563; text-anchor: middle; dominant-baseline: central; }\n",
        "</style>\n",
        f"<text class='title' x='30' y='38'>Active graph topology for "
        f"{escape(result.xyz_file.stem)}</text>\n",
        f"<text class='subtitle' x='30' y='62'>Final order: "
        f"{escape(str(final_order))}</text>\n",
        f"<text class='subtitle' x='30' y='82'>Topology order: "
        f"{escape(str(topology_order))}</text>\n",
        f"<text class='subtitle' x='30' y='102'>Crossing count: "
        f"{result.active_edge_crossing_count}</text>\n",
        f"<circle cx='{center_x:.3f}' cy='{center_y:.3f}' r='{radius:.3f}' "
        "fill='none' stroke='#e5e7eb' stroke-width='2'/>\n",
    ]

    # 先画 active graph 边；发生交叉的边用红色，其余边用蓝色。
    for edge in sorted(edge_set):
        begin_atom, end_atom = edge
        if begin_atom not in positions or end_atom not in positions:
            continue
        begin_x, begin_y = positions[begin_atom]
        end_x, end_y = positions[end_atom]
        is_crossing = edge in crossing_edges
        stroke = "#dc2626" if is_crossing else "#2563eb"
        stroke_width = "3.2" if is_crossing else "2.0"
        parts.append(
            f"<line x1='{begin_x:.3f}' y1='{begin_y:.3f}' "
            f"x2='{end_x:.3f}' y2='{end_y:.3f}' "
            f"stroke='{stroke}' stroke-width='{stroke_width}' stroke-linecap='round' "
            "stroke-opacity='0.82'/>\n"
        )

    # 再画圆周顺序的相邻连接，作为读顺序的参考。
    if len(topology_order) >= 2:
        order_points = " ".join(
            f"{positions[atom][0]:.3f},{positions[atom][1]:.3f}"
            for atom in topology_order + [topology_order[0]]
        )
        parts.append(
            f"<polyline points='{order_points}' fill='none' stroke='#9ca3af' "
            "stroke-width='1.4' stroke-dasharray='4 6' stroke-linejoin='round'/>\n"
        )

    for atom_number in topology_order:
        x_value, y_value = positions[atom_number]
        label_x, label_y = label_positions[atom_number]
        symbol = result.atom_symbols.get(atom_number, "")
        order_label = ",".join(
            str(index)
            for index in final_order_positions.get(atom_number, [])
        )
        parts.extend(
            [
                f"<circle cx='{x_value:.3f}' cy='{y_value:.3f}' r='18' "
                "fill='#111827' stroke='#111827' stroke-width='1.2'/>\n",
                f"<text class='atom' x='{x_value:.3f}' y='{y_value:.3f}' "
                f"fill='#ffffff'>{escape(symbol + str(atom_number))}</text>\n",
                f"<circle cx='{label_x:.3f}' cy='{label_y:.3f}' r='11' "
                "fill='#f9fafb' stroke='#9ca3af' stroke-width='1'/>\n",
                f"<text class='order' x='{label_x:.3f}' y='{label_y:.3f}'>"
                f"{escape(order_label)}</text>\n",
            ]
        )

    parts.extend(
        [
            "<text class='subtitle' x='30' y='724'>"
            "Blue lines = active graph edges; red lines = edges participating in crossings; "
            "gray dashed polygon = deduplicated topology order; repeated sites share one topology node."
            "</text>\n",
            "</svg>\n",
        ]
    )
    return "".join(parts)


def crossing_edge_set(
    atom_order: Sequence[int],
    edges: Sequence[tuple[int, int]],
) -> set[tuple[int, int]]:
    """找出在给定圆周顺序中参与交叉的 active graph 边。

    Args:
        atom_order: active atoms 的圆周顺序。
        edges: active graph 中的直接连接边。

    Returns:
        至少参与一次交叉的边集合。
    """
    atom_positions = {
        atom_number: index
        for index, atom_number in enumerate(atom_order)
    }
    crossing_edges: set[tuple[int, int]] = set()

    for edge_index, first_edge in enumerate(edges):
        if first_edge[0] not in atom_positions or first_edge[1] not in atom_positions:
            continue
        first_begin = atom_positions[first_edge[0]]
        first_end = atom_positions[first_edge[1]]
        if first_begin > first_end:
            first_begin, first_end = first_end, first_begin

        for second_edge in edges[edge_index + 1 :]:
            if len(set(first_edge + second_edge)) < 4:
                continue
            if (
                second_edge[0] not in atom_positions
                or second_edge[1] not in atom_positions
            ):
                continue

            second_begin = atom_positions[second_edge[0]]
            second_end = atom_positions[second_edge[1]]
            if second_begin > second_end:
                second_begin, second_end = second_end, second_begin

            crosses = (
                first_begin < second_begin < first_end < second_end
                or second_begin < first_begin < second_end < first_end
            )
            if crosses:
                crossing_edges.add(tuple(sorted(first_edge)))
                crossing_edges.add(tuple(sorted(second_edge)))

    return crossing_edges
