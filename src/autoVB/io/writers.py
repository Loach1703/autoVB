from pathlib import Path
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass
import datetime
import json
import re

from ..utils.constants import SUPPORTED_METHODS, to_gaussian_basis_name

if TYPE_CHECKING:
    from ..main import XMIPassthrough, autoVBInputData
    from .xmo_output_parser import XmoParsedData, XmoStructureWeight
    from pyscf import gto

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
    ncharge: int
    nmul: int
    basis_set: str
    sort: bool
    orb_section: str
    geo_section: str
    init_guess_section: str

def write_xmi_file(filename: str, xmidata: XMIData, xmi_passthrough: 'XMIPassthrough'= None):
    '''
    将 XMIData 数据格式写入 .xmi 文件。
    Args:
        filename (str): 输出文件名
        xmidata (XMIData): 包含轨道数据和相关信息的对象
        xmi_passthrough (XMIPassthrough): 包含透传数据的对象，可选
    '''
    xmi_path = Path(filename).with_suffix('.xmi')
    def ctrl_key(line: str) -> str:
        s = line.strip().lower()
        if not s:
            return ""
        if "=" in s:
            return s.split("=", 1)[0].strip()
        return s.split()[0]

    if xmi_passthrough:
        extra_lines = list(xmi_passthrough.ctrl_extra_lines) if xmi_passthrough.ctrl_extra_lines else []
        extra_keys = {ctrl_key(line) for line in extra_lines if line.strip()}
    else:
        extra_lines = []
        extra_keys = set()

    def is_nstr() -> bool:
        if xmi_passthrough and xmi_passthrough.str_section_text is not None:
            return True
        return False

    # 如果用户提供了$str
    if is_nstr():
        str_body = xmi_passthrough.str_section_text
        length = len(str_body.splitlines())
        str_line= f"nstr={length}"
    else:
        str_line = f"str={xmidata.stru_type}"

    ctrl_lines = [
        f"{xmidata.method}",
        f"{str_line}",
        f"nao={xmidata.nao}",
        f"nae={xmidata.nae}",
        f"ncharge={xmidata.ncharge}",
        f"nmul={xmidata.nmul}",
        f"iscf={xmidata.iscf}",
        f"int={xmidata.int_type}",
        f"basis={xmidata.basis_set}",
    ]

    default_extra_lines = [
        "iprint=3",
        "orbtyp=hao",
        "frgtyp=atom",
        "itmax=2000",
        "molden",
        "output=aim",
    ]
    for line in default_extra_lines:
        if ctrl_key(line) not in extra_keys:
            ctrl_lines.append(line)

    ctrl_lines.extend(extra_lines)

    if xmidata.sort and "sort" not in extra_keys:
        ctrl_lines.append('sort')
    ctrl_text = "\n".join(ctrl_lines)
    
    xmi_text = f'''{xmidata.molecule_name} Created by autoVB {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
$ctrl
{ctrl_text}
$end

$orb
{xmidata.orb_section}
$end

$geo
{xmidata.geo_section}
$end
'''

    # 如果用户提供了$str
    if is_nstr():
        str_body = f'''
$str
{xmi_passthrough.str_section_text}
$end
'''
        xmi_text += str_body

    xmi_text += f'''
$gus
{xmidata.init_guess_section}
$end
'''
    with open(xmi_path, 'w') as f:
        f.write(xmi_text)


def write_xmi_with_orbital_guess(
    xmi_text: str,
    orb_text: str,
    output: str | Path,
    method: str | None = None,
    iscf: int | None = None,
) -> Path:
    """替换 XMI 的轨道初猜，并保留其余内容。

    Args:
        xmi_text: 原始 XMI 文件文本。
        orb_text: 新 ``.orb`` 文件文本，用于替换 ``$gus`` 主体。
        output: 新 XMI 文件路径。
        method: 可选的新计算方法，只替换 ``$ctrl`` 中原有的方法行。
        iscf: 可选的新 ISCF 值，只替换 ``$ctrl`` 中原有的 ``iscf``。

    Returns:
        写出的 XMI 文件路径。
    """
    output_text = _replace_xmi_section_body(xmi_text, "gus", orb_text)
    if method is not None:
        output_text = _replace_xmi_method(output_text, method)
    if iscf is not None:
        output_text = _replace_xmi_ctrl_option(output_text, "iscf", str(iscf))

    output_path = Path(output)
    output_path.write_bytes(output_text.encode("utf-8"))
    return output_path


def _replace_xmi_section_body(text: str, section: str, body: str) -> str:
    """替换指定 XMI section 的主体，保留 section 标记和其他文本。"""
    section_pattern = re.compile(
        rf"^(?P<header>[ \t]*\${re.escape(section)}[ \t]*\r?\n)"
        rf"(?P<body>.*?)"
        rf"(?P<footer>^[ \t]*\$end[ \t]*(?=\r?$))",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = section_pattern.search(text)
    if match is None:
        raise ValueError(f"Failed to find ${section} section in XMI text.")

    newline = "\r\n" if match.group("header").endswith("\r\n") else "\n"
    normalized_body = body.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    normalized_body = normalized_body.replace("\n", newline) + newline
    return (
        text[: match.start("body")]
        + normalized_body
        + text[match.end("body") :]
    )


def _replace_xmi_method(text: str, method: str) -> str:
    """只替换 XMI ``$ctrl`` 中的计算方法行。"""
    ctrl_pattern = re.compile(
        r"^(?P<header>[ \t]*\$ctrl[ \t]*\r?\n)"
        r"(?P<body>.*?)"
        r"(?P<footer>^[ \t]*\$end[ \t]*(?=\r?$))",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    ctrl_match = ctrl_pattern.search(text)
    if ctrl_match is None:
        raise ValueError("Failed to find $ctrl section in XMI text.")

    method_pattern = re.compile(
        rf"^(?P<indent>[ \t]*)(?:{'|'.join(map(re.escape, SUPPORTED_METHODS))})"
        r"(?P<trailing>[ \t]*)(?P<carriage>\r?)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    ctrl_body, replacement_count = method_pattern.subn(
        rf"\g<indent>{method}\g<trailing>\g<carriage>",
        ctrl_match.group("body"),
        count=1,
    )
    if replacement_count == 0:
        raise ValueError("Failed to find method line in XMI $ctrl section.")
    return (
        text[: ctrl_match.start("body")]
        + ctrl_body
        + text[ctrl_match.end("body") :]
    )


def _replace_xmi_ctrl_option(text: str, key: str, value: str) -> str:
    """替换 XMI ``$ctrl`` 中一个已有的 ``key=value`` 控制项。"""
    ctrl_pattern = re.compile(
        r"^(?P<header>[ \t]*\$ctrl[ \t]*\r?\n)"
        r"(?P<body>.*?)"
        r"(?P<footer>^[ \t]*\$end[ \t]*(?=\r?$))",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    ctrl_match = ctrl_pattern.search(text)
    if ctrl_match is None:
        raise ValueError("Failed to find $ctrl section in XMI text.")

    option_pattern = re.compile(
        rf"(?<!\S){re.escape(key)}\s*=\s*\S+",
        flags=re.IGNORECASE,
    )
    ctrl_body, replacement_count = option_pattern.subn(
        f"{key}={value}",
        ctrl_match.group("body"),
        count=1,
    )
    if replacement_count == 0:
        raise ValueError(f"Failed to find {key} in XMI $ctrl section.")
    return (
        text[: ctrl_match.start("body")]
        + ctrl_body
        + text[ctrl_match.end("body") :]
    )


def write_json_summary(
    input_data: 'autoVBInputData',
    parsed_data: 'XmoParsedData',
    output: str | Path | None = None,
) -> Path:
    """将 XMVB 解析结果写成便于后续处理的 JSON 摘要。

    Args:
        input_data: autoVB 输入数据，提供分子名称、电荷和自旋多重度。
        parsed_data: ``XmoParser`` 返回的 XMVB 解析结果。
        output: JSON 输出路径；不提供时使用 ``<分子名>.json``。

    Returns:
        写出的 JSON 文件路径。
    """
    charge = int(parsed_data.ctrl_options.get("ncharge", input_data.charge))
    multiplicity = int(parsed_data.ctrl_options.get("nmul", input_data.spin))

    summary = {
        "molecule": {
            "name": input_data.filename,
            "title": input_data.title,
            "charge": charge,
            "multiplicity": multiplicity,
            "geometry": [atom.to_dict() for atom in parsed_data.geo],
        },
        "orb": {
            "nae": parsed_data.nae,
            "nao": parsed_data.nao,
            "section": parsed_data.orb,
            "orbital_to_atom": parsed_data.orbital_to_atom,
        },
        "calculation": {
            "method": parsed_data.method,
            "basis": parsed_data.basis,
            "converged": parsed_data.converged,
            "steps": parsed_data.steps,
            "energy": parsed_data.energy,
            "energy_terms": parsed_data.energy_terms,
        },
        "structures": _merge_structure_weights(parsed_data),
    }

    output_path = Path(output) if output else Path(f"{input_data.filename}.json")
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _merge_structure_weights(parsed_data: 'XmoParsedData') -> list[dict[str, Any]]:
    """按结构序号合并 XMVB 输出中的四套结构权重。"""
    weight_tables: tuple[tuple[str, list['XmoStructureWeight']], ...] = (
        ("cc", parsed_data.cc_weights),
        ("lowdin", parsed_data.lowdin_weights),
        ("inverse", parsed_data.inverse_weights),
        ("renormalized", parsed_data.renormalized_weights),
    )
    structures: dict[int, dict[str, Any]] = {}

    for weight_name, rows in weight_tables:
        for row in rows:
            if row.index not in structures:
                structure = row.to_dict()
                structure.pop("weight")
                structure["structure"] = structure.pop("structure_name")
                structure["weights"] = {}
                structures[row.index] = structure
            structures[row.index]["weights"][weight_name] = row.weight

    return [structures[index] for index in sorted(structures)]

def write_gjf_nbo_file(mol: 'gto.Mole',filename: str, method: str='hf', mem: str='4GB', nproc: int=4):
    from ..utils.utils import pyscf_to_xyz
    geometry_text = pyscf_to_xyz(mol)
    gaussian_basis = to_gaussian_basis_name(mol.basis)
    filetext = f'''%chk={filename}.chk
%mem={mem}
%nprocshared={nproc}
#p {method}/{gaussian_basis} pop=nboread nosymm int(nobasistransform) 6D 10F scf(xqc,maxcycle=512)

{filename} generated by autoVB

{mol.charge} {mol.spin + 1}
{geometry_text}

$NBO plot file={filename} $END


'''
    with open(f'{filename}.gjf', 'w') as f:
        f.write(filetext)

def write_automr(
    mol: 'gto.Mole',
    filename: str,
    method: str = 'GVB',
    mem: str = '8GB',
    nproc: int = 4,
    mokit_keywords: str = '',
):
    '''
    写入适用于MOKIT automr的输入文件
    Args:
        mol (gto.Mole): pyscf的分子对象，包含分子结构、基组等信息
        filename (str): 输出文件名（不带扩展名）
        method (str): 计算方法，默认为GVB
        mem (str): 内存设置，默认为4GB
        nproc (int): 使用的CPU核心数，默认为4
        mokit_keywords (str): 额外的MOKIT关键词字符串，例如"GVB_prog=GAMESS"，默认为空
    '''
    from ..utils.utils import pyscf_to_xyz
    # automr要求内存至少为nproc的两倍，否则会阻止计算
    need_mem = 2 * int(nproc)
    if int(mem[:-2]) < need_mem:
        raise ValueError(f"GVB calculation requires at least {need_mem}GB memory(at least twice the number of processors). Please increase the memory allocation.")
    geometry_text = pyscf_to_xyz(mol)
    gaussian_basis = to_gaussian_basis_name(mol.basis)
    mokit_line = f"mokit{{{mokit_keywords}}}" if mokit_keywords else "mokit{}"
    filetext = f'''%mem={mem}
%nprocshared={nproc}
#p {method}/{gaussian_basis}

{mokit_line}

{mol.charge} {mol.spin + 1}
{geometry_text}


'''
    with open(f'{filename}.gjf', 'w') as f:
        f.write(filetext)
