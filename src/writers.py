from pathlib import Path
from typing import TYPE_CHECKING
import datetime

if TYPE_CHECKING:
    from .main import XMIData, XMIPassthrough

def write_xmi_file(filename: str, xmidata: 'XMIData', xmi_passthrough: 'XMIPassthrough'= None):
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

    if xmidata.method.lower() == 'bovb':
        print("BOVB method detected, only iscf=2 will be used regardless of user input.")
        xmidata.iscf = 2

    ctrl_lines = [
        f"{xmidata.method}",
        f"str={xmidata.stru_type}",
        f"nao={xmidata.nao}",
        f"nae={xmidata.nae}",
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
    if xmi_passthrough and xmi_passthrough.str_section_text is not None:
        str_body = xmi_passthrough.str_section_text
        xmi_text += f'''
$str
{str_body}
$end
'''

    xmi_text += f'''

$gus
{xmidata.init_guess_section}
$end
'''
    with open(xmi_path, 'w') as f:
        f.write(xmi_text)
