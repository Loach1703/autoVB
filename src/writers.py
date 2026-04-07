from pathlib import Path
from typing import TYPE_CHECKING
import datetime

if TYPE_CHECKING:
    from .main import XMIData

def write_xmi_file(filename: str, xmidata: 'XMIData'):
    '''
    将 XMIData 数据格式写入 .xmi 文件。
    Args:
        filename (str): 输出文件名
        xmidata (XMIData): 包含轨道数据和相关信息的对象
    '''
    xmi_path = Path(filename).with_suffix('.xmi')
    ctrl_text = f"""{xmidata.method}
str={xmidata.stru_type}
nao={xmidata.nao}
nae={xmidata.nae}
iscf={xmidata.iscf}
iprint=3
orbtyp=hao
frgtyp=atom
int={xmidata.int_type}
basis={xmidata.basis_set}
itmax=2000
molden
output=aim"""
    
    if xmidata.sort:
        ctrl_text += '\nsort'

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

$gus
{xmidata.init_guess_section}
$end
'''
    with open(xmi_path, 'w') as f:
        f.write(xmi_text)