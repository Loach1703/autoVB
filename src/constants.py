import re

VERSION = "0.1.2-dev"
FLOAT_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$')
BASIS_FUNCTION_DICT = {
    'S': 'S',
    'X': 'PX',
    'Y': 'PY',
    'Z': 'PZ',
    'XX': 'DXX',
    'YY': 'DYY',
    'ZZ': 'DZZ',
    'XY': 'DXY',
    'XZ': 'DXZ',
    'YZ': 'DYZ',
    'XXX': 'FXXX',
    'XXY': 'FXXY',
    'XXZ': 'FXXZ',
    'YYX': 'FXYY',
    'XYZ': 'FXYZ',
    'ZZX': 'FXZZ',
    'YYY': 'FYYY',
    'YYZ': 'FYYZ',
    'ZZY': 'FYZZ',
    'ZZZ': 'FZZZ',
}
# 轨道替换规则字典，例如D_ORBITAL_3TO4的规则为，原本的第1号轨道需要替换为新的第3号轨道，原本的第2号轨道替换为新的第5号轨道
# 3对应的是XX YY ZZ XY XZ YZ，4对应的是XX XY XZ YY YZ ZZ
D_ORBITAL_3TO4 = {0: 0, 1: 3, 2: 5, 3: 1, 4: 2, 5: 4}
D_ORBITAL_4TO3 = {0: 0, 1: 3, 2: 4, 3: 1, 4: 5, 5: 2}
F_ORBITAL_3TO4 = {0: 0, 1: 6, 2: 9, 3: 1, 4: 2, 5: 3, 6: 7, 7: 5, 8: 8, 9: 4}
F_ORBITAL_4TO3 = {0: 0, 1: 3, 2: 4, 3: 5, 4: 9, 5: 7, 6: 1, 7: 6, 8: 8, 9: 2}

STRU_CHOICES = ["full", "cov", "ion()"]
SUPPORTED_METHODS = ["vbscf", "vbpt2", "lam-dfvb", "bovb", "blw"]