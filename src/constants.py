import re

VERSION = "0.1.2"
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

GAUSSIAN_BASIS_NAME_MAP = {
    "def2sv(p)": "def2SV(P)",
    "def2svp(p)": "def2SV(P)",
    "def2svp": "def2SVP",
    "def2svpd": "def2SVPD",
    "def2tzvp": "def2TZVP",
    "def2tzvpp": "def2TZVPP",
    "def2tzvpd": "def2TZVPD",
    "def2tzvppd": "def2TZVPPD",
    "def2qzvp": "def2QZVP",
    "def2qzvpp": "def2QZVPP",
    "def2qzvpd": "def2QZVPD",
    "def2qzvppd": "def2QZVPPD",
    "ccpvdz": "cc-pVDZ",
    "ccpvtz": "cc-pVTZ",
    "ccpvqz": "cc-pVQZ",
    "ccpv5z": "cc-pV5Z",
    "augccpvdz": "aug-cc-pVDZ",
    "augccpvtz": "aug-cc-pVTZ",
    "augccpvqz": "aug-cc-pVQZ",
    "augccpv5z": "aug-cc-pV5Z",
}


def basis_name_key(basis_name: str) -> str:
    return re.sub(r"[\s_-]+", "", basis_name).lower()


def to_gaussian_basis_name(basis_name):
    if not isinstance(basis_name, str):
        return basis_name
    return GAUSSIAN_BASIS_NAME_MAP.get(basis_name_key(basis_name), basis_name)


STRU_CHOICES = ["full", "cov", "ion()"]
SUPPORTED_METHODS = ["vbscf", "vbpt2", "lam-dfvb", "bovb", "blw", 'tbvbscf']
