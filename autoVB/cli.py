#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from autoVB import GaussianNBO, XMVBNBO, XMVBNBOMain
from mokit.lib.gaussian import load_mol_from_fch
from pyscf import gto

def autovb_nbo(argv=None):
    """命令：autovb-nbo <molecule.xyz> [--basis BASIS]"""
    parser = argparse.ArgumentParser(prog="autovb-nbo", description="Generate Gaussian NBO .gjf from .xyz")
    parser.add_argument("xyz", type=Path, help="input .xyz file")
    parser.add_argument("--basis", default="cc-pvdz", help="basis set for gjf (default: cc-pvdz)")
    args = parser.parse_args(argv)

    xyz = args.xyz
    if not xyz.exists():
        print(f"Error: {xyz} not found", file=sys.stderr)
        return 2

    # 用 pyscf 从 xyz 构建分子对象
    mol = gto.M(atom=str(xyz), basis=args.basis)
    name = xyz.stem
    gn = GaussianNBO(name, mol)
    gn.write_gjf()
    return 0

def autovb_xmi(argv=None):
    """命令：autovb-xmi <file.fch|file.chk|basename> [--active N] [--electrons M] [--basis BASIS]"""
    parser = argparse.ArgumentParser(prog="autovb-xmi", description="Generate XMVB .xmi from .fch/.chk or basename")
    parser.add_argument("file", type=Path, help=".fch file, .chk file or basename")
    parser.add_argument("--active", "-a", type=int, default=8, help="active orbitals (default 8)")
    parser.add_argument("--electrons", "-e", type=int, default=8, help="active electrons (default 8)")
    parser.add_argument("--basis", default=None, help="basis set override for xmi header")
    args = parser.parse_args(argv)

    p = args.file
    # case: provided actual .fch
    if p.exists() and p.suffix == ".fch":
        mol = load_mol_from_fch(p)
        name = p.stem
        wxp = XMVBNBO(name, mol)
    else:
        # try basename or .chk
        base = p.with_suffix("").name
        wxp = XMVBNBOMain(base)

    wxp.set_active_space(args.active, args.electrons)
    if args.basis:
        wxp.set_basis_set(args.basis)
    wxp.split_inactive_active_orbitals()  # optional but safe
    inact, act = wxp.split_inactive_active_orbitals()
    wxp.write_xmi(inact, act, reorder=False, atom_slice=False)
    return 0

# 供直接脚本调用（开发测试）
if __name__ == "__main__":
    # 简单分发：根据脚本名决定
    prog = Path(sys.argv[0]).stem
    if prog.endswith("autovb-nbo"):
        sys.exit(autovb_nbo(sys.argv[1:]))
    elif prog.endswith("autovb-xmi"):
        sys.exit(autovb_xmi(sys.argv[1:]))
    else:
        # 默认进入 xmi
        sys.exit(autovb_xmi(sys.argv[1:]))