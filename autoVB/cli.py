#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from autoVB import GaussianNBO, XMVBNBO, generate_fch_from_chk
from mokit.lib.gaussian import load_mol_from_fch
from pyscf import gto

def autovb_nbo(argv=None):
    """命令：autovb-nbo <molecule.xyz> [--basis BASIS]"""
    parser = argparse.ArgumentParser(prog="autovb-nbo", description="Generate Gaussian NBO .gjf from .xyz")
    parser.add_argument("xyz", type=Path, help="input .xyz file")
    parser.add_argument("basis", help="basis set for gjf")
    parser.add_argument("-c", "--charge", type=int, default=0, help="number of molecule charges (default 0)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--spin", type=int, default=0, help="spin (2S) (default 0)")
    group.add_argument("-m", "--multiplicity", type=int, default=1, help="spin multiplicity (2S+1) (default 1)")
    args = parser.parse_args(argv)

    xyz: Path = args.xyz
    if not xyz.exists():
        print(f"Error: {xyz} not found", file=sys.stderr)
        return 2
    basis = args.basis
    if not basis:
        print("Error: basis set is required", file=sys.stderr)
        return 2
    charge = args.charge
    # Gaussian 的自旋多重度是 2S+1，而 pyscf 是 2S，所以需要转换一下。这里允许用户直接输入 spin 或 multiplicity，互斥选项，默认 spin=0 (singlet)
    final_spin = 0
    if args.multiplicity:
        final_spin = args.multiplicity - 1
    elif args.spin:
        final_spin = args.spin
    else:
        final_spin = 0
    if final_spin < 0:
        print(f"Error: invalid spin multiplicity {args.spin}", file=sys.stderr)
        return 2
    
    # 用 pyscf 从 xyz 构建分子对象
    mol = gto.M(
        atom=str(xyz), 
        basis=basis,
        charge = charge,
        spin = final_spin,
    )
    name = xyz.stem
    gn = GaussianNBO(name, mol)
    gn.write_gjf()
    return 0

def autovb_xmi(argv=None):
    """命令：autovb-xmi <file.fch|file.chk|basename> [--active_eletron N] [--active_orbital M] [--active_orbital_atom BASIS]"""
    parser = argparse.ArgumentParser(prog="autovb-xmi", description="Generate XMVB .xmi from .fch/.chk or basename")
    parser.add_argument("file", type=Path, help=".fch file, .chk file or basename")
    parser.add_argument("basis", help="basis set override for xmi header")
    parser.add_argument("--active_eletron", "-nae", type=int, default=0, help="active electrons (default 0, meaning auto-detect)")
    parser.add_argument("--active_orbital", "-nao", type=int, default=0, help="active orbitals (default 0 meaning auto-detect)")
    parser.add_argument("--active_orbital_atom", "-aoa", nargs='*', type=int, default=[], help="active orbital atom list, e.g. [[4,13],[9,12]] (default None)")
    parser.add_argument('--reorder', '-r', action='store_true', help='whether reorder orbitals by its index (default False)')
    parser.add_argument('--slice', '-s', action='store_true', help='whether slice the active orbitals from molecule orbital to atom orbital (default False)')
    args = parser.parse_args(argv)

    p: Path = args.file
    fchname = p.with_suffix(".fch")
    # 输入chk文件
    if p.suffix != ".fch":
        chkname = p.with_suffix(".chk")
        generate_fch_from_chk(chkname, fchname)

    mol = load_mol_from_fch(fchname)
    name = p.stem
    wxp = XMVBNBO(name, mol)
    aoa = args.active_orbital_atom

    def split_into_chunks(lst, n):
        """将列表 lst 每 n 个元素切分成一个子列表"""
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    if args.active_orbital_atom:
        aoa = split_into_chunks(args.active_orbital_atom, 2)

    # 设置活性空间
    nae = args.active_eletron
    nao = args.active_orbital
    wxp.set_basis_set(args.basis)
    # 设置了aoa
    if aoa and nae > 0 and nao > 0:
        wxp.set_active_space(nae, nao)
        wxp.set_active_orbital_atom(aoa)
    elif nae > 0 and nao > 0:
        wxp.set_active_space(nae, nao)
    else:
        nae, nao = wxp.auto_select_active_space(auto_set=True)
    inact, act = wxp.split_inactive_active_orbitals()
    wxp.write_xmi(inact, act, reorder=args.reorder, atom_slice=args.slice)
    return 0

if __name__ == "__main__":
    pass