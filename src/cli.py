#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

from .commands import autovb_nbo_impl, autovb_xmi_impl
from .main import autoVBMain, autoVBInputData, VBSettings
from .readers import autoVBInputParser, GaussianNBOParser
from .utils import generate_fch_from_chk, pyscf_to_xyz, print_warning, print_subroutine
from .constants import VERSION
from mokit.lib.gaussian import load_mol_from_fch

def autovb_nbo(argv=None):
    parser = argparse.ArgumentParser(prog="autovb-nbo", description="Generate Gaussian NBO .gjf from .xyz")
    parser.add_argument("xyz", type=Path, help="input .xyz file")
    parser.add_argument("basis", help="basis set for gjf")
    parser.add_argument("-c", "--charge", type=int, default=0)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--spin", type=int, default=0)
    group.add_argument("-m", "--multiplicity", type=int, default=1)
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

    return autovb_nbo_impl(xyz, basis, charge, final_spin)

def autovb_xmi(argv=None):
    parser = argparse.ArgumentParser(prog="autovb-xmi", description="Generate XMVB .xmi from .fch/.chk or basename")
    parser.add_argument("file", type=Path)
    parser.add_argument("basis", help="basis set override for xmi header")
    parser.add_argument("--threshold", "-t", type=float, default=1.8)
    parser.add_argument("--active_electron", "-nae", type=int, default=0)
    parser.add_argument("--active_orbital", "-nao", type=int, default=0)
    parser.add_argument("--active_orbital_atom", "-aoa", nargs='*', type=int, default=[])
    parser.add_argument('--reorder', '-r', action='store_true')
    parser.add_argument('--slice', '-s', action='store_true')
    args = parser.parse_args(argv)

    p: Path = args.file
    fchname = p.with_suffix(".fch")
    if p.suffix != ".fch":
        chkname = p.with_suffix(".chk")
        generate_fch_from_chk(chkname, fchname)

    mol = load_mol_from_fch(fchname)
    name = p.stem
    aoa = args.active_orbital_atom

    def split_into_chunks(lst, n):
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    if args.active_orbital_atom:
        aoa = split_into_chunks(args.active_orbital_atom, 2)

    nae = args.active_electron
    nao = args.active_orbital
    threshold = args.threshold if hasattr(args, 'threshold') else 1.9
    geometry = pyscf_to_xyz(mol)
    atvb_input = autoVBInputData(
        title=name,
        filepath=p,
        filename=p.stem,
        method='vbscf',
        basis=args.basis,
        geometry=geometry,
    )
    vbsetting = VBSettings(
        nae=nae,
        nao=nao,
        aoa=aoa,
        threshold=threshold,
        reorder=args.reorder,
        atom_slice=args.slice,
    )
    atvb_input.vbsettings = vbsetting

    return autovb_xmi_impl(name, mol, args.basis, nae, nao, aoa, threshold, args.reorder, args.slice)

def autovb_main(argv=None):
    print(f"Welcome to autoVB! Version {VERSION}")

    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv:
        print("Usage: autovb <input-file>", file=sys.stderr)
        return 2

    input_file = Path(argv[0])
    resolved = input_file.resolve()
    if not resolved.exists():
        print(f"Error: input file not found: {input_file}", file=sys.stderr)
        return 2
    parser = autoVBInputParser(input_file)

    # 命令行参数优先级最高，其次是输入文件参数，最后是默认值
    mem = parser.input_data.mem if parser.input_data.mem else "4GB"
    nproc = parser.input_data.nproc if parser.input_data.nproc else "1"
    mem = argv[1] if len(argv) > 1 else "4GB"
    nproc = argv[2] if len(argv) > 2 else "1"
    # G to GB
    if mem.lower().endswith("g"):
        mem = mem[:-1] + "GB"
    if mem.lower().endswith("m"):
        mem = mem[:-1] + "MB"
    # 纯数字默认单位为MB
    if mem.isdigit():
        mem = mem + "MB"
    print(f"Using memory: {mem}, nproc: {nproc}")
    parser.input_data.mem = mem
    parser.input_data.nproc = nproc

    main_obj = autoVBMain(parser.input_data)
    main_obj.main()

def autovb_test():
    input_file = Path('C2B2Me2_nbo.out')
    nbo_orb_file = Path('C2B2ME2_NBO.37')
    mol = load_mol_from_fch('C2B2Me2_nbo.fch')
    gnp = GaussianNBOParser(input_file, nbo_orb_file, mol)
    for i in gnp.nbo_data:
        print(i.connection, i.orbital_type, i.occupancy)