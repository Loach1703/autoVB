import argparse
from pathlib import Path
import sys

from pyscf import gto


def autovb_nbo_impl(xyz: Path, basis: str, charge: int, spin: int) -> int:
    from ..writers import write_gjf_nbo_file
    mol = gto.M(
        atom=str(xyz),
        basis=basis,
        charge=charge,
        spin=spin,
    )
    name = xyz.stem
    write_gjf_nbo_file(mol, name, mem='4GB', nproc=4)
    print(f"Wrote Gaussian NBO input file to {name}.gjf")
    print(f'You need to manually verify if the charge and spin multiplicity are correct.')
    return 0


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
