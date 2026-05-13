from pathlib import Path

from mokit.lib.gaussian import load_mol_from_fch

from ..readers import GaussianNBOParser


def autovb_test():
    input_file = Path('C2B2Me2_nbo.out')
    nbo_orb_file = Path('C2B2ME2_NBO.37')
    mol = load_mol_from_fch('C2B2Me2_nbo.fch')
    gnp = GaussianNBOParser(input_file, nbo_orb_file, mol)
    for i in gnp.nbo_data:
        print(i.connection, i.orbital_type, i.occupancy)
