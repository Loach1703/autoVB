from pathlib import Path

from mokit.lib.gaussian import load_mol_from_fch

from ..io.readers import GaussianNBOParser


def autovb_test():
    from mokit.lib.fch2py import fch2py
    from mokit.lib.rwwfn import read_nbf_and_nif_from_fch, read_eigenvalues_from_fch
    
