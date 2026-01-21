from pathlib import Path
from autoVB import GVBGI

input_file_name = Path("C10H8.gjf")
actorb = 1
actele = 1
allorb = 1

gi = GVBGI(input_file_name, actorb, actele, allorb)

gi.get_input_file()
gi.main_make_xmi()