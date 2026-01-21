from pathlib import Path
import numpy as np
from autoVB import read_nbo_orbital, array_to_orb, print_localized_orbitals_info, XMVBOrbSection

input_file_name = Path("C10H8.31")
orb_array_all: np.ndarray = read_nbo_orbital(input_file_name,166)
orb_array = orb_array_all[:-1]
occupation_numbers = orb_array_all[-1]
need_orb = orb_array[:34]
text = array_to_orb(need_orb, reorder_d=True, reorder_f=False, dxx_indices=[10, 25, 40, 55, 70, 85, 100, 115, 130, 145])
with open("nbo_orbital_test.orb", "w") as f:
    f.write(text)
atom_label = {1: {'atom': 'C', 'basis_functions': {1: 'S', 2: 'S', 3: 'PX', 4: 'PY', 5: 'PZ', 6: 'S', 7: 'PX', 8: 'PY', 9: 'PZ', 10: 'DXX', 11: 'DYY', 12: 'DZZ', 13: 'DXY', 14: 'DXZ', 15: 'DYZ'}}, 2: {'atom': 'C', 'basis_functions': {16: 'S', 17: 'S', 18: 'PX', 19: 'PY', 20: 'PZ', 21: 'S', 22: 'PX', 23: 'PY', 24: 'PZ', 25: 'DXX', 26: 'DYY', 27: 'DZZ', 28: 'DXY', 29: 'DXZ', 30: 'DYZ'}}, 3: {'atom': 'C', 'basis_functions': {31: 'S', 32: 'S', 33: 'PX', 34: 'PY', 35: 'PZ', 36: 'S', 37: 'PX', 38: 'PY', 39: 'PZ', 40: 'DXX', 41: 'DYY', 42: 'DZZ', 43: 'DXY', 44: 'DXZ', 45: 'DYZ'}}, 4: {'atom': 'C', 'basis_functions': {46: 'S', 47: 'S', 48: 'PX', 49: 'PY', 50: 'PZ', 51: 'S', 52: 'PX', 53: 'PY', 54: 'PZ', 55: 'DXX', 56: 'DYY', 57: 'DZZ', 58: 'DXY', 59: 'DXZ', 60: 'DYZ'}}, 5: {'atom': 'C', 'basis_functions': {61: 'S', 62: 'S', 63: 'PX', 64: 'PY', 65: 'PZ', 66: 'S', 67: 'PX', 68: 'PY', 69: 'PZ', 70: 'DXX', 71: 'DYY', 72: 'DZZ', 73: 'DXY', 74: 'DXZ', 75: 'DYZ'}}, 6: {'atom': 'C', 'basis_functions': {76: 'S', 77: 'S', 78: 'PX', 79: 'PY', 80: 'PZ', 81: 'S', 82: 'PX', 83: 'PY', 84: 'PZ', 85: 'DXX', 86: 'DYY', 87: 'DZZ', 88: 'DXY', 89: 'DXZ', 90: 'DYZ'}}, 7: {'atom': 'C', 'basis_functions': {91: 'S', 92: 'S', 93: 'PX', 94: 'PY', 95: 'PZ', 96: 'S', 97: 'PX', 98: 'PY', 99: 'PZ', 100: 'DXX', 101: 'DYY', 102: 'DZZ', 103: 'DXY', 104: 'DXZ', 105: 'DYZ'}}, 8: {'atom': 'C', 'basis_functions': {106: 'S', 107: 'S', 108: 'PX', 109: 'PY', 110: 'PZ', 111: 'S', 112: 'PX', 113: 'PY', 114: 'PZ', 115: 'DXX', 116: 'DYY', 117: 'DZZ', 118: 'DXY', 119: 'DXZ', 120: 'DYZ'}}, 9: {'atom': 'C', 'basis_functions': {121: 'S', 122: 'S', 123: 'PX', 124: 'PY', 125: 'PZ', 126: 'S', 127: 'PX', 128: 'PY', 129: 'PZ', 130: 'DXX', 131: 'DYY', 132: 'DZZ', 133: 'DXY', 134: 'DXZ', 135: 'DYZ'}}, 10: {'atom': 'C', 'basis_functions': {136: 'S', 137: 'S', 138: 'PX', 139: 'PY', 140: 'PZ', 141: 'S', 142: 'PX', 143: 'PY', 144: 'PZ', 145: 'DXX', 146: 'DYY', 147: 'DZZ', 148: 'DXY', 149: 'DXZ', 150: 'DYZ'}}, 11: {'atom': 'H', 'basis_functions': {151: 'S', 152: 'S'}}, 12: {'atom': 'H', 'basis_functions': {153: 'S', 154: 'S'}}, 13: {'atom': 'H', 'basis_functions': {155: 'S', 156: 'S'}}, 14: {'atom': 'H', 'basis_functions': {157: 'S', 158: 'S'}}, 15: {'atom': 'H', 'basis_functions': {159: 'S', 160: 'S'}}, 16: {'atom': 'H', 'basis_functions': {161: 'S', 162: 'S'}}, 17: {'atom': 'H', 'basis_functions': {163: 'S', 164: 'S'}}, 18: {'atom': 'H', 'basis_functions': {165: 'S', 166: 'S'}}}

# 根据NBO占据数判断活性轨道
actorb = 10
actele = 10
half_orb = int(actorb / 2)

not_greater_than_one = occupation_numbers <= 1
if not np.any(not_greater_than_one):
    # 如果数组里全都是大于 1 的，则考虑整个数组
    break_idx = len(occupation_numbers)
else:
    break_idx = np.argmax(not_greater_than_one)
relevant_values = occupation_numbers[:break_idx]

# 活性轨道数量
actual_half_orb = min(half_orb, len(relevant_values))
# 活性轨道索引
actorb_indices = np.argsort(relevant_values)[:actual_half_orb]
# print(actorb_indices)
# 1. 切片出选中的项
active_orbital_matrix = need_orb[actorb_indices]

# 2. 获取剩下的项
inactive_orbital_matrix = np.delete(need_orb, actorb_indices, axis=0)

print(f"原数组形状: {need_orb.shape}")
print(f"选中项形状: {active_orbital_matrix.shape}")
print(f"剩余项形状: {inactive_orbital_matrix.shape}")
with open("nbo_active_orbital.orb", "w") as f:
    duplicated_data = np.repeat(active_orbital_matrix, repeats=2, axis=0)
    text = array_to_orb(duplicated_data, reorder_d=True, reorder_f=False, dxx_indices=[10, 25, 40, 55, 70, 85, 100, 115, 130, 145])
    f.write(text)
with open("nbo_inactive_orbital.orb", "w") as f:
    text = array_to_orb(inactive_orbital_matrix, reorder_d=True, reorder_f=False, dxx_indices=[10, 25, 40, 55, 70, 85, 100, 115, 130, 145])
    f.write(text)
xo = XMVBOrbSection()
a = xo.get_xmi_orb_section_hao(inactive_orbital_matrix, active_orbital_matrix, actorb_index=[1,2,3,4,5,6,7,8,9,10], actele=actele, actorb=actorb)
print(a)
for orb in active_orbital_matrix:
    j = print_localized_orbitals_info(1, orb, atom_label, need_print=True, need_atom_number=2)