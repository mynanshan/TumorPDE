import numpy as np
from tumorpde.calc.dist_boundary import signed_distance_and_nearest_index

mask = np.zeros((7,7), dtype=bool)
mask[2:5,2:5] = True

signed_d, idx = signed_distance_and_nearest_index(mask)

print("Mask:")
print(mask.astype(int))
print("\nSigned distance map:")
print(np.round(signed_d, 4))
print("\nNearest‑boundary linear indices:")
print(idx)
print("\nNearest‑boundary coords (row, col):")
coords = np.unravel_index(idx, mask.shape)
print(coords)
