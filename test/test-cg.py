import torch
from tumorpde.calc.linalg import CG


A = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 2.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 2.0, -1.0],
                  [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]])

b = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

sol1, info = CG(A.to_sparse_csr(), b)
print(f'Solution by CG: {sol1}')

sol2 = torch.linalg.solve(A, b)
print(f'Solution by torch.linalg.solve: {sol2}')
