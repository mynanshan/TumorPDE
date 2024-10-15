import numpy as np
import itertools
from typing import List, Tuple, Literal


class VolumeDomain:

    """
    Define a volume with arbitrary shape. The shape is specificed by a density array
    """

    def __init__(self, voxel, voxel_widths):
        """
        Define a volume with arbitrary shape specified by a density array.

        Arguments:
            voxel: ndarray, voxel data representing the density.
            voxel_widths: sequence of floats, widths of the voxels in each dimension.
        """
        self.voxel = np.asarray(voxel)
        self.dim = self.voxel.ndim
        self.voxel_grad = np.gradient(self.voxel)
        self.voxel_widths = np.asarray(voxel_widths)
        self.voxel_volume = np.prod(self.voxel_widths)
        if len(self.voxel_widths) != self.dim:
            raise ValueError("voxel_widths does not match the dimension.")
        self.voxel_shape = self.voxel.shape
        # self.voxel_grad = np.zeros_like(self.voxel)
        if self.dim == 1:
            self.voxel_grad = [self.voxel_grad]
        self.xmin = np.array([0.] * self.dim)
        self.xmax = np.array(
            [w * n for w, n in zip(self.voxel_widths, self.voxel_shape)])
        self.bbox = (self.xmin, self.xmax)
        self.bbox_widths = self.xmax - self.xmin
        self.bbox_volume = np.prod(self.bbox_widths)
        self.voxel_marginal_coords = tuple(np.linspace(
            l + h/2, u - h/2, n) for l, u, n, h in
            zip(self.xmin, self.xmax, self.voxel_shape, self.voxel_widths))
        self.voxel_marginal_divides = tuple(np.linspace(
            l, u, n + 1) for l, u, n in zip(self.xmin, self.xmax, self.voxel_shape))
        self.boundary_voxels = self._get_boundary_locations()
        boundary_indices = np.argwhere(self.boundary_voxels)
        # self.boundary_anchors = np.hstack(
        #     [self.voxel_marginal_coords[i][boundary_indices[:, i:i+1]] for i in range(self.dim)])
        self.boundary_anchors = np.column_stack(
            [self.voxel_marginal_coords[i][boundary_indices[:, i]]
                for i in range(self.dim)]
        )

    def _get_voxel_indices(self, x):
        idx = tuple(np.searchsorted(
            self.voxel_marginal_divides[i], x[:, i], side='left') - 1 for i in range(self.dim))
        return idx

    def get_voxel_values(self, x):
        idx = self._get_voxel_indices(x)
        return self.voxel[idx]

    def get_voxel_grads(self, x):
        idx = self._get_voxel_indices(x)
        return [g[idx] for g in self.voxel_grad]

    def inside(self, x):
        ind = self.get_voxel_values(x) > 0.
        return ind

    def _get_boundary_locations(self):
        # Ensure X is a numpy array
        X = np.asarray(self.voxel)
        ndim = self.dim  # Get the number of dimensions of X
        # Step 1: Identify positions of nonzero elements
        nonzero_pos = X > 0
        # Step 2: Create a padded version of nonzero_pos to avoid boundary issues
        padded_nonzero = np.pad(nonzero_pos, pad_width=1,
                                mode='constant', constant_values=0)
        # Step 3: Create an empty array A of the same shape as padded_nonzero
        A = np.zeros_like(padded_nonzero, dtype=bool)
        # Dynamically create neighbors based on the number of dimensions
        for dim in range(ndim):
            for shift in [-1, 1]:
                index = tuple(
                    slice(1 + (shift if i == dim else 0),
                          padded_nonzero.shape[i] - 1 + (shift if i == dim else 0))
                    for i in range(ndim)
                )
                A[index] |= padded_nonzero[tuple(
                    slice(1, -1) for _ in range(ndim))]
        # Step 4: "Hollow out" A from X by selecting boundaries (nonzero padded, zero in original)
        boundary_slices = tuple(slice(1, -1) for _ in range(ndim))
        boundary = A[boundary_slices] & ~nonzero_pos
        return boundary

    def on_boundary(self, x):
        idx = self._get_voxel_indices(x)
        return self.boundary_voxels[idx]

    def uniform_points(self, n, boundary=True):
        dx = (self.bbox_volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.bbox_widths[i] / dx))
            if boundary:
                xi.append(
                    np.linspace(self.xmin[i], self.xmax[i], num=ni)
                )
            else:
                xi.append(
                    np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False
                    )[1:]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                f"Warning: {n} points required, but {len(x)} points sampled.")
        return x

    def uniform_interior_points(self, n, boundary=True, verbose=False):
        x = self.uniform_points(n, boundary)
        ind = self.inside(x)
        if verbose:
            print(f"{n} points expected, {sum(ind)} points sampled")
        return x[ind]

    def uniform_boundary_points(self, n, boundary=True, verbose=False):
        if verbose:
            print("Uniform boundary points not supported in VolumeDomain. " +
                  "Return random points by default.")
        # NOTE: other choices?
        return self.random_boundary_points(n, random="pseudo")

    def random_points(self, n, random="pseudo"):
        if random == "pseudo":
            x = np.random.rand(n, self.dim)
        else:
            raise ValueError(f"Unknown random type: {random}")
        return (self.xmax - self.xmin) * x + self.xmin

    def random_interior_points(self, n, random="pseudo", achieve_n=True,
                               verbose=False, n_step_max=100):
        n_step = min(n, n_step_max)
        x = self.random_points(n, random)
        ind = self.inside(x)
        if achieve_n and sum(ind) < n:
            count = sum(ind)
            xx = []
            while count < n:
                xtmp = self.random_points(n_step, random)
                indtmp = self.inside(xtmp)
                xx.append(xtmp[indtmp])
                count += sum(indtmp)
            xx.append(x[ind])
            x = np.vstack(xx)
            x = x[:n]
            return x
        if verbose:
            print(f"{n} points expected, {sum(ind)} points sampled")
        return x[ind]

    def random_boundary_points(self, n, random="pseudo"):
        # sample from boundary anchors
        idx = np.random.randint(self.boundary_anchors.shape[0], size=n)
        return self.boundary_anchors[idx]
