import nibabel as nib
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from torch import Tensor
from tumorpde._typing import TensorLike

def ndarray_from_nifty(filename: str):

    nifti_file = filename
    img = nib.load(nifti_file)
    A = img.get_fdata()
    affine_info = img.affine
    header = img.header

    return A, affine_info, header


def weighted_center(u: ndarray) -> ndarray:
    N = u.shape
    axis_indices = np.indices(N)
    weighted_sum = np.array([np.sum(u * I) for I in axis_indices])
    total_weight = np.sum(u)
    return weighted_sum / total_weight


# def visualize_model_fit(
#         u, idx, t, brain, tumor1, tumor2,
#         slice_fracs=[[0.4,0.5,0.6], [0.4,0.5,0.6], [0.4,0.5,0.6]],
#         figsize=(5, 5), show=True, main_title="Patient", file_prefix="", save_dir=None):

#     if isinstance(u, Tensor):
#         u = u.detach().cpu().numpy()
#     u = np.asarray(u)

#     brain = 0.8 * brain / brain.max()

#     fig, ax = plt.subplots(nrows=3, ncols=3,
#                            figsize=(figsize[0]*3, figsize[1]*3))

#     for j in [2,1,0]:
#         for i in range(3):

#             example_idx = int(brain.shape[j] * slice_fracs[j][i])
#             sl = [slice(None)] * 3
#             sl[j] = slice(example_idx, example_idx+1)
#             sl = tuple(sl)
#             ax[j][i].imshow(brain[sl].squeeze(), cmap="gist_gray", vmin=0., vmax=1.)
#             masked_tumor = np.ma.masked_where(
#                 tumor1[sl].squeeze() < 0.5, tumor1[sl].squeeze())
#             ax[j][i].imshow(masked_tumor, cmap='Reds', alpha=0.3, vmin=0., vmax=1.)
#             masked_tumor = np.ma.masked_where(
#                 np.logical_or(tumor2[sl].squeeze() < 0.5,
#                                tumor1[sl].squeeze() >= 0.5),
#                 tumor2[sl].squeeze())
#             ax[j][i].imshow(masked_tumor, cmap='viridis',
#                             alpha=0.3, vmin=0., vmax=1.)
#             masked_u = np.ma.masked_where(
#                 u[sl].squeeze() < 1e-1, u[sl].squeeze())
#             norm_u = (masked_u - masked_u.min()) / (masked_u.max() -
#                                                     masked_u.min())  # Normalize to 0-1 range
#             alpha_channel = np.minimum(norm_u, 0.6)
#             # alpha_channel = np.minimum(masked_u, 0.4)
#             cmap = plt.get_cmap("Blues_r")
#             rgba_u = cmap(masked_u)  # Apply the colormap
#             rgba_u[..., -1] = alpha_channel  # Set the custom alpha channel
#             # alpha=1.0 since alpha is handled in rgba_u
#             ax[j][i].imshow(rgba_u, alpha=0.9, vmin=0., vmax=1.)
#             ax[j][i].set_title(f"Slice {slice_fracs[j][i]}")

#     # Add a main title to all plots
#     fig.suptitle(f"{main_title} t={round(t, 3)}", fontsize=16)

#     # Adjust layout to make room for the main title
#     # fig.tight_layout()
#     fig.tight_layout(rect=[0, 0, 1, 0.95])

#     if save_dir is not None:
#         fig.savefig(f"{save_dir}/{file_prefix}-i{idx}.jpg")

#     if show:
#         plt.show()
    
#     plt.close(fig)


# def visualize_model_fit_multiscan(
#         u, idx, t, brain, tumor1, tumor2,
#         t_scan: List[float], real_t_diff: Optional[float] = None, time_unit="day",
#         slice_fracs=[[0.4,0.5,0.6], [0.4,0.5,0.6], [0.4,0.5,0.6]],
#         figsize=(5, 5), show=True, main_title="Patient", file_prefix="", save_dir=None):

#     if isinstance(u, Tensor):
#         u = u.detach().cpu().numpy()
#     u = np.asarray(u)

#     if real_t_diff is None:
#         time_unit = ""
#         time_scale = 1.
#     else:
#         time_scale = real_t_diff / (t_scan[1] - t_scan[0])
#     ndigits = 3
#     if time_unit == "day":
#         ndigits = 1

#     brain = 0.8 * brain / brain.max()

#     fig, ax = plt.subplots(nrows=3, ncols=3,
#                            figsize=(figsize[0]*3, figsize[1]*3))

#     for j in [2,1,0]:
#         for i in range(3):

#             example_idx = int(brain.shape[j] * slice_fracs[j][i])
#             sl = [slice(None)] * 3
#             sl[j] = slice(example_idx, example_idx+1)
#             sl = tuple(sl)
#             ax[j][i].imshow(brain[sl].squeeze(), cmap="gist_gray", vmin=0., vmax=1.)
#             masked_tumor = np.ma.masked_where(
#                 tumor1[sl].squeeze() < 0.5, tumor1[sl].squeeze())
#             ax[j][i].imshow(masked_tumor, cmap='Reds', alpha=0.3, vmin=0., vmax=1.)
#             masked_tumor = np.ma.masked_where(
#                 np.logical_or(tumor2[sl].squeeze() < 0.5,
#                                tumor1[sl].squeeze() >= 0.5),
#                 tumor2[sl].squeeze())
#             ax[j][i].imshow(masked_tumor, cmap='viridis',
#                             alpha=0.3, vmin=0., vmax=1.)
#             masked_u = np.ma.masked_where(
#                 u[sl].squeeze() < 1e-1, u[sl].squeeze())
#             norm_u = (masked_u - masked_u.min()) / (masked_u.max() -
#                                                     masked_u.min())  # Normalize to 0-1 range
#             alpha_channel = np.minimum(norm_u, 0.6)
#             # alpha_channel = np.minimum(masked_u, 0.4)
#             cmap = plt.get_cmap("Blues_r")
#             rgba_u = cmap(masked_u)  # Apply the colormap
#             rgba_u[..., -1] = alpha_channel  # Set the custom alpha channel
#             # alpha=1.0 since alpha is handled in rgba_u
#             ax[j][i].imshow(rgba_u, alpha=0.9, vmin=0., vmax=1.)
#             ax[j][i].set_title(f"Slice {slice_fracs[j][i]}")

#     # Add a main title to all plots
#     fig.suptitle(f"{main_title} t={round(t * time_scale, ndigits)} {time_unit};  " + \
#                  f"t-t1={round((t - t_scan[0]) * time_scale, ndigits)} {time_unit};  " + \
#                  f"t-t2={round((t - t_scan[1]) * time_scale, ndigits)} {time_unit}", fontsize=16)

#     # Adjust layout to make room for the main title
#     # fig.tight_layout()
#     fig.tight_layout(rect=[0, 0, 1, 0.95])

#     if save_dir is not None:
#         fig.savefig(f"{save_dir}/{file_prefix}-i{idx}.jpg")

#     if show:
#         plt.show()
    
#     plt.close(fig)


def _vis_brain_scan(
        u: TensorLike, brain: ndarray, tumor1: ndarray, tumor2: ndarray,
        slice_fracs: List[List[float]], figsize: Tuple[float, float], main_title: str,
        time_info: str = "", show: bool = True, file_prefix: str = "", save_dir: Optional[str] = None, idx: int = 0):

    fig, ax = plt.subplots(nrows=3, ncols=3,
                           figsize=(figsize[0]*3, figsize[1]*3))

    for j in [2, 1, 0]:
        for i in range(3):

            example_idx = int(brain.shape[j] * slice_fracs[j][i])
            sl = [slice(None)] * 3
            sl[j] = slice(example_idx, example_idx+1)
            sl = tuple(sl)
            ax[j][i].imshow(brain[sl].squeeze(), cmap="gist_gray", vmin=0., vmax=1.)
            masked_tumor = np.ma.masked_where(
                tumor1[sl].squeeze() < 0.5, tumor1[sl].squeeze())
            ax[j][i].imshow(masked_tumor, cmap='Reds', alpha=0.3, vmin=0., vmax=1.)
            masked_tumor = np.ma.masked_where(
                np.logical_or(tumor2[sl].squeeze() < 0.5,
                               tumor1[sl].squeeze() >= 0.5),
                tumor2[sl].squeeze())
            ax[j][i].imshow(masked_tumor, cmap='viridis',
                            alpha=0.3, vmin=0., vmax=1.)
            masked_u = np.ma.masked_where(
                u[sl].squeeze() < 1e-1, u[sl].squeeze())
            norm_u = (masked_u - masked_u.min()) / (masked_u.max() -
                                                    masked_u.min())  # Normalize to 0-1 range
            alpha_channel = np.minimum(norm_u, 0.6)
            cmap = plt.get_cmap("Blues_r")
            rgba_u = cmap(masked_u)  # Apply the colormap
            rgba_u[..., -1] = alpha_channel  # Set the custom alpha channel
            ax[j][i].imshow(rgba_u, alpha=0.9, vmin=0., vmax=1.)
            ax[j][i].set_title(f"Slice {slice_fracs[j][i]}")

    # Add a main title to all plots
    fig.suptitle(f"{main_title} {time_info}", fontsize=20)

    # Adjust layout to make room for the main title
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir is not None:
        fig.savefig(f"{save_dir}/{file_prefix}-i{idx}.jpg")

    if show:
        plt.show()

    plt.close(fig)


def visualize_model_fit(
        u: TensorLike, idx: int, t: float, brain: ndarray, tumor1: ndarray, tumor2: ndarray,
        slice_fracs: List[List[float]] = [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],
        figsize: Tuple[float, float] = (5, 5), show: bool = True, main_title: str = "Patient",
        file_prefix: str = "", save_dir: Optional[str] = None):

    if isinstance(u, Tensor):
        u = u.detach().cpu().numpy()
    u = np.asarray(u)

    brain = 0.8 * brain / brain.max()
    time_info = f"t={round(t, 3)}"

    _vis_brain_scan(u, brain, tumor1, tumor2, slice_fracs, figsize, main_title, time_info,
                    show=show, file_prefix=file_prefix, save_dir=save_dir, idx=idx)


def visualize_model_fit_multiscan(
        u: TensorLike, idx: int, t: float, brain: ndarray, tumor1: ndarray, tumor2: ndarray,
        t_scan: List[float], real_t_diff: Optional[float] = None, time_unit: str = "day",
        slice_fracs: List[List[float]] = [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],
        figsize: Tuple[float, float] = (5, 5), show: bool = True, main_title: str = "Patient",
        file_prefix: str = "", save_dir: Optional[str] = None):

    if isinstance(u, Tensor):
        u = u.detach().cpu().numpy()
    u = np.asarray(u)

    if real_t_diff is None:
        time_unit = ""
        time_scale = 1.
    else:
        time_scale = real_t_diff / (t_scan[1] - t_scan[0])
    ndigits = 3
    if time_unit == "day":
        ndigits = 1

    brain = 0.8 * brain / brain.max()

    time_info = f"t={round(t * time_scale, ndigits)} {time_unit}; " + \
                f"t-t1={round((t - t_scan[0]) * time_scale, ndigits)} {time_unit}; " + \
                f"t-t2={round((t - t_scan[1]) * time_scale, ndigits)} {time_unit}"

    _vis_brain_scan(u, brain, tumor1, tumor2, slice_fracs, figsize, main_title, time_info,
                    show=show, file_prefix=file_prefix, save_dir=save_dir, idx=idx)

