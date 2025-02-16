import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

import datetime
import csv

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from torch import Tensor
from typing import List, Optional, Tuple
from tumorpde._typing import TensorLike

def resize_to_match(target, source):
    """
    Resizes the `source` array to match the shape of the `target` array.
    """
    zoom_factors = np.array(target.shape) / np.array(source.shape)
    return zoom(source, zoom_factors, order=1)

def read_patient_data(patient: str, test: bool = False, mask_ids: List[int] = [1,2], ref: int = 1):

    if test:
        zoom_factors = 0.05
    else:
        zoom_factors = 0.5

    dir_path = "../data/PatienTumorMultiScan2024/"
    brain, aff_info, header = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_brain_normalized.nii.gz'))
    raw_t1, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_t1_resized.nii.gz'))
    gm, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_gm_normalized.nii.gz'))
    wm, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_wm_normalized.nii.gz'))
    csf, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_csf_normalized.nii.gz'))
    tumor_list = []
    for i in mask_ids:
        tumor_list[i], _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}{i}_t1mask_resized.nii.gz'))

    # Ensure tumor2 matches tumor1 shape
    if len(tumor_list) > 1:
        for i in range(1,len(tumor_list)):
            if tumor_list[i].shape != tumor_list[0].shape:
                print(f"Scan {i+1} and Scan 1 have different shapes. Resizing tumor mask {i+1}.")
                tumor_list[i] = resize_to_match(tumor_list[0], tumor_list[i])

    # If image is too large
    if zoom_factors < 1.:
        zoom_factors = tuple([zoom_factors]*3)
        brain = zoom(brain, zoom_factors, order=1)
        raw_t1 = zoom(raw_t1, zoom_factors, order=1)
        gm = zoom(gm, zoom_factors, order=1)
        wm = zoom(wm, zoom_factors, order=1)
        csf = zoom(csf, zoom_factors, order=1)
        for i in range(len(tumor_list)):
            tumor_list[i] = zoom(tumor_list[i], zoom_factors, order=1)

    return {
        "patient": patient,
        "dir_path": dir_path,
        "brain": brain,
        "t1": raw_t1,
        "gm": gm,
        "wm": wm,
        "csf": csf,
        "tumor": tumor_list,
        "aff_info": aff_info,
        "header": header
    }


def append_parameters_to_file(file_path, patient, experiment_type, D, rho, x0, t1=None):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = [current_datetime, patient,
               experiment_type, D, rho, x0[0], x0[1], x0[2],
               t1 if t1 is not None else ""]

    # Read existing content
    existing_rows = []
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            existing_rows = list(reader)
    except FileNotFoundError:
        # If file doesn't exist, create it with a header
        existing_rows = [["Datetime", "Patient", "Experiment Type",
                          "D", "rho", "x0[0]", "x0[1]", "x0[2]", "t1"]]

    # Insert new row at the beginning (after header)
    existing_rows.insert(1, new_row)

    # Write updated content back to file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(existing_rows)

def ndarray_from_nifty(filename: str):

    nifti_file = filename
    img = nib.load(nifti_file)
    A = img.get_fdata()
    affine_info = img.affine
    header = img.header

    return A, affine_info, header


def weighted_center(u: NDArray) -> NDArray:
    N = u.shape
    axis_indices = np.indices(N)
    weighted_sum = np.array([np.sum(u * I) for I in axis_indices])
    total_weight = np.sum(u)
    return weighted_sum / total_weight


def _vis_brain_scan(
        u: TensorLike, brain: NDArray,
        tumor1: NDArray, tumor2: NDArray,
        figsize: Tuple[float, float],
        main_title: str, time_info: str = "",
        show: bool = True, file_prefix: str = "",
        save_dir: Optional[str] = None, idx: int = 0):

    fig, ax = plt.subplots(nrows=3, ncols=3,
                           figsize=(figsize[0]*3, figsize[1]*3))

    nrow = 4
    ncol = 4
    slice_fracs = np.linspace(0.2, 0.8, nrow * ncol).reshape((nrow,ncol))

    for j in range(ncol):
        for i in range(nrow):

            example_idx = int(brain.shape[j] * slice_fracs[j][i])
            sl = [slice(None)] * 3
            sl[2] = slice(example_idx, example_idx+1) # the axiel slice
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
    fig.tight_layout(rect=(0., 0., 1., 0.95))

    if save_dir is not None:
        fig.savefig(f"{save_dir}/{file_prefix}-i{idx}.jpg")

    if show:
        plt.show()

    plt.close(fig)


def visualize_model_fit(
        u: TensorLike, idx: int, t: float, brain: NDArray, tumor1: NDArray, tumor2: NDArray,
        figsize: Tuple[float, float] = (5, 5), show: bool = True, main_title: str = "Patient",
        file_prefix: str = "", save_dir: Optional[str] = None):

    if isinstance(u, Tensor):
        u = u.detach().cpu().numpy()
    u = np.asarray(u)

    brain = 0.8 * brain / brain.max()
    time_info = f"t={round(t, 3)}"

    _vis_brain_scan(u, brain, tumor1, tumor2, figsize, main_title, time_info,
                    show=show, file_prefix=file_prefix, save_dir=save_dir, idx=idx)


def visualize_model_fit_multiscan(
        u: TensorLike, idx: int, t: float, brain: NDArray, tumor1: NDArray, tumor2: NDArray,
        t_scan: List[float], real_t_diff: Optional[float] = None, time_unit: str = "day",
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

    _vis_brain_scan(u, brain, tumor1, tumor2, figsize, main_title, time_info,
                    show=show, file_prefix=file_prefix, save_dir=save_dir, idx=idx)
