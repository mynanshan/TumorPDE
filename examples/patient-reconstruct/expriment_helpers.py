import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage import measure

import datetime
import csv

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from torch import Tensor
from typing import List, Optional, Tuple
from tumorpde._typing import TensorLike


def ndarray_from_nifty(filename: str):

    nifti_file = filename
    img = nib.load(nifti_file)
    A = img.get_fdata()
    affine_info = img.affine
    header = img.header

    return A, affine_info, header

def read_patient_data(patient: str, test: bool = False, mask_ids: List[int] = [1,2], ref: int = 1):

    if test:
        zoom_factors = 0.05
    else:
        zoom_factors = 1.

    dir_path = "../data/PatienTumorMultiScan2024/"
    brain, aff_info, header = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_brain_normalized.nii.gz'))
    raw_t1, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_t1_resized.nii.gz'))
    gm, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_gm_normalized.nii.gz'))
    wm, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_wm_normalized.nii.gz'))
    csf_in, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_csf_in_normalized.nii.gz'))
    csf_out, _, _ = ndarray_from_nifty(os.path.join(
        dir_path, patient, f'{patient}{ref}_csf_out_normalized.nii.gz'))
    tumor_list = []
    for ii in mask_ids:
        tumor, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}{ii}_t1mask_resized.nii.gz'))
        tumor = binarize_img(tumor)
        tumor_list.append(tumor)

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
        csf_in = zoom(csf_in, zoom_factors, order=1)
        csf_out = zoom(csf_out, zoom_factors, order=1)
        for i in range(len(tumor_list)):
            tumor_list[i] = zoom(tumor_list[i], zoom_factors, order=1)
    
    print("Read data:")
    print(f"Patient brain, shape: {brain.shape}")
    print(f"Patient raw scan, shape: {raw_t1.shape}")
    print(f"Grey matter, shape: {gm.shape}")
    print(f"White matter, shape: {wm.shape}")
    for i in range(len(tumor_list)):
        print(f"Tumor mask {i+1}, shape: {tumor_list[i].shape}")

    return {
        "patient": patient,
        "dir_path": dir_path,
        "brain": brain,
        "t1": raw_t1,
        "gm": gm,
        "wm": wm,
        "csf_in": csf_in,
        "csf_out": csf_out,
        "tumor": tumor_list,
        "aff_info": aff_info,
        "header": header
    }

def resize_to_match(target, source):
    """
    Resizes the `source` array to match the shape of the `target` array.
    """
    zoom_factors = np.array(target.shape) / np.array(source.shape)
    return zoom(source, zoom_factors, order=1)

def binarize_img(img: NDArray, thresh: float = 0.5) -> NDArray:
    
    img = (img - img.min()) / (img.max() - img.min())
    img[img > thresh] = 1.
    img[img <= thresh] = 0.
    return img

def weighted_center(u: NDArray) -> NDArray:
    N = u.shape
    axis_indices = np.indices(N)
    weighted_sum = np.array([np.sum(u * I) for I in axis_indices])
    total_weight = np.sum(u)
    return weighted_sum / total_weight


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


def _vis_brain_scan(
        u: NDArray,
        underlays: List[NDArray],
        tumors: List[NDArray],
        figsize: Tuple[float, float],
        main_title: str = "",
        time_info: str = "",
        show: bool = True,
        file_prefix: List[str] = [""],
        save_dir: Optional[str] = None, idx: int = 0):
    
    nrow = 4
    ncol = 4
    slice_fracs = np.linspace(0.2, 0.8, nrow * ncol).reshape((nrow,ncol))
    tumor_colors = ["yellow", (0.2,1.,0.2,1.), (0.2,0.2,1.,1.)]
    
    for img, fpre in zip(underlays, file_prefix):

        img = 0.8 * img / img.max()

        fig, ax = plt.subplots(
            nrows=nrow, ncols=ncol,
            figsize=(figsize[0]*nrow, figsize[1]*ncol))

        for j in range(ncol):
            for i in range(nrow):

                # get the current slices
                example_idx = int(img.shape[2] * slice_fracs[j][i])
                sl = [slice(None)] * 3
                sl[2] = slice(example_idx, example_idx+1) # the axiel slice
                sl = tuple(sl)

                # plot the underlay
                img_sl = img[sl].squeeze()
                ax[j][i].imshow(img_sl.T, cmap="gist_gray", vmin=0., vmax=1.)

                for k, tumor in enumerate(tumors):
                    tumor_sl = tumor[sl].squeeze()

                    # get contour of the tumor
                    tumor_contours = measure.find_contours(tumor_sl, level=0.5)

                    # plot tumor contours
                    for contour in tumor_contours:
                        ax[j][i].plot(contour[:, 0], contour[:, 1],
                                      color=tumor_colors[k], linewidth=1.1)
                
                # plot simulated tumor density
                u_sl = u[sl].squeeze()
                u_sl = np.ma.masked_where(u_sl < 1e-4, u_sl)
                rgba_u = np.zeros((u_sl.T.shape[0], u_sl.T.shape[1], 4))
                rgba_u[..., 0] = 1.0  # red
                rgba_u[..., 3] = np.sqrt(u_sl.T) * 0.3  # transparency 
                ax[j][i].imshow(rgba_u, vmin=0., vmax=1.)

                # plot thresholded contour
                u_contours = measure.find_contours(u_sl, level=0.3)
                for contour in u_contours:
                    ax[j][i].plot(contour[:, 0], contour[:, 1], color='red', linewidth=1.1)

                ax[j][i].invert_yaxis()  # invert y-axis to match image coordinate system (origin at top-left)
                ax[j][i].set_title(f"Slice {slice_fracs[j][i]}")

        # Add a main title to all plots
        fig.suptitle(f"{main_title} {time_info}", fontsize=20)

        # Adjust layout to make room for the main title
        fig.tight_layout(rect=(0., 0., 1., 0.95))

        if save_dir is not None:
            fig.savefig(f"{save_dir}/{fpre}-i{idx}.jpg")

        if show:
            plt.show()

        plt.close(fig)


def visualize_model_fit(
        u: TensorLike, idx: int, t: float,
        underlays: NDArray | List[NDArray], tumors: NDArray | List[NDArray],
        figsize: Tuple[float, float] = (5, 5), show: bool = True,
        main_title: str = "Patient",
        file_prefix: str | List[str] = "",
        save_dir: Optional[str] = None):

    if isinstance(u, Tensor):
        u = u.detach().cpu().numpy()
    u = np.asarray(u)
    if not isinstance(underlays, list):
        underlays = [underlays]
    if not isinstance(tumors, list):
        tumors = [tumors]
    if not isinstance(file_prefix, list):
        file_prefix = [file_prefix]
    if len(underlays) > 1 and len(file_prefix) == 1:
        file_prefix = [str(file_prefix) + str(i+1) for i in range(len(underlays))]
    elif len(underlays) != len(file_prefix):
        raise ValueError("Number of underlays mush match the number of file name prefixs.")
    
    time_info = f"t={round(t, 3)}"

    _vis_brain_scan(u, underlays, tumors, figsize, main_title, time_info,
                    show=show, file_prefix=file_prefix, save_dir=save_dir, idx=idx)


def visualize_model_fit_multiscan(
        u: TensorLike, idx: int, t: float,
        underlays: NDArray | List[NDArray], tumors: NDArray | List[NDArray],
        t_scan: List[float], real_t_diff: Optional[float] = None, time_unit: str = "day",
        figsize: Tuple[float, float] = (5, 5), show: bool = True, main_title: str = "Patient",
        file_prefix: str | List[str] = "",
        save_dir: Optional[str] = None):

    if isinstance(u, Tensor):
        u = u.detach().cpu().numpy()
    u = np.asarray(u)
    if not isinstance(underlays, list):
        underlays = [underlays]
    if not isinstance(tumors, list):
        tumors = [tumors]
    if not isinstance(file_prefix, list):
        file_prefix = [file_prefix]
    if len(underlays) > 1 and len(file_prefix) == 1:
        file_prefix = [str(file_prefix) + str(i+1) for i in range(len(underlays))]
    elif len(underlays) != len(file_prefix):
        raise ValueError("Number of underlays mush match the number of file name prefixs.")

    if real_t_diff is None:
        time_unit = ""
        time_scale = 1.
    else:
        time_scale = real_t_diff / (t_scan[1] - t_scan[0])
    ndigits = 3
    if time_unit == "day":
        ndigits = 1

    time_info = f"t={round(t * time_scale, ndigits)} {time_unit}; " + \
                f"t-t1={round((t - t_scan[0]) * time_scale, ndigits)} {time_unit}; " + \
                f"t-t2={round((t - t_scan[1]) * time_scale, ndigits)} {time_unit}"

    _vis_brain_scan(u, underlays, tumors, figsize, main_title, time_info,
                    show=show, file_prefix=file_prefix, save_dir=save_dir, idx=idx)
