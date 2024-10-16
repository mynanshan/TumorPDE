import os
from scan_utils import ndarray_from_nifty
from scipy.ndimage import zoom

import datetime
import csv


def read_patient_data(patient: str, test: bool = False):

    if patient == "STT":
        zoom_factors = 0.7
        visualize_pos = [[0.4, 0.45, 0.5], [
            0.55, 0.6, 0.65], [0.65, 0.73, 0.81]]

    elif patient == "HR":
        zoom_factors = 1.
        visualize_pos = [[0.35, 0.4, 0.45], [
            0.35, 0.4, 0.45], [0.35, 0.4, 0.45]]

    elif patient == "LY":
        zoom_factors = 1.
        visualize_pos = [[0.15, 0.2, 0.25], [
            0.55, 0.6, 0.65], [0.3, 0.35, 0.4]]

    elif patient == "XXH":
        zoom_factors = 1.
        visualize_pos = [[0.35, 0.4, 0.45], [
            0.50, 0.55, 0.60], [0.40, 0.45, 0.50]]

    elif patient == "YXB":
        zoom_factors = 1.
        visualize_pos = [[0.25, 0.3, 0.35], [
            0.50, 0.55, 0.60], [0.34, 0.38, 0.42]]

    else:
        raise ValueError("No visualize_pos available.")

    # patient = "HR"  # (STT, HR, LY, XXH, YXB)
    if test == 0:
        dir_path = "../data/PatienTumorMultiScan2024/"
        brain, aff_info, header = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_brain_normalized.nii.gz'))
        brain_raw, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_brain_resized.nii.gz'))
        gm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_gm_normalized.nii.gz'))
        wm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_wm_normalized.nii.gz'))
        csf, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_csf_normalized.nii.gz'))
        tumor1, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_tumor_resized.nii.gz'))
        tumor2, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}2_tumor_resized.nii.gz'))
    else:
        dir_path = "../data/PatienTumorMultiScan2024/test_data/"
        brain, aff_info, header = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_brain_normalized.nii.gz'))
        brain_raw, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_brain_resized.nii.gz'))
        gm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_gm_normalized.nii.gz'))
        wm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_wm_normalized.nii.gz'))
        csf, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_csf_normalized.nii.gz'))
        tumor1, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_tumor_resized.nii.gz'))
        tumor2, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}2_tumor_resized.nii.gz'))

    # If image is too large
    if zoom_factors < 1.:
        zoom_factors = tuple([zoom_factors]*3)
        brain = zoom(brain, zoom_factors, order=1)
        brain_raw = zoom(brain_raw, zoom_factors, order=1)
        gm = zoom(gm, zoom_factors, order=1)
        wm = zoom(wm, zoom_factors, order=1)
        csf = zoom(csf, zoom_factors, order=1)
        tumor1 = zoom(tumor1, zoom_factors, order=1)
        tumor2 = zoom(tumor2, zoom_factors, order=1)

    return {
        "patient": patient,
        "dir_path": dir_path,
        "brain": brain,
        "brain_raw": brain_raw,
        "gm": gm,
        "wm": wm,
        "csf": csf,
        "tumor1": tumor1,
        "tumor2": tumor2,
        "aff_info": aff_info,
        "header": header,
        "visualize_pos": visualize_pos,
    }

def read_patient_data_2d(patient: str, test: bool = False):

    if patient == "STT":
        zoom_factors = 0.7
        slice_pos = 0.73

    elif patient == "HR":
        zoom_factors = 1.
        slice_pos = 0.4

    elif patient == "LY":
        zoom_factors = 1.
        slice_pos = 0.35

    elif patient == "XXH":
        zoom_factors = 1.
        slice_pos = 0.45

    elif patient == "YXB":
        zoom_factors = 1.
        slice_pos = 0.38

    else:
        raise ValueError("No visualize_pos available.")

    # patient = "HR"  # (STT, HR, LY, XXH, YXB)
    if test == 0:
        dir_path = "../data/PatienTumorMultiScan2024/"
        brain, aff_info, header = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_brain_normalized.nii.gz'))
        idx = int(brain.shape[2] * slice_pos)
        brain = brain[:, :, idx]
        brain_raw, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_brain_resized.nii.gz'))
        brain_raw = brain_raw[:, :, idx]
        gm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_gm_normalized.nii.gz'))
        gm = gm[:, :, idx]
        wm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_wm_normalized.nii.gz'))
        wm = wm[:, :, idx]
        csf, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_csf_normalized.nii.gz'))
        csf = csf[:, :, idx]
        tumor1, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}1_tumor_resized.nii.gz'))
        tumor1 = tumor1[:, :, idx]
        tumor2, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'{patient}2_tumor_resized.nii.gz'))
        tumor2 = tumor2[:, :, idx]
    else:
        dir_path = "../data/PatienTumorMultiScan2024/test_data/"
        brain, aff_info, header = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_brain_normalized.nii.gz'))
        idx = int(brain.shape[2] * slice_pos)
        brain = brain[:, :, idx]
        brain_raw, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_brain_resized.nii.gz'))
        brain_raw = brain_raw[:, :, idx]
        gm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_gm_normalized.nii.gz'))
        gm = gm[:, :, idx]
        wm, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_wm_normalized.nii.gz'))
        wm = wm[:, :, idx]
        csf, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_csf_normalized.nii.gz'))
        csf = csf[:, :, idx]
        tumor1, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}1_tumor_resized.nii.gz'))
        tumor1 = tumor1[:, :, idx]
        tumor2, _, _ = ndarray_from_nifty(os.path.join(
            dir_path, patient, f'test_{patient}2_tumor_resized.nii.gz'))
        tumor2 = tumor2[:, :, idx]
    # If image is too large
    if zoom_factors < 1.:
        zoom_factors = tuple([zoom_factors]*2)
        brain = zoom(brain, zoom_factors, order=1)
        brain_raw = zoom(brain_raw, zoom_factors, order=1)
        gm = zoom(gm, zoom_factors, order=1)
        wm = zoom(wm, zoom_factors, order=1)
        csf = zoom(csf, zoom_factors, order=1)
        tumor1 = zoom(tumor1, zoom_factors, order=1)
        tumor2 = zoom(tumor2, zoom_factors, order=1)

    return {
        "patient": patient,
        "dir_path": dir_path,
        "brain": brain,
        "brain_raw": brain_raw,
        "gm": gm,
        "wm": wm,
        "csf": csf,
        "tumor1": tumor1,
        "tumor2": tumor2,
        "aff_info": aff_info,
        "header": header,
        "slice_pos": slice_pos,
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
                          "D", "rho", "x0[0]", "x0[1]", "x0[2]"], "t1"]

    # Insert new row at the beginning (after header)
    existing_rows.insert(1, new_row)

    # Write updated content back to file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(existing_rows)


def append_parameters_to_file_2d(file_path, patient, experiment_type, D, rho, x0, t1=None):
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
                          "D", "rho", "x0[0]", "x0[1]"], "t1"]

    # Insert new row at the beginning (after header)
    existing_rows.insert(1, new_row)

    # Write updated content back to file
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(existing_rows)
