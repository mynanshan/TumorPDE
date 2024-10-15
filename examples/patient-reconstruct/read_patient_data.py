import os
from scan_utils import ndarray_from_nifty
from scipy.ndimage import zoom


def read_patient_data(patient: str, test: bool = False):

    if patient=="STT":
        zoom_factors = 0.7
        visualize_pos = [[0.4,0.45,0.5],[0.55,0.6,0.65],[0.65,0.73,0.81]]

    elif patient=="HR":
        zoom_factors = 1.
        visualize_pos = [[0.35,0.4,0.45],[0.35,0.4,0.45],[0.35,0.4,0.45]]

    elif patient=="LY":
        zoom_factors = 1.
        visualize_pos = [[0.15,0.2,0.25],[0.55,0.6,0.65],[0.3,0.35,0.4]]

    elif patient=="XXH":
        zoom_factors = 1.
        visualize_pos = [[0.35,0.4,0.45],[0.50,0.55,0.60],[0.40,0.45,0.50]]

    elif patient=="YXB":
        zoom_factors = 1.
        visualize_pos = [[0.25,0.3,0.35],[0.50,0.55,0.60],[0.34,0.38,0.42]]

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
        # tumor_id = 1
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
        # tumor_id = 1
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