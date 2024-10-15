import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os

def resize_nifti(input_filepath, output_filepath, zoom_factor):
    """
    Reads a NIfTI file, resizes it by the given zoom factor, and saves the resized image.

    Parameters:
    - input_filepath: Path to the input NIfTI file.
    - output_filepath: Path where the resized NIfTI file will be saved.
    - zoom_factor: A tuple or list with zoom factors for each axis (e.g., (0.07, 0.07, 0.07)).
    """
    # Load the NIfTI file
    nifti_img = nib.load(input_filepath)
    data = nifti_img.get_fdata()
    affine = nifti_img.affine
    header = nifti_img.header

    print(f"Original shape: {data.shape}")

    # Apply zoom to the data
    resized_data = zoom(data, zoom=zoom_factor, order=1)  # order=1 for linear interpolation

    print(f"Resized shape: {resized_data.shape}")

    # Adjust the affine to account for the zoom
    new_affine = affine.copy()
    # Assuming isotropic scaling, adjust the voxel sizes
    # If zoom factors are different, you need to adjust accordingly for each axis
    new_affine[:3, :3] = affine[:3, :3] * np.array(zoom_factor)[:, np.newaxis]

    # Create a new NIfTI image
    resized_img = nib.Nifti1Image(resized_data, new_affine, header)

    # Save the resized image
    nib.save(resized_img, output_filepath)
    print(f"Resized NIfTI saved to {output_filepath}")


if __name__ == "__main__":

    patient_names = ["HR", "LY", "STT", "XXH", "YXB"]
    scan_nums = [2, 2, 2, 2, 3]
    file_names = ["brain_normalized.nii.gz", "brain_resized.nii.gz",
                  "gm_normalized.nii.gz", "wm_normalized.nii.gz",
                  "csf_normalized.nii.gz", "tumor_resized.nii.gz"]
    data_dir = "../data/PatienTumorMultiScan2024/"
    output_dir = "../data/PatienTumorMultiScan2024/test_data/"
    
    # Define the zoom factor
    zoom_factors = (0.07, 0.07, 0.07)  # For (n, m, k) to (0.07n, 0.07m, 0.07k)

    for i,patient in enumerate(patient_names):
        print(f">>>>>>  Processing {patient} <<<<<<")
        for scan_num in range(1, scan_nums[i]+1):
            print(f"[Scan ID: {scan_num}]")
            prefix = f"{patient}{scan_num}_"
            for fname in file_names:
                print(fname)
                input_file = f"{data_dir}/{patient}/{prefix}{fname}"
                output_file = f"{output_dir}/{patient}/test_{prefix}{fname}"
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Resize and save the NIfTI file
                resize_nifti(input_file, output_file, zoom_factors)
