#!/project/6006512/muye/env/torch/bin/python
#SBATCH --job-name=ants_registration     # Job name
#SBATCH --account=def-jiguocao
#SBATCH --ntasks=1                   # Number of tasks (1 per patient)
#SBATCH --cpus-per-task=16            # Number of CPU cores per task (adjust as needed)
#SBATCH --mem-per-cpu=4G                     # Memory per task (adjust as needed)
#SBATCH --time=01:00:00              # Time limit

import ants
import numpy as np
import os
import sys
sys.path.append(os.getcwd())


atlas_path = "../data/Atlas/"
atlas_t1 = ants.image_read(os.path.join(
    atlas_path, "atlas_t1.nii"), reorient=True)
atlas_mask = ants.image_read(os.path.join(
    atlas_path, "atlas_mask.nii"), reorient=True)
atlas_gm = ants.image_read(os.path.join(
    atlas_path, "atlas_gm.nii"), reorient=True)
atlas_wm = ants.image_read(os.path.join(
    atlas_path, "atlas_wm.nii"), reorient=True)
atlas_csf = ants.image_read(os.path.join(
    atlas_path, "atlas_csf.nii"), reorient=True)

atlas_t1 = atlas_t1 * atlas_mask
del atlas_mask

# Get patient ID from the command line
patient = sys.argv[1]
scan_id = sys.argv[2]
print(f"Registering Patient: {patient} Scan ID: {scan_id}")
dir_path = f"../data/PatienTumorMultiScan2024/{patient}/"
patient_t1 = ants.image_read(os.path.join(
    dir_path, f'{patient}{scan_id}_brain.nii'),
    reorient=True)

tumor_mask = ants.image_read(os.path.join(
    dir_path, f'{patient}{scan_id}_tumor.nii'))

# Get the voxel sizes and resample parameters
voxel_sizes = patient_t1.spacing
min_spacing = min(voxel_sizes)
resample_params = (min_spacing, min_spacing, min_spacing)

# Resample patient image and tumor mask
patient_t1 = patient_t1.resample_image(
    resample_params, use_voxels=False, interp_type=0)
tumor_mask = tumor_mask.resample_image_to_target(patient_t1, interp_type='nearestNeighbor')

# Resample atlas images to match patient image
atlas_t1 = atlas_t1.resample_image_to_target(patient_t1, interp_type='linear')
atlas_gm = atlas_gm.resample_image_to_target(patient_t1, interp_type='linear')
atlas_wm = atlas_wm.resample_image_to_target(patient_t1, interp_type='linear')
atlas_csf = atlas_csf.resample_image_to_target(
    patient_t1, interp_type='linear')


patient_t1.to_file(os.path.join(dir_path, f'{patient}{scan_id}_brain_resized.nii.gz'))
tumor_mask.to_file(os.path.join(dir_path, f'{patient}{scan_id}_tumor_resized.nii.gz'))

patient_t1.plot(overlay=tumor_mask, overlay_alpha=0.3, overlay_cmap="Reds",
                title='Scan & Tumor Mask', axis=2, nslices=16,
                filename=os.path.join(dir_path, f'{patient}{scan_id}_raw_with_tumor.jpg'))
patient_t1.plot(overlay=atlas_t1, overlay_alpha=0.6, overlay_cmap="Blues",
                title='Before Registration', axis=2, nslices=16,
                filename=os.path.join(dir_path, f'{patient}{scan_id}_before_register.jpg'))

# Perform registration
reg = ants.registration(
    fixed=patient_t1,
    moving=atlas_t1,
    type_of_transform='Elastic',
    mask=1.0 - tumor_mask,
    outprefix=os.path.join(dir_path, f'{patient}{scan_id}_reg_')
)

# Apply transforms to atlas images and tumor mask
warped_atlas = reg['warpedmovout']
warped_gm = ants.apply_transforms(
    fixed=patient_t1, moving=atlas_gm, transformlist=reg['fwdtransforms'])
warped_wm = ants.apply_transforms(
    fixed=patient_t1, moving=atlas_wm, transformlist=reg['fwdtransforms'])
warped_csf = ants.apply_transforms(
    fixed=patient_t1, moving=atlas_csf, transformlist=reg['fwdtransforms'])
# warped_tumor = ants.apply_transforms(
#     fixed=patient_t1, moving=tumor_mask, transformlist=reg['fwdtransforms'],
#     interpolator='nearestNeighbor')
# warped_raw = ants.apply_transforms(
#     fixed=warped_atlas, moving=patient_t1, transformlist=reg['invtransforms'])

# Save the transformed images
warped_atlas.to_file(os.path.join(
    dir_path, f'{patient}{scan_id}_brain_normalized.nii.gz'))
warped_gm.to_file(os.path.join(dir_path, f'{patient}{scan_id}_gm_normalized.nii.gz'))
warped_wm.to_file(os.path.join(dir_path, f'{patient}{scan_id}_wm_normalized.nii.gz'))
warped_csf.to_file(os.path.join(dir_path, f'{patient}{scan_id}_csf_normalized.nii.gz'))
# warped_tumor.to_file(os.path.join(
#     dir_path, f'{patient}{scan_id}_tumor_normalized.nii.gz'))
# warped_raw.to_file(os.path.join(
#     dir_path, f'{patient}{scan_id}_brain_raw_inv.nii.gz'))

# Save the transformation filenames
np.savetxt(os.path.join(
    dir_path, f'{patient}{scan_id}_fwdtransforms.txt'), reg['fwdtransforms'], fmt='%s')
np.savetxt(os.path.join(
    dir_path, f'{patient}{scan_id}_invtransforms.txt'), reg['invtransforms'], fmt='%s')

# Plot the registered image
patient_t1.plot(overlay=warped_atlas, overlay_alpha=0.6, overlay_cmap="Blues",
                title='After Registration', axis=2, nslices=16,
                filename=os.path.join(dir_path, f'{patient}{scan_id}_brain_registered.jpg'))
