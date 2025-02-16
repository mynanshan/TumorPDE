#!/project/6006512/muye/env/torch/bin/python
#SBATCH --job-name=ants_registration     # Job name
#SBATCH --output=logs/registration_%A_%a.out
#SBATCH --account=def-jiguocao
#SBATCH --ntasks=1                   # Number of tasks (1 per patient)
#SBATCH --cpus-per-task=16           # Number of CPU cores per task (adjust as needed)
#SBATCH --mem-per-cpu=4G             # Memory per task (adjust as needed)
#SBATCH --time=01:30:00              # Time limit

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
num_scan = int(sys.argv[2])
ref_scan_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1

for i in range(num_scan + 1):

    # process the ref scan in the 1st iteration
    scan_id = i if i != 0 else ref_scan_id
    if i == ref_scan_id:
        continue
    print(f"Registering Patient: {patient} Scan ID: {scan_id}")
    dir_path = f"../data/PatienTumorMultiScan2024/{patient}/"
    patient_t1 = ants.image_read(os.path.join(
        dir_path, f'{patient}{scan_id}_t1.nii'),
        reorient=True)
    tumor_mask = ants.image_read(os.path.join(
        dir_path, f'{patient}{scan_id}_t1mask.nii'))

    # Resample patient image and tumor mask
    if scan_id == ref_scan_id:
        # if processing the ref scan
        # Get the voxel sizes and resample parameters
        voxel_sizes = patient_t1.spacing
        min_spacing = min(voxel_sizes)
        resample_params = (min_spacing, min_spacing, min_spacing)

        patient_t1 = patient_t1.resample_image(
            resample_params, use_voxels=False, interp_type=0)
        ref_image = patient_t1
    else:
        # if processing the rest of the scans
        # patient_t1 = patient_t1.resample_image_to_target(
        #     ref_image, interp_type='linear')

        # Perform registration, aligning different scans
        reg = ants.registration(
            fixed=ref_image,
            moving=patient_t1,
            type_of_transform='Similarity', # only tranlation, rotation and scaling
            mask=1.0 - tumor_mask,
            outprefix=os.path.join(dir_path, f'{patient}{scan_id}_prereg_')
        )

        patient_t1 = reg['warpedmovout']
        tumor_mask = ants.apply_transforms(
            fixed=ref_image, moving=tumor_mask, transformlist=reg['fwdtransforms'])


    tumor_mask = tumor_mask.resample_image_to_target(patient_t1, interp_type='nearestNeighbor')

    # Resample atlas images to match patient image
    atlas_t1 = atlas_t1.resample_image_to_target(patient_t1, interp_type='linear')
    atlas_gm = atlas_gm.resample_image_to_target(patient_t1, interp_type='linear')
    atlas_wm = atlas_wm.resample_image_to_target(patient_t1, interp_type='linear')
    atlas_csf = atlas_csf.resample_image_to_target(
        patient_t1, interp_type='linear')


    patient_t1.to_file(os.path.join(dir_path, f'{patient}{scan_id}_t1_resized.nii.gz'))
    tumor_mask.to_file(os.path.join(dir_path, f'{patient}{scan_id}_t1mask_resized.nii.gz'))

    patient_t1.plot(overlay=tumor_mask, overlay_alpha=0.3, overlay_cmap="Reds",
                    title='Scan & Tumor Mask', axis=2, nslices=16,
                    filename=os.path.join(dir_path, f'{patient}{scan_id}_raw_with_tumor.jpg'))
    patient_t1.plot(overlay=atlas_t1, overlay_alpha=0.6, overlay_cmap="Blues",
                    title='Before Registration', axis=2, nslices=16,
                    filename=os.path.join(dir_path, f'{patient}{scan_id}_before_register.jpg'))
    patient_t1.plot(overlay=ref_image, overlay_alpha=0.3, overlay_cmap="Blues",
                    title=f'Scan{scan_id} vs {ref_scan_id}', axis=2, nslices=16,
                    filename=os.path.join(dir_path, f'{patient}{scan_id}_and_{ref_scan_id}.jpg'))

    # Perform registration
    reg = ants.registration(
        fixed=patient_t1,
        moving=atlas_t1,
        type_of_transform='SyN', # 'Elastic'
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

    # Save the transformed images
    warped_atlas.to_file(os.path.join(
        dir_path, f'{patient}{scan_id}_brain_normalized.nii.gz'))
    warped_gm.to_file(os.path.join(dir_path, f'{patient}{scan_id}_gm_normalized.nii.gz'))
    warped_wm.to_file(os.path.join(dir_path, f'{patient}{scan_id}_wm_normalized.nii.gz'))
    warped_csf.to_file(os.path.join(dir_path, f'{patient}{scan_id}_csf_normalized.nii.gz'))

    # Save the transformation filenames
    np.savetxt(os.path.join(
        dir_path, f'{patient}{scan_id}_fwdtransforms.txt'), reg['fwdtransforms'], fmt='%s')
    np.savetxt(os.path.join(
        dir_path, f'{patient}{scan_id}_invtransforms.txt'), reg['invtransforms'], fmt='%s')

    # Plot the registered image
    patient_t1.plot(overlay=warped_atlas, overlay_alpha=0.6, overlay_cmap="Blues",
                    title='After Registration', axis=2, nslices=16,
                    filename=os.path.join(dir_path, f'{patient}{scan_id}_brain_registered.jpg'))
    warped_atlas.plot(overlay=tumor_mask, overlay_alpha=0.6, overlay_cmap="Reds",
                        title='Registered Brain & Tumor', axis=2, nslices=16,
                        filename=os.path.join(dir_path, f'{patient}{scan_id}_brain_registered_tumor.jpg'))
