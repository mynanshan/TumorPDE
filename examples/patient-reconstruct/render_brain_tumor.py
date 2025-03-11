import os
import sys
sys.path.append(os.getcwd())

import argparse

parser = argparse.ArgumentParser(description='Process patient data.')
parser.add_argument('-p', '--patient', required=True, help='Patient identifier')
parser.add_argument('-r', '--ref_scan', type=int, default=1,
                    required=False, help='Which scan is used as the ref_scan')
parser.add_argument('-e', '--expr_code', type=str, required=True, help='Experiment name')
args = parser.parse_args()

patient = args.patient
scan_id = args.ref_scan
expr_code = args.expr_code

data_path = "../data/PatienTumorMultiScan2024/"
res_path = f"./results/simulation_{expr_code}"

# Load the NIfTI Volumes (Brain and Tumor) -------------------
brainVolumeNode = slicer.util.loadVolume(
    os.path.join(data_path, patient, f'{patient}{scan_id}_t1_resized.nii.gz'))
tumorVolumeNode = slicer.util.loadVolume(
    os.path.join(res_path, patient, f"{patient}-i1000.nii.gz"))

# Enable Volume Rendering for the Brain Volume -----------------
vrLogic = slicer.modules.volumerendering.logic()  
brainVRDisplay = vrLogic.CreateDefaultVolumeRenderingNodes(brainVolumeNode)
brainVRDisplay.SetVisibility(True)  

# Adjust Brain Volume Transparency (Make it Semi-Transparent) -------------------
brainVRDisplay.GetVolumePropertyNode().Copy( vrLogic.GetPresetByName("MR-Default") )

# Get the volume property and its scalar opacity function
brainVolProp = brainVRDisplay.GetVolumePropertyNode().GetVolumeProperty()
scalarOpacityFunc = brainVolProp.GetScalarOpacity()

# Clear existing points
scalarOpacityFunc.RemoveAllPoints()
# Define a custom opacity curve (example values; adjust to your data range)
scalarOpacityFunc.AddPoint(0, 0.0)      # 0 intensity -> 0% opacity (invisible)
scalarOpacityFunc.AddPoint(100, 0.2)    # low intensities -> 20% opacity
scalarOpacityFunc.AddPoint(1000, 0.4)   # high intensities -> 40% opacity

# Overlay the Tumor Volume with a Distinct Color --------------------
tumorVRDisplay = vrLogic.CreateDefaultVolumeRenderingNodes(tumorVolumeNode)
tumorVRDisplay.SetVisibility(True)

# Get the volume property for the tumor
tumorVolProp = tumorVRDisplay.GetVolumePropertyNode().GetVolumeProperty()
colorFunc = tumorVolProp.GetRGBTransferFunction()     # color transfer function
opacityFunc = tumorVolProp.GetScalarOpacity()         # opacity transfer function

# Reset any existing transfer function points
colorFunc.RemoveAllPoints()
opacityFunc.RemoveAllPoints()

# Set background (0) to fully transparent
colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)    # color doesn't matter if opacity=0
opacityFunc.AddPoint(0, 0.0)

# Set tumor (1) to opaque and colored (e.g., red)
colorFunc.AddRGBPoint(1, 1.0, 0.0, 0.0)    # RGB = (1,0,0) for red color
opacityFunc.AddPoint(1, 1.0)              # 100% opacity for tumor label

# # Fine-tuning ----------------------
# threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
# threeDView.resetFocalPoint()
# threeDView.resetCamera()