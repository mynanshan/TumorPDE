#!/project/6006512/muye/env/torch/bin/python
#SBATCH --job-name=gif-maker     # Job name
#SBATCH --account=def-jiguocao
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=1:00:00
#SBATCH --output="gif-maker-%A_%a.out"

from PIL import Image
import os
import re
import sys
import argparse

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser(description='Process patient images to GIFs.')
parser.add_argument('-i', '--image_folder', required=True, help='Folder containing the patient images')
parser.add_argument('-o', '--output_folder', required=True, help='Folder to save the GIFs')
parser.add_argument('-p', '--patient', required=True, help='Patient identifier')
args = parser.parse_args()

image_folder = args.image_folder
output_folder = args.output_folder
patient = args.patient


def extract_info(filename):
    match = re.match(r'(.*)-i(\d+)\.jpg', filename)
    if match:
        return match.group(1), int(match.group(2))
    else:
        return None, None


patient_img_folder = os.path.join(image_folder, patient)
patient_output_folder = os.path.join(output_folder)

if not os.path.exists(patient_img_folder):
    print(f"Patient {patient} not found")
    raise ValueError(f"Patient {patient} does not exist.")
else:
    print(f"Processing patient {patient}")

# Get a list of all the image files in the folder
images = [img for img in os.listdir(patient_img_folder) if img.endswith(".jpg")]

# Sort images based on the name and numeric value extracted from the filenames
images.sort(key=lambda img: extract_info(img)
            if extract_info(img) else ('', 0))

# Create a dictionary to store frames for each name
frames_dict = {}

# Load all images into a dictionary of lists
for image in images:
    name, number = extract_info(image)
    if name:
        if name not in frames_dict:
            frames_dict[name] = []
        frames_dict[name].append(Image.open(os.path.join(patient_img_folder, image)))

os.makedirs(patient_output_folder, exist_ok=True)

# Save the frames as GIFs for each name
for name, frames in frames_dict.items():
    frames[0].save(os.path.join(patient_output_folder, f'{name}.gif'), format='GIF',
                append_images=frames[1:], save_all=True, duration=200, loop=0)
