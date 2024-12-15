#!/project/6006512/muye/env/torch/bin/python
# SBATCH --job-name=ants_registration     # Job name
# SBATCH --account=def-jiguocao
# SBATCH --ntasks=1                   # Number of tasks (1 per patient)
# SBATCH --cpus-per-task=16            # Number of CPU cores per task (adjust as needed)
# SBATCH --mem-per-cpu=2G                     # Memory per task (adjust as needed)
# SBATCH --time=01:00:00              # Time limit

import ants
import numpy as np
import os
import sys
sys.path.append(os.getcwd())


atlas_path = "../data/Atlas/"

