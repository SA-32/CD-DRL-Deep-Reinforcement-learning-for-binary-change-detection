#!/bin/bash
#SBATCH --job-name = burger_test
#SBATCH --partition = gpu_mi300
#SBATCH --gres = gpu:1
#SBATCH --time = 48:00:00
#SBATCH --output = CD_DRL_%j.out
#SBATCH --error = CD_DRL_%j.err

cd "$SLURM_SUBMIT_DIR"

# Load Miniforge
module load devel/miniforge/25.3.1-python-3.12
module load toolkit/rocm/6.3.1
# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate burger_rocm

echo "Python:"
which python
python --version

echo "Jupyter:"
which jupyter

echo "Kernel list:"
jupyter kernelspec list

echo "PyTorch:"
python - <<EOF
import torch

print("Torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

jupyter nbconvert \
    --to notebook \
    --execute Train_Model.ipynb \
    --ExecutePreprocessor.kernel_name=burger_rocm \
    --ExecutePreprocessor.timeout=-1 \
    --output Train_Model_executed.ipynb

echo "Notebook execution completed."
