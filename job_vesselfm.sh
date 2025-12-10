#!/bin/bash
#SBATCH --job-name=ZarrNii
#SBATCH --output=batch_logs/%x_%j.out
#SBATCH --time=2:00:00
#SBATCH --error=batch_logs/%x_%j.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=rri-cbs-h1.schulich.uwo.ca

# Load required modules
module load python/3.10.16
module load pixi

# Activate virtual environment (customize this line)
#source ~/virtual_env/vesselfm_env/bin/activate

# Set experiment name and directories

EXP_NAME="ZarrNii"
TMP_DIR="/tmp/${EXP_NAME}"
PERSISTENT_DIR="/nfs/khan/trainees/ffallahz/ZarrNii" # (can be in /cifs)

# Create logs directory if it doesn't exist
mkdir -p batch_logs
echo "Copying experiment to ${TMP_DIR}..."
mkdir -p ${TMP_DIR}
cp -r ${PERSISTENT_DIR} ${TMP_DIR}






# Copy current working directory (codebase) to /tmp



echo "tmp dir"



# Run your training or main script (edit as needed)
echo "Running experiment..."
nvidia-smi
cd ~/virtual_env/zarrnii_env
pixi install -e gpu
pixi run -e gpu print
pixi run  -e gpu inference 
# Optional: Run inference or post-processing
# python your_inference_script.py --config your_inference_config.json

# Zip results (customize what to include)
cd ${TMP_DIR}
cd ZarrNii
pwd
ls
echo "Zipping results..."
tar -cvf results.tar results/
# Copy results back to persistent storage
echo "Copying results to ${PERSISTENT_DIR}..."
cp results.tar ${PERSISTENT_DIR}/

# Clean up temporary directory
cd /tmp
rm -rf ${TMP_DIR}
