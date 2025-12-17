#!/bin/bash
#SBATCH --job-name=ZarrNii
#SBATCH --output=batch_logs/%x_%j.out
#SBATCH --time=5:00:00
#SBATCH --error=batch_logs/%x_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1


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
mkdir -p ${TMP_DIR}/tmp_results
cp -r ${PERSISTENT_DIR} ${TMP_DIR} #cp -r /nfs/khan/trainees/ffallahz/ZarrNii /tmp/ZarrNii

#cp -r /nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3/bids/sub-AS164F5/micr/sub-AS164F5_sample-brain_acq-imaris4x_SPIM.ome.zarr ${TMP_DIR}/ZarrNii/data/lightsheet
cp "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1/derivatives/spimquant_aae813e/sub-AS134F3/micr/sub-AS134F3_sample-brain_acq-imaris4x_seg-all_from-ABAv3_level-5_desc-deform_dseg.nii.gz" ${TMP_DIR}/ZarrNii/data/lightsheet
cp "/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch1/derivatives/spimquant_aae813e/tpl-ABAv3/seg-all_tpl-ABAv3_dseg.tsv" ${TMP_DIR}/ZarrNii/data/lightsheet


# Copy current working directory (codebase) to /tmp



echo "tmp dir"



# Run your training or main script (edit as needed)
echo "Running experiment..."
nvidia-smi
cd ~/virtual_env/zarrnii_env
pixi install -e gpu
pixi run -e gpu print
pixi run  -e gpu save 
# Optional: Run inference or post-processing
# python your_inference_script.py --config your_inference_config.json

# Zip results (customize what to include)
cd ${TMP_DIR}

pwd
ls
echo "Zipping results..."
tar -cvf results.tar tmp_results/
# Copy results back to persistent storage
echo "Copying results to ${PERSISTENT_DIR}..."
cp results.tar ${PERSISTENT_DIR}/

# Clean up temporary directory
cd /tmp
rm -rf ${TMP_DIR}
