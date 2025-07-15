#!/bin/bash
#SBATCH --job-name=bpe_training
#SBATCH --output=bpe_training_%j.out
#SBATCH --error=bpe_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --partition=compute

# Load required modules (adjust based on your HPC setup)
source /home/orojoa/NetToken/.venv/bin/activate

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/data/orojoa/bpe_models_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# Log system information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CPU count: $(nproc)"
echo "Memory: $(free -h)"
echo "Output directory: $OUTPUT_DIR"

# Run the training script
python3 /home/orojoa/NetToken/src/nettokken/bpe/byte_level_field_aware_tokenizer.py
    --data-dir /data/ \
    --output-dir $OUTPUT_DIR \
    --vocab-sizes 512 1024 4096 8192 16384 32768 \
    --num-workers 34 \
    --chunk-size 50 \
    --checkpoint-interval 1000

# Check exit status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at: $(date)"
else
    echo "Training failed with exit code: $?" >&2
fi

# Cleanup temp files
rm -rf /tmp/tmp*

# Generate summary report
echo "=== Training Summary ===" > $OUTPUT_DIR/summary.txt
echo "Start time: $TIMESTAMP" >> $OUTPUT_DIR/summary.txt
echo "End time: $(date +%Y%m%d_%H%M%S)" >> $OUTPUT_DIR/summary.txt
echo "Node: $(hostname)" >> $OUTPUT_DIR/summary.txt
echo "" >> $OUTPUT_DIR/summary.txt
echo "Models created:" >> $OUTPUT_DIR/summary.txt
ls -la $OUTPUT_DIR/bpe_vocab* >> $OUTPUT_DIR/summary.txt