#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u

# --- Trap Handler ---
# No temporary files to clean up in this version.
if [ -z "${DEVICE:-}" ]; then
  DEVICE=xpu
  export DEVICE
fi

# --- Script Start ---
start_time=$(date +%s)

# Record script start timestamp
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

# Get the directory where this script is located
# SCRIPT_DIR="$(dirname "$0")"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Define the full path to the results directory
RESULTS_DIR="$SCRIPT_DIR/logs/$TIMESTAMP-quantization"
export RESULTS_DIR=$RESULTS_DIR

# --- Setup ---

# Create the timestamped directory for results
echo "Creating results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR" # Use -p to avoid error if directory already exists (unlikely with timestamp, but safe)

export TORCHINDUCTOR_CACHE_DIR="$RESULTS_DIR/torchinductor_cache"
echo "TORCHINDUCTOR_CACHE_DIR = $TORCHINDUCTOR_CACHE_DIR"

# Collect environment information
echo "Collecting environment information..."
# Assuming collect_env.py is one directory level up from where this script is located
COLLECT_ENV_SCRIPT="$SCRIPT_DIR/../collect_env.py"
# Check if the environment collection script exists and is executable
if [ ! -s "$COLLECT_ENV_SCRIPT" ]; then
    echo "Warning: collect_env.py script not found or not executable at $COLLECT_ENV_SCRIPT. Skipping environment collection." >&2
    # Do not exit, just warn and continue script execution
else
    # Redirect output to the results directory
    set +e
    python "$COLLECT_ENV_SCRIPT" >"$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ pip list | grep transformers\n" >>"$RESULTS_DIR/collect_env.log" 2>&1
    pip list | grep transformers >>"$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep igc\n" >>"$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep igc >>"$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep dkms\n" >>"$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep dkms >>"$RESULTS_DIR/collect_env.log" 2>&1
    echo "Done collecting environment info." >>"$RESULTS_DIR/collect_env.log" # Append Done message to the log file
    echo "Environment log saved to $RESULTS_DIR/collect_env.log"
    set -e
fi

echo "Running llama3 models..."

python -u run_generation.py -m meta-llama/Llama-3.1-8B-Instruct --input-tokens 1024 --max-new-tokens 128 \
    --num-iter 2 --num-warmup 1 --batch-size 1 --device $DEVICE --token-latency --num-beams 1 --inductor \
    --use-static-cache --use-hf-code False --woq --woq-type rtn \
    --group-size 128 --quant-dtype uint4 --profile --attn_type flex_attention \
    >>"$RESULTS_DIR/llama31.uint4.fa.$DEVICE.profile.log" 2>&1

# python -u run_generation.py -m meta-llama/Llama-3.1-8B-Instruct --input-tokens 1024 --max-new-tokens 128 \
#     --num-iter 8 --num-warmup 4 --batch-size 1 --device $DEVICE --token-latency --num-beams 1 --inductor \
#     --use-static-cache --use-hf-code False --woq --woq-type rtn \
#     --group-size 128 --quant-dtype uint4 --profile --attn_type sdpa \
#     >>"$RESULTS_DIR/llama31.uint4.sdpa.$DEVICE.profile.log" 2>&1

echo "Finished running llama3 models!"

# --- Script End ---

# Create the final marker file in the results directory.
echo "Done" >"$RESULTS_DIR/finish.log"
echo "All specified test runs completed."
echo "Detailed logs and results, including collected commands, are located in the '$RESULTS_DIR/' directory."
echo "Command files: $RESULTS_DIR/*.commands.txt"
echo "Test logs: $RESULTS_DIR/*.test.log"

# No temporary file cleanup needed as commands files are permanent results.
# Trap handler is not needed as no temporary files are created.

# Exit with status 0 to indicate overall script success (unless set -e caused an earlier exit).
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds."
echo "Total elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds." >>"$RESULTS_DIR/finish.log"
echo "Script completed successfully at $(date '+%Y-%m-%d %H:%M:%S')."

exit 0
