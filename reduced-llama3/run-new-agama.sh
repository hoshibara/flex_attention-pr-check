#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u

# --- Trap Handler ---
# No temporary files to clean up in this version.
DEVICE=xpu
export DEVICE

# --- Script Start ---
start_time=$(date +%s)

# Record script start timestamp
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

# Get the directory where this script is located
# SCRIPT_DIR="$(dirname "$0")"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)


WORKSPACE="/home/xingyuan/projects/20250910-FA/igc-agama/data/waihungt/QuickBuild"
export TargetDriver="open-linux-driver-ci-rel_igc_2.18.x-8"
export IGC_INSTALL=${WORKSPACE}/${TargetDriver}
export PATH=$IGC_INSTALL/usr/bin/:$PATH
export IGC_INSTALL_LIBS=$IGC_INSTALL/usr/lib/x86_64-linux-gnu/
export IGC_INSTALL_LIBS_LOCAL=$IGC_INSTALL/usr/local/lib/
export LD_LIBRARY_PATH=$IGC_INSTALL_LIBS_LOCAL:$IGC_INSTALL_LIBS_LOCAL/intel-opencl:$IGC_INSTALL_LIBS:$IGC_INSTALL_LIBS/intel-opencl:$IGC_INSTALL_LIBS/dri:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "$(readlink -m $(find $IGC_INSTALL -name libigdrcl.so))" > $IGC_INSTALL/etc/OpenCL/vendors/intel.icd || exit 1
export OPENCL_VENDOR_PATH=$IGC_INSTALL/etc/OpenCL/vendors/
export OCL_ICD_VENDORS=$OPENCL_VENDOR_PATH


# Define the full path to the results directory
RESULTS_DIR="$SCRIPT_DIR/logs/$TIMESTAMP-reduced-llama3-newagama"
export RESULTS_DIR

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
    python "$COLLECT_ENV_SCRIPT" > "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ pip list | grep transformers\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    pip list | grep transformers >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep igc\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep igc >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep dkms\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep dkms >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo "Done collecting environment info." >> "$RESULTS_DIR/collect_env.log" # Append Done message to the log file
    echo "Environment log saved to $RESULTS_DIR/collect_env.log"
    set -e
fi

echo "Running llama3 models..."

# FA profile
python -u run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 2 \
  --input-tokens 1024 --num-warmup 2 --num-iter 4 --compile --profile --attn_type=flex_attention \
  --device $DEVICE >> "$RESULTS_DIR/llama31.fa.compile.xpu.profile.log" 2>&1

# FA verbose
# ONEDNN_VERBOSE=all python -u run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 2 \
#   --input-tokens 1024 --num-warmup 2 --num-iter 4 --compile --profile --attn_type=flex_attention \
#   --device $DEVICE >> "$RESULTS_DIR/llama31.fa.compile.xpu.profile.onednn.log" 2>&1

# sdpa profile
python -u run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 2 \
  --input-tokens 1024 --num-warmup 2 --num-iter 4 --compile --profile --attn_type=sdpa \
  --device $DEVICE >> "$RESULTS_DIR/llama31.sdpa.compile.xpu.profile.log" 2>&1


echo "Finished running llama3 models!"

# --- Script End ---

python process_log.py $RESULTS_DIR

# Create the final marker file in the results directory.
echo "Done" > "$RESULTS_DIR/finish.log"
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
echo "Total elapsed time: $((elapsed_time / 60)) minutes and $((elapsed_time % 60)) seconds." >> "$RESULTS_DIR/finish.log"
echo "Script completed successfully at $(date '+%Y-%m-%d %H:%M:%S')."

exit 0