#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u

# --- Trap Handler ---
# No temporary files to clean up in this version.

# --- Script Start ---
start_time=$(date +%s)

# Record script start timestamp
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$0")"

# Define the full path to the results directory
RESULTS_DIR="$TIMESTAMP-triton_pin-1133-entire"

# Define paths to the helper scripts (assuming they are in the same directory as this script)
COLLECT_SCRIPT="$SCRIPT_DIR/collect_tests.py"
RUNNER_SCRIPT="$SCRIPT_DIR/run_tests_sequentially.sh"

# Define the specific test files you want to run
TEST_FILES=(
    "../hoshibara-pytorch/test/inductor/test_flex_attention.py"
    "../hoshibara-pytorch/test/inductor/test_flex_decoding.py"
)

# Check if the specified test files exist before starting
for TEST_FILE in "${TEST_FILES[@]}"; do
    if [ ! -f "$TEST_FILE" ]; then
        echo "Error: Specified test file not found: $TEST_FILE" >&2
        echo "Please ensure the path is correct relative to where you run this script." >&2
        exit 1 # Exit if any of the specified test files are missing
    fi
done


# --- Setup ---

# Create the timestamped directory for results
echo "Creating results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR" # Use -p to avoid error if directory already exists (unlikely with timestamp, but safe)


# Collect environment information
echo "Collecting environment information..."
# Assuming collect_env.py is one directory level up from where this script is located
COLLECT_ENV_SCRIPT="$SCRIPT_DIR/../collect_env.py"
# Check if the environment collection script exists and is executable
set +e
if [ ! -s "$COLLECT_ENV_SCRIPT" ]; then
    echo "Warning: collect_env.py script not found or not executable at $COLLECT_ENV_SCRIPT. Skipping environment collection." >&2
    # Do not exit, just warn and continue script execution
else
    # Redirect output to the results directory
    python "$COLLECT_ENV_SCRIPT" > "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep igc\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep igc >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo -e "\n\n$ dpkg -l | grep dkms\n" >> "$RESULTS_DIR/collect_env.log" 2>&1
    dpkg -l | grep dkms >> "$RESULTS_DIR/collect_env.log" 2>&1
    echo "Done collecting environment info." >> "$RESULTS_DIR/collect_env.log" # Append Done message to the log file
    echo "Environment log saved to $RESULTS_DIR/collect_env.log"
fi
set -e


TORCH_LOGS="+output_code" python run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 100 \
  --input-tokens 1024 --num-warmup 5 --num-iter 15 --compile --profile >> "$RESULTS_DIR/llama31.compile.xpu.profile.log" 2>&1

# --- Test Execution Loop ---

# Loop through each specified test file and run the collection and execution sequence
for TEST_FILE in "${TEST_FILES[@]}"; do
    echo "--- Processing tests for $TEST_FILE ---"

    # Check if the test file exists (already checked, but defensive)
     if [ ! -f "$TEST_FILE" ]; then
        echo "Error: Test file not found at path $TEST_FILE. Skipping this iteration." >&2
        continue # Skip to the next file
     fi

    # --- Step 2: Run Collected Commands Sequentially ---
    # Call the run_tests_sequentially.sh script. It takes the commands file path
    # and the original test file path.
    # Paths are relative to the directory where THIS script is run.
    # We redirect its entire standard output and standard error to the test log file
    # within the results directory.
    TEST_LOG_FILE="$RESULTS_DIR/$(basename "$TEST_FILE" .py).result.log"
    echo "Full log will be saved to $TEST_LOG_FILE"
    # The bash runner script exit status is checked by 'set -e'.
    set +e
    echo "--- Starting test execution at $(date) ---" >> "$TEST_LOG_FILE"
    python "$TEST_FILE" >> "$TEST_LOG_FILE" 2>&1
    echo "--- Test execution finished at $(date) ---" >> "$TEST_LOG_FILE"
    set -e
    # Note: Any 'tee' commands within run_tests_sequentially.sh will now have their
    # console output redirected into the log file as well.
    echo "Finished running commands for $TEST_FILE."


    echo "" # Add a blank line for readability between test file runs

done # End loop over test files


# --- Script End ---

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