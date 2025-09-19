#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u

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
RESULTS_DIR="$SCRIPT_DIR/logs/$TIMESTAMP-quantization-newagama"
export RESULTS_DIR

bash run.sh

exit 0