#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================
CONDA_ENV_NAME="stock"  # <--- CHANGE THIS to your actual conda environment name

# Lists of targets to process
SYMBOLS=("NVDA" "MSTR" "IONQ")
BAR_SIZES=("5 mins" "1 hour" "4 hours" "1 day" "1 w")

# ==============================================================================
# Execution Logic
# ==============================================================================

# Find the machine's shell profile to safely source Conda path binaries
if [ -f "$HOME/.zshrc" ]; then
    source "$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

echo "========================================================================"
echo "Initializing Multi-Asset Trading Data Merger Suite"
echo "========================================================================"

# Activate the conda environment safely
echo "Activating Conda environment: '${CONDA_ENV_NAME}'..."
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'."
    exit 1
fi

# Outer Loop: Iterate through each stock symbol
for SYMBOL in "${SYMBOLS[@]}"; do
    echo "========================================================================"
    echo "STARTING PROCESSING FOR TICKER: ${SYMBOL}"
    echo "========================================================================"
    
    # Inner Loop: Iterate through each bar size bracket for the current stock
    for BS in "${BAR_SIZES[@]}"; do
        echo "Processing [${SYMBOL}] at bar size [${BS}]..."
        
        # Run your python script with quotes to preserve spacing
        # python data_merger.py -s "$SYMBOL" -bs "$BS"
        python feature_engineering.py -s "$SYMBOL" -bs "$BS"

        
        # Error tracking fallback check
        if [ $? -ne 0 ]; then
            echo "Warning: Python crashed on ${SYMBOL} for bar size '${BS}'."
        else
            echo "Success: Done with ${SYMBOL} (${BS})."
        fi
        echo "------------------------------------------------------------------------"
    done
done

echo "========================================================================"
echo "All stock assets and bar size matrices processed completely."
echo "========================================================================"
