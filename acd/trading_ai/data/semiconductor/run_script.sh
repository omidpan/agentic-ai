#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================
CONDA_ENV_NAME="data-engineer"  # <--- CHANGE THIS to your actual conda environment name

# Lists of targets to process
SYMBOLS=("amat" "amd" "avgo" "intc" "mrvl" "mu" "nvda" "tsm" "spy" "smh")
BAR_SIZES=("4hours")
CONTEXT=("spy" "smh")

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
    
    # Convert $SYMBOL to lowercase safely across both Bash and Zsh using tr
    lower_symbol=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
    
    # Inner Loop: Iterate through each bar size bracket for the current stock
    for BS in "${BAR_SIZES[@]}"; do
        echo "Processing [${SYMBOL}] at bar size [${BS}]..."
        
        # Check if symbol is in CONTEXT safely
        if [[ " ${CONTEXT[*]} " =~ " ${lower_symbol} " ]]; then
            echo "Match found: '$SYMBOL' is in CONTEXT!"
            ## uncomment when bar size is less than 1day
            python insertion_process_single.py  -s "$SYMBOL" -bs "$BS" --context
            python single_item_feature_engineering_wip.py   -s "$SYMBOL" -bs "$BS" --context
        else
            echo "'$SYMBOL' is NOT in CONTEXT."
            python insertion_process_single.py  -s "$SYMBOL" -bs "$BS"
            python single_item_feature_engineering_wip.py   -s "$SYMBOL" -bs "$BS"
        fi
        
        # Error tracking fallback check
        if [ $? -ne 0 ]; then
            echo "Warning: Python crashed on ${SYMBOL} for bar size '${BS}'."
        else
            echo "Success: Done with ${SYMBOL} (${BS})."
        fi
        echo "------------------------------------------------------------------------"
    done
    
done

echo "######### removing audit files #####################"
rm *_audit*
echo "========================================================================"
echo "All stock assets and bar size matrices processed completely."
echo "========================================================================"