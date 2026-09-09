#!/bin/bash

# Run FENCE on all four datasets in test mode (5k samples)

echo "=========================================="
echo "FENCE - Testing All Datasets (5k samples)"
echo "=========================================="

# Set environment
export PYTHONPATH=$PYTHONPATH:$(pwd)/code

# Run each dataset
for dataset in fasdd dfire ustc_smokers flame; do
    echo ""
    echo "=========================================="
    echo "Testing dataset: $dataset"
    echo "=========================================="
    
    python code/fence/main.py --dataset $dataset --test_mode --no_wandb
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed on $dataset"
        exit 1
    fi
    
    echo "✅ Completed $dataset"
done

echo ""
echo "=========================================="
echo "✅ All datasets tested successfully!"
echo "=========================================="