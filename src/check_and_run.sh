#!/bin/bash

PID=3839639

# Check if the GPU is still being used by the PID
while true; do
    # Check GPU usage for the specific PID
    if ! nvidia-smi | grep -q "$PID"; then
        echo "GPU is free, starting the next experiment."
        # Start your second experiment here, using nohup
        nohup python3 roberta_only.py > logs/experiments_roberta_7.log 2>&1 & 
        break
    else
        echo "Waiting for GPU to be free..."
        sleep 300  # Check every 5 minutes
    fi
done
