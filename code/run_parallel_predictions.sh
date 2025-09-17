#!/bin/bash
# Simple Parallel Policy Prediction Runner

# Configuration
MODEL="gpt-4o"
PROMPT_TYPE="trustee_ls"
PROMPT_NUM=0
N_USERS=""  # Leave empty for all users, or set to a number
MAX_PARALLEL=4  # Maximum number of parallel processes

# Policies to run (modify this list as needed)
POLICIES=(1 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
#POLICIES=(0 1 2 3)
# Create logs directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "Starting parallel policy predictions..."
echo "Model: $MODEL, Prompt: $PROMPT_TYPE, Max parallel: $MAX_PARALLEL"
echo "Policies: ${POLICIES[*]}"
echo "Logs will be saved to: $LOG_DIR/"

# Process policies in batches of MAX_PARALLEL
for ((i=0; i<${#POLICIES[@]}; i+=MAX_PARALLEL)); do
    # Start batch of policies
    for ((j=0; j<MAX_PARALLEL && i+j<${#POLICIES[@]}; j++)); do
        policy=${POLICIES[i+j]}
        echo "Starting policy $policy..."

        # Build command
        cmd="python predict_policy.py --policy $policy --prompt-type $PROMPT_TYPE --model $MODEL --prompt-num $PROMPT_NUM"
        if [ -n "$N_USERS" ]; then
            cmd="$cmd --n-users $N_USERS"
        fi

        # Run in background, redirect output to log file in logs directory
        $cmd > "$LOG_DIR/policy_${policy}.log" 2>&1 &
    done

    echo "Batch started. Waiting for this batch to complete..."

    # Wait for current batch to finish before starting next batch
    wait

    echo "Batch completed."
done

echo "All policies completed! Check $LOG_DIR/policy_*.log files for details."