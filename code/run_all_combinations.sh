#!/bin/bash
# Master script to run all prompt type/model combinations
# Calls run_parallel_predictions.sh with all combinations of:
# - delegate (prompts 0-4) x models (gpt-4o, claude-3-sonnet-v2)
# - trustee_ls (prompts 0-2) x models (gpt-4o, claude-3-sonnet-v2)
# - trustee_lsd (prompts 0-2) x models (gpt-4o, claude-3-sonnet-v2)

# Configuration
MODELS=("claude-3-haiku-v2-mini" "gpt-4o-mini")
DELEGATE_PROMPTS=(3 4)
TRUSTEE_LS_PROMPTS=()
TRUSTEE_LSD_PROMPTS=()

# Options
DRY_RUN=false
RESUME=false
SKIP_EXISTING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dry-run               Show what would be run without executing"
            echo "  --resume                Resume from where we left off"
            echo "  --skip-existing         Skip combinations that already have log files"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Create master log directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG_DIR="logs/all_combinations_$TIMESTAMP"
PROGRESS_FILE="$MASTER_LOG_DIR/progress.txt"
SUMMARY_FILE="$MASTER_LOG_DIR/summary.txt"

if [ "$DRY_RUN" = false ]; then
    mkdir -p "$MASTER_LOG_DIR"
    echo "Master run started at $(date)" > "$SUMMARY_FILE"
    echo "0" > "$PROGRESS_FILE"
fi

# Function to check if a combination has already been completed
check_existing() {
    local model=$1
    local prompt_type=$2
    local prompt_num=$3

    # Check if there are any log files for this combination in recent runs
    if [ "$SKIP_EXISTING" = true ]; then
        find logs -name "all_combinations_*" -type d -exec test -f {}/policy_*_${model}_${prompt_type}_${prompt_num}.log \; -print -quit 2>/dev/null | grep -q .
        return $?
    fi
    return 1  # Don't skip by default
}

# Function to run a single combination
run_combination() {
    local model=$1
    local prompt_type=$2
    local prompt_num=$3
    local current=$4
    local total=$5

    echo "========================================="
    echo "[$current/$total] Running: $model + $prompt_type + prompt_$prompt_num"
    echo "Time: $(date)"

    if check_existing "$model" "$prompt_type" "$prompt_num"; then
        echo "SKIPPED (already exists)"
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would execute:"
        echo "  ./run_parallel_predictions.sh --model $model --prompt-type $prompt_type --prompt-num $prompt_num"
        return 0
    fi

    # Record start time
    start_time=$(date +%s)

    # Run the combination
    echo "Executing: ./run_parallel_predictions.sh --model $model --prompt-type $prompt_type --prompt-num $prompt_num"
    if ./run_parallel_predictions.sh --model "$model" --prompt-type "$prompt_type" --prompt-num "$prompt_num"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "SUCCESS: Completed in ${duration}s"
        echo "[$current/$total] SUCCESS: $model + $prompt_type + prompt_$prompt_num (${duration}s)" >> "$SUMMARY_FILE"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "ERROR: Failed after ${duration}s"
        echo "[$current/$total] ERROR: $model + $prompt_type + prompt_$prompt_num (${duration}s)" >> "$SUMMARY_FILE"
        return 1
    fi
}

# Calculate total combinations
total_combinations=0
for model in "${MODELS[@]}"; do
    total_combinations=$((total_combinations + ${#DELEGATE_PROMPTS[@]}))
    total_combinations=$((total_combinations + ${#TRUSTEE_LS_PROMPTS[@]}))
    total_combinations=$((total_combinations + ${#TRUSTEE_LSD_PROMPTS[@]}))
done

echo "========================================="
echo "Master Script: Run All Combinations"
echo "========================================="
echo "Total combinations to run: $total_combinations"
echo "Models: ${MODELS[*]}"
echo "Delegate prompts: ${DELEGATE_PROMPTS[*]}"
echo "Trustee LS prompts: ${TRUSTEE_LS_PROMPTS[*]}"
echo "Trustee LSD prompts: ${TRUSTEE_LSD_PROMPTS[*]}"
echo "Master log directory: $MASTER_LOG_DIR"
echo "Dry run: $DRY_RUN"
echo "Resume: $RESUME"
echo "Skip existing: $SKIP_EXISTING"
echo "========================================="

# Get starting position for resume
start_position=0
if [ "$RESUME" = true ] && [ -f "$PROGRESS_FILE" ]; then
    start_position=$(cat "$PROGRESS_FILE")
    echo "Resuming from combination $start_position"
fi

# Track overall progress
current_combination=0
successful_runs=0
failed_runs=0
skipped_runs=0

# Record overall start time
overall_start_time=$(date +%s)

# Run all combinations
for model in "${MODELS[@]}"; do
    # Delegate combinations
    for prompt_num in "${DELEGATE_PROMPTS[@]}"; do
        current_combination=$((current_combination + 1))

        if [ $current_combination -le $start_position ]; then
            echo "[$current_combination/$total_combinations] SKIPPING (resume): $model + delegate + prompt_$prompt_num"
            continue
        fi

        if run_combination "$model" "delegate" "$prompt_num" "$current_combination" "$total_combinations"; then
            successful_runs=$((successful_runs + 1))
        else
            failed_runs=$((failed_runs + 1))
        fi

        # Update progress
        if [ "$DRY_RUN" = false ]; then
            echo "$current_combination" > "$PROGRESS_FILE"
        fi
    done

    # Trustee LS combinations
    for prompt_num in "${TRUSTEE_LS_PROMPTS[@]}"; do
        current_combination=$((current_combination + 1))

        if [ $current_combination -le $start_position ]; then
            echo "[$current_combination/$total_combinations] SKIPPING (resume): $model + trustee_ls + prompt_$prompt_num"
            continue
        fi

        if run_combination "$model" "trustee_ls" "$prompt_num" "$current_combination" "$total_combinations"; then
            successful_runs=$((successful_runs + 1))
        else
            failed_runs=$((failed_runs + 1))
        fi

        # Update progress
        if [ "$DRY_RUN" = false ]; then
            echo "$current_combination" > "$PROGRESS_FILE"
        fi
    done

    # Trustee LSD combinations
    for prompt_num in "${TRUSTEE_LSD_PROMPTS[@]}"; do
        current_combination=$((current_combination + 1))

        if [ $current_combination -le $start_position ]; then
            echo "[$current_combination/$total_combinations] SKIPPING (resume): $model + trustee_lsd + prompt_$prompt_num"
            continue
        fi

        if run_combination "$model" "trustee_lsd" "$prompt_num" "$current_combination" "$total_combinations"; then
            successful_runs=$((successful_runs + 1))
        else
            failed_runs=$((failed_runs + 1))
        fi

        # Update progress
        if [ "$DRY_RUN" = false ]; then
            echo "$current_combination" > "$PROGRESS_FILE"
        fi
    done
done

# Calculate overall duration
overall_end_time=$(date +%s)
overall_duration=$((overall_end_time - overall_start_time))

# Final summary
echo "========================================="
echo "FINAL SUMMARY"
echo "========================================="
echo "Total combinations: $total_combinations"
echo "Successful: $successful_runs"
echo "Failed: $failed_runs"
echo "Skipped: $skipped_runs"
echo "Total duration: ${overall_duration}s ($(($overall_duration / 60))m $(($overall_duration % 60))s)"
echo "Master log directory: $MASTER_LOG_DIR"

if [ "$DRY_RUN" = false ]; then
    echo "" >> "$SUMMARY_FILE"
    echo "FINAL SUMMARY:" >> "$SUMMARY_FILE"
    echo "Total combinations: $total_combinations" >> "$SUMMARY_FILE"
    echo "Successful: $successful_runs" >> "$SUMMARY_FILE"
    echo "Failed: $failed_runs" >> "$SUMMARY_FILE"
    echo "Skipped: $skipped_runs" >> "$SUMMARY_FILE"
    echo "Total duration: ${overall_duration}s" >> "$SUMMARY_FILE"
    echo "Completed at: $(date)" >> "$SUMMARY_FILE"
fi

if [ $failed_runs -gt 0 ]; then
    echo "WARNING: $failed_runs combinations failed. Check $SUMMARY_FILE for details."
    exit 1
else
    echo "All combinations completed successfully!"
    exit 0
fi