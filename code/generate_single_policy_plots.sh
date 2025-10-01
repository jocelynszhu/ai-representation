#!/bin/bash

# Define policies to generate plots for (default: all 30 policies)
POLICIES=($(seq 0 29))

# You can also manually specify specific policies like:
# POLICIES=(0 1 2 5 10)

echo "Generating single policy plots for ${#POLICIES[@]} policies..."

# Loop over each policy
for policy in "${POLICIES[@]}"; do
    echo "Generating plot for policy $policy..."
    python3 -c "
from facet_agreement_plot import create_facet_agreement_plot

create_facet_agreement_plot(
    policy_indices=[$policy],
    delegate_prompt_nums=[0, 1, 2, 3, 4],
    trustee_prompt_nums=[0, 1, 2],
    figsize=(12, 8),
    save_path='dummy'  # Path will be auto-generated for single policy
)
"
done

echo "Done! All plots saved to ../data/plots/line_agreement_plots/"
