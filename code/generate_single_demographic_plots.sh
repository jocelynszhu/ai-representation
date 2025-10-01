#!/bin/bash

# Define policies to generate plots for (default: all 30 policies)
POLICIES=($(seq 0 29))

# You can also manually specify specific policies like:
# POLICIES=(0 1 2 5 10)

echo "Generating single policy demographic plots for ${#POLICIES[@]} policies..."

# Loop over each policy
for policy in "${POLICIES[@]}"; do
    echo "Generating demographic plot for policy $policy..."
    python3 -c "
import pandas as pd
from demographic_agreement_combined_plot import create_combined_demographic_plot

# Load biographies
bio_df = pd.read_json('rep_biographies.jsonl', lines=True)
bio_df.rename(columns={'ID': 'participant_id'}, inplace=True)

# Load policy data to check for expert consensus
policies_df = pd.read_json('../self_selected_policies_new.jsonl', lines=True)
policy_data = policies_df.iloc[$policy]
has_expert_vote = pd.notna(policy_data.get('expert_vote'))

# Determine which list to use
if has_expert_vote:
    expert_consensus_policies = [$policy]
    no_consensus_policies = []
else:
    expert_consensus_policies = []
    no_consensus_policies = [$policy]

create_combined_demographic_plot(
    expert_consensus_policies=expert_consensus_policies,
    no_consensus_policies=no_consensus_policies,
    delegate_prompt_nums=[0, 1, 2, 3, 4],
    trustee_prompt_nums=[0, 1, 2],
    trustee_type='both',
    bio_df=bio_df,
    demographics=['Political Affiliation', 'Income'],
    alpha=1.0,
    output_file='dummy'  # Path will be auto-generated for single policy
)
"
done

echo "Done! All plots saved to ../data/plots/demographic_agreement_plots/"
