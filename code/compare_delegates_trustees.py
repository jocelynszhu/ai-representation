# %%
# Delegate vs Trustee Comparison and Analysis
# Load existing prediction data and analyze differences between delegate and trustee voting patterns

# %%
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
# Configuration
policy_index = 0  # Which policy to analyze (0-based index)

# %%
# Load Data

# Load prompts (for reference)
with open('../prompts_long_short.json', 'r') as f:
    prompts = json.load(f)

# Load user profiles (for reference)
written_profiles = pd.read_json("gpt-4o_written_profiles.jsonl", encoding='cp1252', lines=True)

# Load policies (for reference)
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

print(f"Loaded {len(prompts)} prompt sets")
print(f"Loaded {len(written_profiles)} user profiles")
print(f"Loaded {len(policies)} policies")

# %%
# Utility Weighting Function (shared from predict_utilities.py)
def calculate_weighted_vote(parsed_response, long_term_weight):
    """
    Calculate weighted utility scores and determine vote.

    Args:
        parsed_response (dict): Parsed JSON response with yes_vote/no_vote utilities
        long_term_weight (float): Weight for long-term utility (0-1), short-term gets (1-long_term_weight)

    Returns:
        dict: Contains weighted scores and final vote decision
    """
    if long_term_weight < 0 or long_term_weight > 1:
        raise ValueError("long_term_weight must be between 0 and 1")

    short_term_weight = 1 - long_term_weight

    # Calculate weighted utility for YES vote
    yes_weighted = (parsed_response["yes_vote"]["short_util"] * short_term_weight +
                   parsed_response["yes_vote"]["long_util"] * long_term_weight)

    # Calculate weighted utility for NO vote
    no_weighted = (parsed_response["no_vote"]["short_util"] * short_term_weight +
                  parsed_response["no_vote"]["long_util"] * long_term_weight)

    # Determine vote
    if yes_weighted > no_weighted:
        vote = "Yes"
    elif no_weighted > yes_weighted:
        vote = "No"
    else:
        vote = "Neutral"

    return {
        "yes_weighted_utility": yes_weighted,
        "no_weighted_utility": no_weighted,
        "vote": vote,
        "short_term_weight": short_term_weight,
        "long_term_weight": long_term_weight
    }

# %%
# Delegate-Trustee Comparison Function
def create_delegate_trustee_comparison(model, policy_index, long_term_weight, prompt_num=0):
    """
    Create a comparison DataFrame of delegate vs trustee votes for a given policy.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        long_term_weight (float): Weight for long-term utility (0-1)
        prompt_num (int): Prompt number (default 0)

    Returns:
        pd.DataFrame: Columns: participant_id, policy_id, delegate_vote, trustee_vote
    """
    import os

    policy_id = policy_index + 1

    # File paths
    delegate_file = f"../data/delegate/{model}/prompt-{prompt_num}/d_policy_{policy_id}_votes.jsonl"
    trustee_file = f"../data/trustee_ls/{model}/prompt-{prompt_num}/t_policy_{policy_id}_votes.jsonl"

    # Check if files exist
    if not os.path.exists(delegate_file):
        raise FileNotFoundError(f"Delegate file not found: {delegate_file}")
    if not os.path.exists(trustee_file):
        raise FileNotFoundError(f"Trustee file not found: {trustee_file}")

    # Load delegate data
    delegate_data = []
    with open(delegate_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                delegate_data.append({
                    'participant_id': data['id'],
                    'delegate_vote': data['vote']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing delegate data: {e}")
                continue

    delegate_df = pd.DataFrame(delegate_data)

    # Load and process trustee data
    trustee_data = []
    with open(trustee_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # Use calculate_weighted_vote to get the final vote
                vote_result = calculate_weighted_vote(data, long_term_weight)

                trustee_data.append({
                    'participant_id': data['id'],
                    'trustee_vote': vote_result['vote']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing trustee data: {e}")
                continue
            except Exception as e:
                print(f"Error calculating weighted vote for participant {data.get('id', 'unknown')}: {e}")
                continue

    trustee_df = pd.DataFrame(trustee_data)

    # Merge the dataframes
    comparison_df = pd.merge(delegate_df, trustee_df, on='participant_id', how='inner')

    # Add policy_id column
    comparison_df['policy_id'] = policy_id

    # Reorder columns
    comparison_df = comparison_df[['participant_id', 'policy_id', 'delegate_vote', 'trustee_vote']]

    # Sort by participant_id
    comparison_df = comparison_df.sort_values('participant_id').reset_index(drop=True)

    print(f"Loaded {len(delegate_df)} delegate votes and {len(trustee_df)} trustee votes")
    print(f"Final comparison DataFrame: {len(comparison_df)} participants")

    return comparison_df

# %%
# Disagreement Rate Analysis Function
def plot_disagreement_by_weight(model, policy_index, prompt_num=0, show_plot=True):
    """
    Analyze disagreement rates between delegate and trustee votes across different long-term weights.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        prompt_num (int): Prompt number (default 0)
        show_plot (bool): Whether to display the plot (default True)

    Returns:
        pd.DataFrame: Contains weights and corresponding disagreement rates
    """
    # Generate weight range from 0.0 to 1.0 in steps of 0.1
    weights = np.arange(0.0, 1.1, 0.1)
    disagreement_rates = []

    print(f"Analyzing disagreement rates for {len(weights)} different long-term weights...")

    for weight in weights:
        try:
            # Get comparison data for this weight
            comparison_df = create_delegate_trustee_comparison(model, policy_index, weight, prompt_num)

            # Calculate disagreement rate
            disagreements = comparison_df['delegate_vote'] != comparison_df['trustee_vote']
            disagreement_rate = disagreements.sum() / len(comparison_df)
            disagreement_rates.append(disagreement_rate)

            print(f"Weight {weight:.1f}: {disagreement_rate:.3f} disagreement rate ({disagreements.sum()}/{len(comparison_df)} participants)")

        except Exception as e:
            print(f"Error at weight {weight:.1f}: {e}")
            disagreement_rates.append(np.nan)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'long_term_weight': weights,
        'disagreement_rate': disagreement_rates
    })

    # Create plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(weights, disagreement_rates, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Long-term Weight', fontsize=12)
        plt.ylabel('Disagreement Rate', fontsize=12)
        plt.title(f'Delegate vs Trustee Disagreement Rate by Long-term Weight\\n{model}, Policy {policy_index + 1}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add annotations for key points
        min_idx = np.nanargmin(disagreement_rates)
        max_idx = np.nanargmax(disagreement_rates)

        plt.annotate(f'Min: {disagreement_rates[min_idx]:.1%}',
                    xy=(weights[min_idx], disagreement_rates[min_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.annotate(f'Max: {disagreement_rates[max_idx]:.1%}',
                    xy=(weights[max_idx], disagreement_rates[max_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.show()

    return results_df

# %%
# Example Analysis: Compare delegate vs trustee for specific weight
comparison_df = create_delegate_trustee_comparison("claude-3-sonnet-v2", policy_index, 0.5, 0)
disagreements_df = comparison_df[comparison_df['delegate_vote'] != comparison_df['trustee_vote']]
print(f"Disagreement rate: {len(disagreements_df)/len(comparison_df):.1%}")

# %%
# Show the comparison data
comparison_df.head()

# %%
# Example: Plot disagreement rates across all weights
results = plot_disagreement_by_weight("claude-3-sonnet-v2", policy_index, prompt_num=0)

# %%