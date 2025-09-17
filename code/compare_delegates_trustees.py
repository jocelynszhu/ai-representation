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
def create_delegate_trustee_comparison(model, policy_index, long_term_weight, trustee_prompt_num=0, delegate_prompt_num=0):
    """
    Create a comparison DataFrame of delegate vs trustee votes for a given policy.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        long_term_weight (float): Weight for long-term utility (0-1)
        trustee_prompt_num (int): Prompt number (default 0)
        delegate_prompt_num (int): Prompt number (default 0)

    Returns:
        pd.DataFrame: Columns: participant_id, policy_id, delegate_vote, trustee_vote
    """
    import os   

    policy_id = policy_index + 1

    # File paths
    delegate_file = f"../data/delegate/{model}/prompt-{delegate_prompt_num}/d_policy_{policy_id}_votes.jsonl"
    trustee_file = f"../data/trustee_ls/{model}/prompt-{trustee_prompt_num}/t_policy_{policy_id}_votes.jsonl"

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
# Multi-Delegate Prompt Comparison Function
def plot_disagreement_by_delegate_prompts(model, policy_index, delegate_prompt_nums, trustee_prompt_num=0, show_plot=True):
    """
    Compare disagreement patterns across different delegate prompts paired with the same trustee prompt.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        delegate_prompt_nums (list): List of delegate prompt numbers to compare
        trustee_prompt_num (int): Trustee prompt number to use for all comparisons
        show_plot (bool): Whether to display the plot (default True)

    Returns:
        dict: Results for each delegate prompt {prompt_num: results_df}
    """
    # Generate weight range from 0.0 to 1.0 in steps of 0.01 (fine granularity)
    weights = np.arange(0.0, 1.01, 0.1)
    all_results = {}

    print(f"Comparing {len(delegate_prompt_nums)} delegate prompts against trustee prompt {trustee_prompt_num}")
    print(f"Analyzing {len(weights)} weight points for each prompt pair...")
    print("=" * 80)

    # Process each delegate prompt
    for delegate_prompt_num in delegate_prompt_nums:
        print(f"\nProcessing delegate prompt {delegate_prompt_num}...")
        disagreement_rates = []

        for i, weight in enumerate(weights):
            try:
                # Get comparison data for this weight and prompt pair
                comparison_df = create_delegate_trustee_comparison(
                    model, policy_index, weight, trustee_prompt_num, delegate_prompt_num
                )

                # Calculate disagreement rate
                disagreements = comparison_df['delegate_vote'] != comparison_df['trustee_vote']
                disagreement_rate = disagreements.sum() / len(comparison_df)
                disagreement_rates.append(disagreement_rate)

                # Progress indicator every 20 weight points
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(weights)} weights processed ({(i+1)/len(weights)*100:.0f}%)")

            except Exception as e:
                print(f"  Error at weight {weight:.2f}: {e}")
                disagreement_rates.append(np.nan)

        # Store results for this delegate prompt
        results_df = pd.DataFrame({
            'long_term_weight': weights,
            'disagreement_rate': disagreement_rates,
            'delegate_prompt': delegate_prompt_num
        })
        all_results[delegate_prompt_num] = results_df

        # Summary stats for this prompt
        valid_rates = [r for r in disagreement_rates if not np.isnan(r)]
        if valid_rates:
            print(f"  Delegate prompt {delegate_prompt_num}: min={min(valid_rates):.1%}, max={max(valid_rates):.1%}, avg={np.mean(valid_rates):.1%}")

    # Create plot
    if show_plot:
        plt.figure(figsize=(12, 8))

        # Define colors and line styles for different prompts
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        line_styles = ['--', '--', '--', '--']

        # Plot each delegate prompt as a separate line
        mean_disagreement_rates = []
        for i, delegate_prompt_num in enumerate(delegate_prompt_nums):
            results_df = all_results[delegate_prompt_num]
            color = colors[i % len(colors)]
            line_style = line_styles[i % len(line_styles)]

            plt.plot(results_df['long_term_weight'], results_df['disagreement_rate'],
                    color=color, linestyle=line_style, linewidth=2,
                    markersize=8,
                    marker='o',
                    label=f'Delegate Prompt {delegate_prompt_num}', alpha=0.5)

            # Store for mean calculation
            mean_disagreement_rates.append(results_df['disagreement_rate'].values)

        # Calculate and plot mean across all delegate prompts
        mean_disagreement_rates = np.array(mean_disagreement_rates)
        mean_across_prompts = np.nanmean(mean_disagreement_rates, axis=0)

        plt.plot(weights, mean_across_prompts,
                color='black', linewidth=5, label='Mean Across All Prompts', alpha=0.9)

        plt.xlabel('Long-term Weight', fontsize=12)
        plt.ylabel('Disagreement Rate', fontsize=12)
        plt.title(f'Disagreement Patterns by Delegate Prompt Type\\n{model}, Policy {policy_index + 1}, Trustee Prompt {trustee_prompt_num}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0,.5)  # Set y-axis range from 15% to 40%
        plt.xlim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()
        plt.show()

        # Print summary comparison
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON:")
        print("=" * 80)
        for delegate_prompt_num in delegate_prompt_nums:
            results_df = all_results[delegate_prompt_num]
            valid_rates = results_df['disagreement_rate'].dropna()
            if len(valid_rates) > 0:
                min_disagree = valid_rates.min()
                max_disagree = valid_rates.max()
                avg_disagree = valid_rates.mean()
                min_weight = results_df.loc[results_df['disagreement_rate'] == min_disagree, 'long_term_weight'].iloc[0]
                max_weight = results_df.loc[results_df['disagreement_rate'] == max_disagree, 'long_term_weight'].iloc[0]

                print(f"Delegate Prompt {delegate_prompt_num}:")
                print(f"  Min disagreement: {min_disagree:.1%} (at long-term weight {min_weight:.2f})")
                print(f"  Max disagreement: {max_disagree:.1%} (at long-term weight {max_weight:.2f})")
                print(f"  Average disagreement: {avg_disagree:.1%}")
                print(f"  Range: {max_disagree - min_disagree:.1%}")
                print()

    return all_results

# %%
# All Policies Overview Function
def plot_all_policies_overview(model, policies_list, delegate_prompt_nums, trustee_prompt_num=0, show_plot=True):
    """
    Create an overview plot showing all policies and delegate prompts on a single plot.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policies_list (list): List of policy indices to analyze (0-based)
        delegate_prompt_nums (list): List of delegate prompt numbers to compare
        trustee_prompt_num (int): Trustee prompt number to use for all comparisons
        show_plot (bool): Whether to display the plot (default True)

    Returns:
        dict: Aggregated results including all individual curves and overall mean
    """
    weights = np.arange(0.0, 1.01, 0.1)
    all_curves = []
    successful_combinations = []

    print(f"Generating overview plot for {len(policies_list)} policies and {len(delegate_prompt_nums)} delegate prompts")
    print(f"Model: {model}, Trustee prompt: {trustee_prompt_num}")
    print("=" * 80)

    # Collect data from all policy-delegate prompt combinations
    for policy_index in policies_list:
        for delegate_prompt_num in delegate_prompt_nums:
            try:
                print(f"Processing policy {policy_index + 1}, delegate prompt {delegate_prompt_num}...", end=" ")

                disagreement_rates = []
                for weight in weights:
                    try:
                        # Get comparison data for this weight and prompt pair
                        comparison_df = create_delegate_trustee_comparison(
                            model, policy_index, weight, trustee_prompt_num, delegate_prompt_num
                        )

                        # Calculate disagreement rate
                        disagreements = comparison_df['delegate_vote'] != comparison_df['trustee_vote']
                        disagreement_rate = disagreements.sum() / len(comparison_df)
                        disagreement_rates.append(disagreement_rate)

                    except Exception as e:
                        disagreement_rates.append(np.nan)

                # Store successful curve
                all_curves.append(np.array(disagreement_rates))
                successful_combinations.append((policy_index, delegate_prompt_num))
                print("✓")

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

    print(f"\nSuccessfully processed {len(all_curves)} policy-delegate combinations")

    if not all_curves:
        print("⚠ No data available for overview plot")
        return {}

    # Convert to numpy array for easier manipulation
    all_curves = np.array(all_curves)

    # Calculate overall mean (ignoring NaN values)
    overall_mean = np.nanmean(all_curves, axis=0)

    # Create plot if requested
    if show_plot:
        plt.figure(figsize=(12, 8))

        # Plot individual curves as very light red lines
        for i, curve in enumerate(all_curves):
            policy_idx, delegate_idx = successful_combinations[i]
            plt.plot(weights, curve, color='#ff9999', alpha=0.2, linewidth=0.5)

        # Plot overall mean as thick black line
        plt.plot(weights, overall_mean, color='black', linewidth=4, label='Overall Mean', alpha=0.9, markersize=15, marker='o')

        # Add a single legend entry for individual lines
        plt.plot([], [], color='#ff9999', alpha=0.2, linewidth=0.5, label='Individual Policy-Prompt Combinations')

        # Format plot
        plt.xlabel('Long-term Weight', fontsize=12)
        plt.ylabel('Disagreement Rate', fontsize=12)
        plt.title(f'Disagreement Patterns Overview - All Policies and Delegate Prompts\n{model}, Trustee Prompt {trustee_prompt_num}, {len(all_curves)} combinations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 0.5)
        plt.xlim(0, 1)
        plt.legend(loc='upper right', fontsize=10)

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()
        plt.show()

    # Return results
    return {
        'weights': weights,
        'all_curves': all_curves,
        'overall_mean': overall_mean,
        'successful_combinations': successful_combinations,
        'num_combinations': len(all_curves)
    }

# %%
# Example: Compare multiple delegate prompts
# results = plot_disagreement_by_delegate_prompts("claude-3-sonnet-v2", policy_index,
#                                                delegate_prompt_nums=[0, 1, 2, 3],
#                                                trustee_prompt_num=0)


# %%