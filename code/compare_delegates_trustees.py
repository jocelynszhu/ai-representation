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
        long_term_weight (float): Weight for long-term utility, short-term gets (1-long_term_weight)

    Returns:
        dict: Contains weighted scores and final vote decision
    """
    # Allow any long_term_weight value for flexibility

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
# Exponential Discounting Function for Time-Based Utilities
def calculate_discounted_vote(parsed_response, sigma):
    """
    Calculate exponentially discounted utility scores and determine vote.

    Args:
        parsed_response (dict): Parsed JSON response with time-based utilities
        sigma (float): Discount factor, where 0 = only immediate matters, 1 = no discounting, >1 = anti-discounting (future matters more)

    Returns:
        dict: Contains discounted scores, individual period contributions, and final vote decision
    """
    # Input validation - allow any sigma value for flexibility

    # Define time periods in order
    time_periods = ["0-5 years", "5-10 years", "10-15 years", "15-20 years", "20-25 years", "25-30 years"]

    # Initialize totals and detailed breakdowns
    yes_discounted_total = 0
    no_discounted_total = 0
    yes_period_details = []
    no_period_details = []

    # Process each time period
    for t, period in enumerate(time_periods):
        try:
            # Extract scores for this period
            yes_score = parsed_response["yes"][period]["score"]
            no_score = parsed_response["no"][period]["score"]

            # Calculate discount factor for this time period
            discount_factor = sigma ** t

            # Apply exponential discounting
            yes_discounted = yes_score * discount_factor
            no_discounted = no_score * discount_factor

            # Accumulate totals
            yes_discounted_total += yes_discounted
            no_discounted_total += no_discounted

            # Store detailed breakdown for analysis
            yes_period_details.append({
                "period": period,
                "raw_score": yes_score,
                "discount_factor": discount_factor,
                "discounted_score": yes_discounted
            })

            no_period_details.append({
                "period": period,
                "raw_score": no_score,
                "discount_factor": discount_factor,
                "discounted_score": no_discounted
            })

        except KeyError as e:
            raise ValueError(f"Missing required field in parsed_response for period '{period}': {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid score value in period '{period}': {e}")

    # Determine vote based on discounted utilities
    if yes_discounted_total > no_discounted_total:
        vote = "Yes"
    elif no_discounted_total > yes_discounted_total:
        vote = "No"
    else:
        vote = "Neutral"

    return {
        "yes_discounted_utility": yes_discounted_total,
        "no_discounted_utility": no_discounted_total,
        "vote": vote,
        "sigma": sigma,
        "yes_period_breakdown": yes_period_details,
        "no_period_breakdown": no_period_details
    }

# %%
# Delegate-Trustee Comparison Function
def create_delegate_trustee_comparison(model, policy_index, weight_param, trustee_prompt_num=0, delegate_prompt_num=0, trustee_format='trustee_ls'):
    """
    Create a comparison DataFrame of delegate vs trustee votes for a given policy.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        weight_param (float): Weight parameter - used as long_term_weight for ls format or sigma for lsd format
        trustee_prompt_num (int): Prompt number (default 0)
        delegate_prompt_num (int): Prompt number (default 0)
        trustee_format (str): Trustee data format ('trustee_ls' or 'trustee_lsd')

    Returns:
        pd.DataFrame: Columns: participant_id, policy_id, delegate_vote, trustee_vote
    """
    import os   

    policy_id = policy_index + 1

    # File paths
    delegate_file = f"../data/delegate/{model}/prompt-{delegate_prompt_num}/d_policy_{policy_id}_votes.jsonl"
    trustee_file = f"../data/{trustee_format}/{model}/prompt-{trustee_prompt_num}/t_policy_{policy_id}_votes.jsonl"

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

                # Use appropriate calculation function based on format
                if trustee_format == 'trustee_ls':
                    vote_result = calculate_weighted_vote(data, weight_param)
                elif trustee_format == 'trustee_lsd':
                    vote_result = calculate_discounted_vote(data, weight_param)
                else:
                    raise ValueError(f"Unsupported trustee format: {trustee_format}")

                trustee_data.append({
                    'participant_id': data['id'],
                    'trustee_vote': vote_result['vote']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing trustee data: {e}")
                continue
            except Exception as e:
                print(f"Error calculating vote for participant {data.get('id', 'unknown')}: {e}")
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
def plot_disagreement_by_delegate_prompts(model, policy_index, delegate_prompt_nums, trustee_prompt_num=0, trustee_format='trustee_ls', show_plot=True):
    """
    Compare disagreement patterns across different delegate prompts paired with the same trustee prompt.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        delegate_prompt_nums (list): List of delegate prompt numbers to compare
        trustee_prompt_num (int): Trustee prompt number to use for all comparisons
        trustee_format (str): Trustee data format ('trustee_ls' or 'trustee_lsd')
        show_plot (bool): Whether to display the plot (default True)

    Returns:
        dict: Results for each delegate prompt {prompt_num: results_df}
    """
    # Generate weight range from 0.0 to 2.0 in steps of 0.1 (extended range for anti-discounting)
    weights = np.arange(0.0, 2.01, 0.1)
    all_results = {}

    print(f"Comparing {len(delegate_prompt_nums)} delegate prompts against trustee prompt {trustee_prompt_num}")
    print(f"Analyzing {len(weights)} weight points for each prompt pair...")
    print("=" * 80)

    # Calculate trustee policy support (same for all delegate prompts, so do it once)
    print(f"\nCalculating trustee policy support across {len(weights)} weight points...")
    trustee_support_rates = []

    for i, weight in enumerate(weights):
        try:
            # Get comparison data for this weight (using first delegate prompt for trustee data)
            comparison_df = create_delegate_trustee_comparison(
                model, policy_index, weight, trustee_prompt_num, delegate_prompt_nums[0], trustee_format
            )

            # Calculate trustee policy support rate (proportion voting "Yes")
            trustee_yes_votes = (comparison_df['trustee_vote'] == 'Yes').sum()
            trustee_support_rate = trustee_yes_votes / len(comparison_df)
            trustee_support_rates.append(trustee_support_rate)

        except Exception as e:
            print(f"  Error calculating trustee support at weight {weight:.2f}: {e}")
            trustee_support_rates.append(np.nan)

    # Process each delegate prompt
    for delegate_prompt_num in delegate_prompt_nums:
        print(f"\nProcessing delegate prompt {delegate_prompt_num}...")
        disagreement_rates = []

        for i, weight in enumerate(weights):
            try:
                # Get comparison data for this weight and prompt pair
                comparison_df = create_delegate_trustee_comparison(
                    model, policy_index, weight, trustee_prompt_num, delegate_prompt_num, trustee_format
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
                color='black', linewidth=5, label='Mean Disagreement Rate', alpha=0.9)

        # Add trustee policy support line
        plt.plot(weights, trustee_support_rates,
                color='purple', linewidth=3, linestyle='-',
                marker='s', markersize=6, label='Trustee Policy Support', alpha=0.8)

        plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title(f'Disagreement Patterns & Trustee Policy Support\\n{model}, Policy {policy_index + 1}, Trustee Prompt {trustee_prompt_num}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)  # Set y-axis range from 0% to 100% to accommodate both metrics
        plt.xlim(0, 2)
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

    # Add trustee support data to results
    all_results['trustee_support'] = {
        'weights': weights,
        'support_rates': trustee_support_rates
    }

    return all_results

# %%
# All Policies Overview Function
def plot_all_policies_overview(model, policies_list, delegate_prompt_nums, trustee_prompt_num=0, trustee_format='trustee_ls', show_plot=True):
    """
    Create an overview plot showing all policies and delegate prompts on a single plot.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policies_list (list): List of policy indices to analyze (0-based)
        delegate_prompt_nums (list): List of delegate prompt numbers to compare
        trustee_prompt_num (int): Trustee prompt number to use for all comparisons
        trustee_format (str): Trustee data format ('trustee_ls' or 'trustee_lsd')
        show_plot (bool): Whether to display the plot (default True)

    Returns:
        dict: Aggregated results including all individual curves and overall mean
    """
    weights = np.arange(0.0, 2.01, 0.1)
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
                            model, policy_index, weight, trustee_prompt_num, delegate_prompt_num, trustee_format
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
        plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=12)
        plt.ylabel('Disagreement Rate', fontsize=12)
        plt.title(f'Disagreement Patterns Overview - All Policies and Delegate Prompts\n{model}, Trustee Prompt {trustee_prompt_num}, {len(all_curves)} combinations', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 0.5)
        plt.xlim(0, 2)
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