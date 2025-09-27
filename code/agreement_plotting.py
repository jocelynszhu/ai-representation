#%%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os

# Import utility functions
from vote_data_loader import load_policy_votes
from compare_delegates_trustees import calculate_weighted_vote, calculate_discounted_vote

#%%
def create_agreement_dataframe(
    policy_index: int,
    prompt_nums: List[int],
    model: str,
    trustee_type: str = "trustee_ls"
) -> pd.DataFrame:
    """
    Create a DataFrame with agreement rates across alpha/sigma parameters for different prompts.

    Args:
        policy_index (int): 0-based policy index
        prompt_nums (List[int]): List of prompt numbers to analyze
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        trustee_type (str): Either "trustee_ls" or "trustee_lsd"

    Returns:
        pd.DataFrame: DataFrame with alpha/sigma values and agreement rates for each prompt
    """
    # Parameter range from 0.0 to 1.0 in steps of 0.1
    alphas = np.arange(0.0, 1.01, 0.1)

    # Initialize result data structure
    result_data = {
        'alpha_sigma': alphas
    }

    print(f"Processing policy {policy_index + 1} with {len(prompt_nums)} prompts...")

    # Load default vote for this policy
    default_data = load_policy_votes(model, trustee_type, policy_index, prompt_nums[0])
    default_vote = default_data['defaults']['vote'].iloc[0] if len(default_data['defaults']) > 0 else "Unknown"
    result_data['default_vote'] = [default_vote] * len(alphas)

    print(f"Default model vote for policy {policy_index + 1}: {default_vote}")

    # Process each prompt
    for prompt_num in prompt_nums:
        print(f"  Processing prompt {prompt_num}...")

        try:
            # Load data for this prompt
            data = load_policy_votes(model, trustee_type, policy_index, prompt_num)

            # Calculate trustee agreement rates across alpha values
            trustee_agreements = []
            for alpha in alphas:
                agreement_rate = _calculate_trustee_agreement_rate(
                    data['trustee'], alpha, trustee_type, default_vote
                )
                trustee_agreements.append(agreement_rate)

            # Calculate delegate agreement rate (constant across alpha)
            delegate_agreement = _calculate_delegate_agreement_rate(
                data['delegate'], default_vote
            )
            delegate_agreements = [delegate_agreement] * len(alphas)

            # Store results
            result_data[f'trustee_prompt_{prompt_num}_agreement'] = trustee_agreements
            result_data[f'delegate_prompt_{prompt_num}_agreement'] = delegate_agreements

        except Exception as e:
            print(f"  Error processing prompt {prompt_num}: {e}")
            # Fill with NaN values
            result_data[f'trustee_prompt_{prompt_num}_agreement'] = [np.nan] * len(alphas)
            result_data[f'delegate_prompt_{prompt_num}_agreement'] = [np.nan] * len(alphas)

    # Create DataFrame
    df = pd.DataFrame(result_data)

    # Calculate mean agreement rates
    trustee_cols = [col for col in df.columns if col.startswith('trustee_prompt_')]
    delegate_cols = [col for col in df.columns if col.startswith('delegate_prompt_')]

    if trustee_cols:
        df['trustee_mean_agreement'] = df[trustee_cols].mean(axis=1)

    if delegate_cols:
        df['delegate_mean_agreement'] = df[delegate_cols].mean(axis=1)

    return df

#%%
def _calculate_trustee_agreement_rate(
    trustee_data: pd.DataFrame,
    alpha: float,
    trustee_type: str,
    default_vote: str
) -> float:
    """Calculate agreement rate between trustee votes and default vote for given alpha."""
    votes = []

    for _, row in trustee_data.iterrows():
        try:
            if trustee_type == "trustee_ls":
                # Reconstruct the entry format for calculate_weighted_vote
                entry = {
                    "yes_vote": {
                        "short_util": row['yes_short_util'],
                        "long_util": row['yes_long_util']
                    },
                    "no_vote": {
                        "short_util": row['no_short_util'],
                        "long_util": row['no_long_util']
                    }
                }
                vote_result = calculate_weighted_vote(entry, alpha)

            elif trustee_type == "trustee_lsd":
                # Reconstruct the entry format for calculate_discounted_vote
                periods = ["0_5_years", "5_10_years", "10_15_years", "15_20_years", "20_25_years", "25_30_years"]
                period_mapping = {
                    "0_5_years": "0-5 years",
                    "5_10_years": "5-10 years",
                    "10_15_years": "10-15 years",
                    "15_20_years": "15-20 years",
                    "20_25_years": "20-25 years",
                    "25_30_years": "25-30 years"
                }

                entry = {"yes": {}, "no": {}}
                for period in periods:
                    period_name = period_mapping[period]
                    entry["yes"][period_name] = {"score": row[f'yes_{period}_score']}
                    entry["no"][period_name] = {"score": row[f'no_{period}_score']}

                vote_result = calculate_discounted_vote(entry, alpha)

            else:
                raise ValueError(f"Unknown trustee_type: {trustee_type}")

            votes.append(vote_result['vote'])

        except Exception as e:
            continue

    if not votes:
        return np.nan

    # Calculate agreement rate with default vote
    agreements = sum(1 for vote in votes if vote == default_vote)
    return agreements / len(votes)

#%%
def _calculate_delegate_agreement_rate(
    delegate_data: pd.DataFrame,
    default_vote: str
) -> float:
    """Calculate agreement rate between delegate votes and default vote."""
    if len(delegate_data) == 0:
        return np.nan

    agreements = sum(1 for vote in delegate_data['vote'] if vote == default_vote)
    return agreements / len(delegate_data)

#%%
def plot_agreement_rates(
    agreement_df: pd.DataFrame,
    policy_index: int,
    trustee_type: str = "trustee_ls",
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create visualization of agreement rates vs alpha/sigma parameter.
    Shows all individual prompt lines (both delegate and trustee) plus their averages.

    Args:
        agreement_df (pd.DataFrame): DataFrame from create_agreement_dataframe()
        policy_index (int): For labeling
        trustee_type (str): For axis labeling ("Long-term Weight" vs "Sigma")
        show_plot (bool): Whether to display the plot
        figsize (Tuple[int, int]): Figure size
    """
    plt.figure(figsize=figsize)

    # Get data
    alphas = agreement_df['alpha_sigma']
    default_vote = agreement_df['default_vote'].iloc[0]

    # Get individual prompt columns
    trustee_cols = [col for col in agreement_df.columns if col.startswith('trustee_prompt_') and 'mean' not in col]
    delegate_cols = [col for col in agreement_df.columns if col.startswith('delegate_prompt_') and 'mean' not in col]

    # Plot individual trustee lines (lighter opacity)
    trustee_colors = ['blue', 'navy', 'steelblue', 'royalblue', 'cornflowerblue', 'mediumblue']
    for i, col in enumerate(trustee_cols):
        prompt_num = col.split('_')[2]  # Extract prompt number
        color = trustee_colors[i % len(trustee_colors)]
        plt.plot(alphas, agreement_df[col],
                color=color, linewidth=2, alpha=0.7, linestyle='-',
                label=f'Trustee Prompt {prompt_num}')

    # Plot individual delegate lines (lighter opacity)
    delegate_colors = ['red', 'darkred', 'crimson', 'lightcoral', 'indianred', 'firebrick']
    for i, col in enumerate(delegate_cols):
        prompt_num = col.split('_')[2]  # Extract prompt number
        color = delegate_colors[i % len(delegate_colors)]
        plt.plot(alphas, agreement_df[col],
                color=color, linewidth=2, alpha=0.7, linestyle='--',
                label=f'Delegate Prompt {prompt_num}')

    # Plot mean lines (dark, thick) - these will be on top
    if 'trustee_mean_agreement' in agreement_df.columns and len(trustee_cols) > 1:
        plt.plot(alphas, agreement_df['trustee_mean_agreement'],
                color='darkblue', linewidth=4, alpha=1.0, linestyle='-',
                label='Trustee Average', zorder=10)

    if 'delegate_mean_agreement' in agreement_df.columns and len(delegate_cols) > 1:
        plt.plot(alphas, agreement_df['delegate_mean_agreement'],
                color='darkred', linewidth=4, alpha=1.0, linestyle='--',
                label='Delegate Average', zorder=10)

    # Load policy statement for title
    try:
        policies_df = pd.read_json("../self_selected_policies_new.jsonl", lines=True)
        policy_statement = policies_df.iloc[policy_index]['statement']
        # Truncate if too long for display
        if len(policy_statement) > 100:
            policy_title = policy_statement[:97] + "..."
        else:
            policy_title = policy_statement
    except:
        policy_title = f'Policy {policy_index + 1}'

    # Formatting
    param_label = "Long-term Weight" if trustee_type == "trustee_ls" else "Sigma"
    plt.xlabel(f'{param_label}', fontsize=12)
    plt.ylabel('Agreement Rate with Default Vote', fontsize=12)
    plt.title(f'Agreement Rates vs {param_label}\\n{policy_title}\\nDefault Vote: {default_vote}',
              fontsize=13, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.xlim(0, 1)

    # Format y-axis as percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add legend with better organization
    handles, labels = plt.gca().get_legend_handles_labels()

    # Separate trustee and delegate entries
    trustee_entries = [(h, l) for h, l in zip(handles, labels) if 'Trustee' in l]
    delegate_entries = [(h, l) for h, l in zip(handles, labels) if 'Delegate' in l]

    # Reorganize: individual prompts first, then averages
    trustee_individual = [(h, l) for h, l in trustee_entries if 'Prompt' in l]
    trustee_avg = [(h, l) for h, l in trustee_entries if 'Average' in l]
    delegate_individual = [(h, l) for h, l in delegate_entries if 'Prompt' in l]
    delegate_avg = [(h, l) for h, l in delegate_entries if 'Average' in l]

    # Combine in order
    ordered_entries = trustee_individual + trustee_avg + delegate_individual + delegate_avg
    ordered_handles, ordered_labels = zip(*ordered_entries) if ordered_entries else ([], [])

    plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    if show_plot:
        plt.show()

#%%
for policy_index in range(10):
    df = create_agreement_dataframe(
        policy_index=policy_index,
        prompt_nums=[0, 1],
        model="claude-3-sonnet-v2",
        trustee_type="trustee_ls"
    )

    print("Agreement DataFrame created successfully")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Default vote: {df['default_vote'].iloc[0]}")

    # Create plot
    plot_agreement_rates(df, policy_index=policy_index, trustee_type="trustee_ls")
#%%
df
# Example usage
# if __name__ == "__main__":
#     # Test with a single policy
#     print("Testing agreement plotting functions...")

#     try:
#         # Create agreement DataFrame for policy 0 with prompts 0 and 1
#         df = create_agreement_dataframe(
#             policy_index=0,
#             prompt_nums=[0],
#             model="claude-3-sonnet-v2",
#             trustee_type="trustee_ls"
#         )

#         print("Agreement DataFrame created successfully")
#         print(f"Shape: {df.shape}")
#         print(f"Columns: {list(df.columns)}")
#         print(f"Default vote: {df['default_vote'].iloc[0]}")

#         # Create plot
#         plot_agreement_rates(df, policy_index=0, trustee_type="trustee_ls")

#     except Exception as e:
#         print(f"Error in example: {e}")
# %%
