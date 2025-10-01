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
    trustee_type: str = "trustee_ls",
    compare_expert: bool = False
) -> pd.DataFrame:
    """
    Create a DataFrame with agreement rates across alpha/sigma parameters for different prompts.

    Args:
        policy_index (int): 0-based policy index
        prompt_nums (List[int]): List of prompt numbers to analyze
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        trustee_type (str): Either "trustee_ls" or "trustee_lsd"
        compare_expert (bool): If True, compare to expert vote instead of model default

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
    default_data = load_policy_votes(model, None, policy_index, prompt_nums[0])
    default_vote = default_data['defaults']['vote'].iloc[0] if len(default_data['defaults']) > 0 else "Unknown"

    # Determine reference vote (expert or default)
    if compare_expert:
        expert_vote = _get_expert_vote(policy_index)
        if expert_vote is not None:
            reference_vote = expert_vote
            vote_type = "expert"
            print(f"Using expert vote for policy {policy_index + 1}: {expert_vote}")
        else:
            reference_vote = default_vote
            vote_type = "default"
            print(f"No expert vote found for policy {policy_index + 1}, using default: {default_vote}")
    else:
        reference_vote = default_vote
        vote_type = "default"
        print(f"Using default model vote for policy {policy_index + 1}: {default_vote}")

    result_data['reference_vote'] = [reference_vote] * len(alphas)
    result_data['vote_type'] = [vote_type] * len(alphas)

    # Process each promptw
    for prompt_num in prompt_nums:
        print(f"  Processing prompt {prompt_num}...")

        try:
            # Load data for this prompt
            data = load_policy_votes(model, trustee_type, policy_index, prompt_num)

            # Calculate trustee agreement rates across alpha values
            if trustee_type:
                trustee_agreements = []
                for alpha in alphas:
                    agreement_rate = _calculate_trustee_agreement_rate(
                        data['trustee'], alpha, trustee_type, reference_vote
                    )
                    trustee_agreements.append(agreement_rate)
                result_data[f'trustee_prompt_{prompt_num}_agreement'] = trustee_agreements
            # Calculate delegate agreement rate (constant across alpha)
            if not trustee_type:
                delegate_agreement = _calculate_delegate_agreement_rate(
                    data['delegate'], reference_vote
                )
                delegate_agreements = [delegate_agreement] * len(alphas)
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

def _get_expert_vote(policy_index: int) -> Optional[str]:
    """Get expert vote for a policy if it exists."""
    try:
        policies_df = pd.read_json("../self_selected_policies_new.jsonl", lines=True)
        if policy_index < len(policies_df):
            policy_data = policies_df.iloc[policy_index]
            return policy_data.get('expert_vote', None)
    except Exception as e:
        print(f"Warning: Could not load expert vote for policy {policy_index + 1}: {e}")
    return None

#%%
def _calculate_trustee_agreement_rate(
    trustee_data: pd.DataFrame,
    alpha: float,
    trustee_type: str,
    reference_vote: str
) -> float:
    """Calculate agreement rate between trustee votes and reference vote for given alpha."""
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

    # Calculate agreement rate with reference vote
    agreements = sum(1 for vote in votes if vote == reference_vote)
    return agreements / len(votes)

#%%
def _calculate_delegate_agreement_rate(
    delegate_data: pd.DataFrame,
    reference_vote: str
) -> float:
    """Calculate agreement rate between delegate votes and reference vote."""
    if len(delegate_data) == 0:
        return np.nan

    agreements = sum(1 for vote in delegate_data['vote'] if vote == reference_vote)
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
    reference_vote = agreement_df.get('reference_vote', agreement_df.get('default_vote', ["Unknown"]))[0]
    vote_type = agreement_df.get('vote_type', ["default"])[0]

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

    # Dynamic labeling based on vote type
    vote_label = "Expert Vote" if vote_type == "expert" else "Default Vote"
    plt.ylabel(f'Agreement Rate with {vote_label}', fontsize=12)
    plt.title(f'Agreement Rates vs {param_label}\\n{policy_title}\\n{vote_label}: {reference_vote}',
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


def plot_mean_across_policies(
    policy_indices: List[int],
    delegate_prompt_nums: List[int],
    trustee_prompt_nums: List[int],
    model: str,
    trustee_type: str = "trustee_ls",
    consensus_filter: Optional[str] = None,
    compare_expert: bool = False,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> pd.DataFrame:
    """
    Create a plot showing mean agreement rates across multiple policies.

    Args:
        policy_indices (List[int]): List of 0-based policy indices to average across
        delegate_prompt_nums (List[int]): List of prompt numbers to analyze for delegates
        trustee_prompt_nums (List[int]): List of prompt numbers to analyze for trustees
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        trustee_type (str): Either "trustee_ls", "trustee_lsd", or "both"
        consensus_filter (Optional[str]): Filter policies by consensus ("Yes", "No", or None for all)
        compare_expert (bool): If True, compare to expert votes instead of model defaults for policies with expert votes
        show_plot (bool): Whether to display the plot
        figsize (Tuple[int, int]): Figure size

    Returns:
        pd.DataFrame: DataFrame with mean agreement rates across policies
    """
    # Filter policies by consensus if specified
    if consensus_filter is not None:
        try:
            policies_df = pd.read_json("../self_selected_policies_new.jsonl", lines=True)
            filtered_indices = []
            for policy_index in policy_indices:
                if policy_index < len(policies_df):
                    policy_consensus = policies_df.iloc[policy_index]['consensus']
                    if policy_consensus == consensus_filter:
                        filtered_indices.append(policy_index)
            policy_indices = filtered_indices
            print(f"Filtered to {len(policy_indices)} policies with consensus='{consensus_filter}'")
        except Exception as e:
            print(f"Warning: Could not filter by consensus: {e}")

    print(f"Processing {len(policy_indices)} policies with delegate prompts {delegate_prompt_nums} and trustee prompts {trustee_prompt_nums}...")

    # Parameter range
    alphas = np.arange(0.0, 1.01, 0.1)

    # Determine which trustee types to process
    if trustee_type == "both":
        trustee_types_to_process = ["trustee_ls", "trustee_lsd"]
    else:
        trustee_types_to_process = [trustee_type]

    # Initialize storage for collecting data across policies
    trustee_data = {}  # {trustee_type: {prompt_num: [policy_data_arrays]}}
    delegate_data = {}  # {prompt_num: [policy_data_arrays]}

    # Initialize trustee data structure
    for tt in trustee_types_to_process:
        trustee_data[tt] = {}
        for prompt_num in trustee_prompt_nums:
            trustee_data[tt][prompt_num] = []

    # Initialize delegate data structure
    for prompt_num in delegate_prompt_nums:
        delegate_data[prompt_num] = []

    successful_policies = []

    # Collect data from all policies
    for policy_index in policy_indices:
        try:
            print(f"  Processing policy {policy_index + 1}...")

            # Process each trustee type
            for tt in trustee_types_to_process:
                # Get agreement data for this policy and trustee type
                df = create_agreement_dataframe(
                    policy_index=policy_index,
                    prompt_nums=trustee_prompt_nums,
                    model=model,
                    trustee_type=tt,
                    compare_expert=compare_expert
                )

                # Extract trustee data for each prompt
                for prompt_num in trustee_prompt_nums:
                    trustee_col = f'trustee_prompt_{prompt_num}_agreement'
                    if trustee_col in df.columns:
                        trustee_data[tt][prompt_num].append(df[trustee_col].values)

            # Process delegate data (only need to do this once)
            df_delegate = create_agreement_dataframe(
                policy_index=policy_index,
                prompt_nums=delegate_prompt_nums,
                model=model,
                trustee_type=None,  # Use any trustee type for delegate data
                compare_expert=compare_expert
            )

            # Extract delegate data for each prompt
            for prompt_num in delegate_prompt_nums:
                delegate_col = f'delegate_prompt_{prompt_num}_agreement'
                if delegate_col in df_delegate.columns:
                    delegate_data[prompt_num].append(df_delegate[delegate_col].values)

            successful_policies.append(policy_index)

        except Exception as e:
            print(f"  Error with policy {policy_index + 1}: {e}")
            continue

    print(f"Successfully processed {len(successful_policies)} policies")

    # Calculate means across policies for each prompt
    result_data = {'alpha_sigma': alphas}

    # Process trustee data for each trustee type
    all_trustee_means_for_overall = []
    for tt in trustee_types_to_process:
        for prompt_num in trustee_prompt_nums:
            if prompt_num in trustee_data[tt] and trustee_data[tt][prompt_num]:
                # Stack all policy data for this prompt and calculate mean
                stacked_data = np.stack(trustee_data[tt][prompt_num], axis=0)  # shape: (n_policies, n_alphas)
                mean_across_policies = np.nanmean(stacked_data, axis=0)

                # Create column name with trustee type if "both"
                if trustee_type == "both":
                    col_name = f'{tt}_prompt_{prompt_num}_mean'
                else:
                    col_name = f'trustee_prompt_{prompt_num}_mean'

                result_data[col_name] = mean_across_policies
                all_trustee_means_for_overall.append(mean_across_policies)
            else:
                # Create column name with trustee type if "both"
                if trustee_type == "both":
                    col_name = f'{tt}_prompt_{prompt_num}_mean'
                else:
                    col_name = f'trustee_prompt_{prompt_num}_mean'

                result_data[col_name] = np.full(len(alphas), np.nan)

    # Process delegate data
    delegate_means_for_overall = []
    for prompt_num in delegate_prompt_nums:
        if prompt_num in delegate_data and delegate_data[prompt_num]:
            stacked_data = np.stack(delegate_data[prompt_num], axis=0)
            mean_across_policies = np.nanmean(stacked_data, axis=0)
            result_data[f'delegate_prompt_{prompt_num}_mean'] = mean_across_policies
            delegate_means_for_overall.append(mean_across_policies)
        else:
            result_data[f'delegate_prompt_{prompt_num}_mean'] = np.full(len(alphas), np.nan)

    # Calculate overall means
    if all_trustee_means_for_overall:
        trustee_overall = np.nanmean(np.stack(all_trustee_means_for_overall, axis=0), axis=0)
        result_data['trustee_overall_mean'] = trustee_overall

    if delegate_means_for_overall:
        delegate_overall = np.nanmean(np.stack(delegate_means_for_overall, axis=0), axis=0)
        result_data['delegate_overall_mean'] = delegate_overall

    # Create DataFrame
    result_df = pd.DataFrame(result_data)

    # Create plot
    if show_plot:
        plt.figure(figsize=figsize)

        # Plot individual prompt means (lighter opacity)
        trustee_colors = ['blue', 'navy', 'steelblue', 'royalblue', 'cornflowerblue', 'mediumblue']
        trustee_lsd_colors = ['green', 'darkgreen', 'forestgreen', 'lime', 'darkseagreen', 'olivedrab']
        delegate_colors = ['red', 'darkred', 'crimson', 'lightcoral', 'indianred', 'firebrick']

        # Plot trustee prompt means
        color_idx = 0
        for tt in trustee_types_to_process:
            for prompt_num in trustee_prompt_nums:
                if trustee_type == "both":
                    col = f'{tt}_prompt_{prompt_num}_mean'
                    if tt == "trustee_ls":
                        color = trustee_colors[color_idx % len(trustee_colors)]
                        label = f'Trustee LS Prompt {prompt_num}'
                    else:  # trustee_lsd
                        color = trustee_lsd_colors[color_idx % len(trustee_lsd_colors)]
                        label = f'Trustee LSD Prompt {prompt_num}'
                else:
                    col = f'trustee_prompt_{prompt_num}_mean'
                    color = trustee_colors[color_idx % len(trustee_colors)]
                    label = f'Trustee Prompt {prompt_num}'

                if col in result_df.columns:
                    plt.plot(alphas, result_df[col],
                            color=color, linewidth=2, alpha=0.7, linestyle='-',
                            label=label)
                color_idx += 1

        # Plot delegate prompt means
        for i, prompt_num in enumerate(delegate_prompt_nums):
            col = f'delegate_prompt_{prompt_num}_mean'
            if col in result_df.columns:
                color = delegate_colors[i % len(delegate_colors)]
                plt.plot(alphas, result_df[col],
                        color=color, linewidth=2, alpha=0.7, linestyle='--',
                        label=f'Delegate Prompt {prompt_num}')

        # Plot overall means (thick, dark lines)
        if 'trustee_overall_mean' in result_df.columns:
            plt.plot(alphas, result_df['trustee_overall_mean'],
                    color='darkblue', linewidth=4, alpha=1.0, linestyle='-',
                    label='Trustee Overall Mean', zorder=10)

        if 'delegate_overall_mean' in result_df.columns:
            plt.plot(alphas, result_df['delegate_overall_mean'],
                    color='darkred', linewidth=4, alpha=1.0, linestyle='--',
                    label='Delegate Overall Mean', zorder=10)

        # Formatting
        if trustee_type == "both":
            param_label = "Alpha/Sigma"
        elif trustee_type == "trustee_ls":
            param_label = "Long-term Weight"
        else:  # trustee_lsd
            param_label = "Sigma"
        plt.xlabel(f'{param_label}', fontsize=12)

        # Determine y-axis label based on comparison type
        y_label = 'Mean Agreement Rate with Expert Vote' if compare_expert else 'Mean Agreement Rate with Default Vote'
        plt.ylabel(y_label, fontsize=12)

        # Create title with consensus filter information
        title_parts = [f'Mean Agreement Rates Across {len(successful_policies)} Policies']
        if compare_expert:
            title_parts.append('(vs Expert Votes)')
        if consensus_filter is not None:
            title_parts.append(f'(Consensus: {consensus_filter})')

        if trustee_type == "both":
            title_parts.append(f'{model}, TRUSTEE_LS + TRUSTEE_LSD')
        else:
            title_parts.append(f'{model}, {trustee_type.upper()}')

        plt.title('\\n'.join(title_parts), fontsize=14, fontweight='bold')

        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Organize legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Separate entries
        trustee_entries = [(h, l) for h, l in zip(handles, labels) if 'Trustee' in l]
        delegate_entries = [(h, l) for h, l in zip(handles, labels) if 'Delegate' in l]

        # Order: individual prompts first, then overall means
        trustee_individual = [(h, l) for h, l in trustee_entries if 'Prompt' in l]
        trustee_overall = [(h, l) for h, l in trustee_entries if 'Overall' in l]
        delegate_individual = [(h, l) for h, l in delegate_entries if 'Prompt' in l]
        delegate_overall = [(h, l) for h, l in delegate_entries if 'Overall' in l]

        ordered_entries = trustee_individual + trustee_overall + delegate_individual + delegate_overall
        ordered_handles, ordered_labels = zip(*ordered_entries) if ordered_entries else ([], [])

        plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

    return result_df


#%%
# policy_index = 2
# df = create_agreement_dataframe(
#     policy_index=policy_index,
#     prompt_nums=[0],
#     model="grok-4",
#     trustee_type="trustee_ls"
# )
#%%

# print("Agreement DataFrame created successfully")
# print(f"Shape: {df.shape}")
# print(f"Columns: {list(df.columns)}")
# print(f"Default vote: {df['default_vote'].iloc[0]}")

# # Create plot
# plot_agreement_rates(df, policy_index=policy_index, trustee_type="trustee_ls")
# for policy_index in range(30):
#     df = create_agreement_dataframe(
#         policy_index=policy_index,
#         prompt_nums=[0],
#         model="gpt-4o-mini",
#         trustee_type="trustee_ls"
#     )

#     # print("Agreement DataFrame created successfully")
#     # print(f"Shape: {df.shape}")
#     # print(f"Columns: {list(df.columns)}")
#     # print(f"Default vote: {df['default_vote'].iloc[0]}")

#     # Create plot
#     plot_agreement_rates(df, policy_index=policy_index, trustee_type="trustee_ls")
# #%%
# # for model in ["claude-3-sonnet-v2", "gpt-4o"]:
# #     for trustee_type in ["trustee_ls", "trustee_lsd"]:
# #         for consensus_filter in ["No", "Yes"]:
# #             plot_mean_across_policies(policy_indices=range(30),
# #                                     delegate_prompt_nums=[0, 1, 2, 3, 4],
# #                                     trustee_prompt_nums=[0, 1, 2],
# #                                     model=model,
# #                                     trustee_type=trustee_type,
# #                                     consensus_filter=consensus_filter,
# #                                     compare_expert=False)
# #         #plt.show()
# # #%%
# plot_mean_across_policies(policy_indices=range(30),
#                         delegate_prompt_nums=[0,1,2,3,4],
#                         trustee_prompt_nums=[0,1,2],
#                         model="claude-3-sonnet-v2",
#                         trustee_type="both",
#                         consensus_filter=None,
#                         compare_expert=False)
# %%

# %%
