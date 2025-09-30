#!/usr/bin/env python3
"""
Create combined demographic agreement visualization with 4 models.

Top row: Agreement with expert consensus (delegate/trustee agreement with expert)
Bottom row: Delegate-trustee agreement (internal agreement)

Both rows show Political Affiliation (left) and Race (right) demographics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from agreement_demographics import create_agreement_dataframe
from vote_data_loader import _load_delegate_data, _load_trustee_data
from compare_delegates_trustees import calculate_weighted_vote, calculate_discounted_vote

# Model configuration
MODELS = [
    "claude-3-sonnet-v2",
    "claude-3-haiku-v2-mini",
    "gpt-4o-mini",
    "gpt-4o"
]

MODEL_DISPLAY_NAMES = {
    "claude-3-sonnet-v2": "Claude Sonnet",
    "claude-3-haiku-v2-mini": "Claude Haiku",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o": "GPT-4o"
}

MODEL_COLORS = {
    "claude-3-sonnet-v2": "#000000",      # Black
    "claude-3-haiku-v2-mini": "#0000FF",  # Blue
    "gpt-4o-mini": "#FFA500",             # Orange
    "gpt-4o": "#FF0000"                   # Red
}


def collect_expert_agreement_data(
    policy_indices: List[int],
    prompt_nums: List[int],
    models: List[str],
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographic: str,
    alphas: List[float]
) -> pd.DataFrame:
    """
    Collect delegate and trustee agreement rates with expert consensus
    across all models and demographic groups.

    Returns DataFrame with columns:
    - demographic_group: The demographic category (e.g., "Republican")
    - condition: "Delegate" or "Trustee"
    - model: Model name
    - agreement_rate: Agreement rate with expert
    """
    all_results = []

    for model in models:
        print(f"\nProcessing model: {model}")

        for policy_idx in policy_indices:
            try:
                df = create_agreement_dataframe(
                    policy_index=policy_idx,
                    prompt_nums=prompt_nums,
                    model=model,
                    alphas=alphas,
                    trustee_type=trustee_type,
                    compare_expert=True,
                    bio_df=bio_df,
                    demographic=demographic
                )

                # Extract delegate and trustee agreement rates by group
                for col in df.columns:
                    if "_agreement_" in col:
                        parts = col.split("_agreement_")
                        if len(parts) == 2:
                            group = parts[1]

                            if col.startswith("trustee_"):
                                # Average across alphas
                                val = df[col].mean()
                                condition = "Trustee"
                            elif col.startswith("delegate_"):
                                # Constant across alphas, take first
                                val = df[col].iloc[0]
                                condition = "Delegate"
                            else:
                                continue

                            all_results.append({
                                "demographic_group": group,
                                "condition": condition,
                                "model": model,
                                "agreement_rate": val,
                                "policy_idx": policy_idx
                            })

            except Exception as e:
                print(f"  Error with policy {policy_idx + 1}: {e}")
                continue

    results_df = pd.DataFrame(all_results)

    # Average across policies
    if len(results_df) > 0:
        results_df = results_df.groupby(
            ["demographic_group", "condition", "model"], as_index=False
        )["agreement_rate"].mean()

    return results_df


def collect_delegate_trustee_proportion_agreement_data(
    policy_indices: List[int],
    delegate_prompt_nums: List[int],
    trustee_prompt_nums: List[int],
    models: List[str],
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographic: str,
    alpha: float
) -> pd.DataFrame:
    """
    Collect delegate-trustee agreement based on proportion differences.

    Agreement = 1 - |delegate_yes_prop - trustee_yes_prop|

    Returns DataFrame with columns:
    - demographic_group: The demographic category
    - model: Model name
    - agreement_rate: Agreement based on yes proportion difference
    """
    all_results = []

    for model in models:
        print(f"\nProcessing model: {model}")

        for policy_idx in policy_indices:
            try:
                # Collect all delegate and trustee votes for this policy
                all_delegate_votes = []
                all_trustee_votes = []

                # Load delegate data from all prompts
                for prompt_num in delegate_prompt_nums:
                    delegate_file = f"../data/delegate/{model}/self_selected_policies_new/prompt-{prompt_num}/d_policy_{policy_idx + 1}_votes.jsonl"
                    try:
                        delegate_df = _load_delegate_data(delegate_file)
                        if len(delegate_df) > 0 and 'participant_id' in delegate_df.columns:
                            # Merge with bio data
                            delegate_df = delegate_df.merge(bio_df, left_on='participant_id', right_on='participant_id', how='left')
                            all_delegate_votes.append(delegate_df)
                    except FileNotFoundError:
                        print(f"    Delegate file not found: prompt {prompt_num}")
                        continue
                    except Exception as e:
                        print(f"    Error loading delegate data from prompt {prompt_num}: {e}")
                        continue

                # Load trustee data from all prompts and calculate votes
                for prompt_num in trustee_prompt_nums:
                    trustee_file = f"../data/{trustee_type}/{model}/self_selected_policies_new/prompt-{prompt_num}/t_policy_{policy_idx + 1}_votes.jsonl"
                    try:
                        trustee_df = _load_trustee_data(trustee_file, trustee_type)

                        if len(trustee_df) == 0:
                            continue

                        # Calculate trustee votes using alpha
                        trustee_votes_list = []
                        for _, row in trustee_df.iterrows():
                            try:
                                if trustee_type == "trustee_ls":
                                    entry = {
                                        "yes_vote": {"short_util": row['yes_short_util'], "long_util": row['yes_long_util']},
                                        "no_vote": {"short_util": row['no_short_util'], "long_util": row['no_long_util']}
                                    }
                                    vote_result = calculate_weighted_vote(entry, alpha)
                                elif trustee_type == "trustee_lsd":
                                    entry = {"yes": {}, "no": {}}
                                    for period in ["0-5 years", "5-10 years", "10-15 years", "15-20 years", "20-25 years", "25-30 years"]:
                                        period_key = period.replace("-", "_").replace(" ", "_")
                                        entry["yes"][period] = {"score": row[f'yes_{period_key}_score']}
                                        entry["no"][period] = {"score": row[f'no_{period_key}_score']}
                                    vote_result = calculate_discounted_vote(entry, alpha)

                                trustee_votes_list.append({
                                    'participant_id': row['participant_id'],
                                    'vote': vote_result['vote']
                                })
                            except Exception as e:
                                continue

                        if trustee_votes_list:
                            trustee_vote_df = pd.DataFrame(trustee_votes_list)
                            trustee_vote_df = trustee_vote_df.merge(bio_df, on='participant_id', how='left')
                            all_trustee_votes.append(trustee_vote_df)

                    except FileNotFoundError:
                        print(f"    Trustee file not found: prompt {prompt_num}")
                        continue
                    except Exception as e:
                        print(f"    Error loading trustee data from prompt {prompt_num}: {e}")
                        continue

                if not all_delegate_votes or not all_trustee_votes:
                    continue

                # Combine all votes
                delegate_combined = pd.concat(all_delegate_votes, ignore_index=True)
                trustee_combined = pd.concat(all_trustee_votes, ignore_index=True)
               # print(delegate_combined)
               # print(trustee_combined)
                #raise Exception("Stop here")
                # Calculate agreement by demographic group
                for group in delegate_combined[demographic].dropna().unique():
                    delegate_group = delegate_combined[delegate_combined[demographic] == group]
                    trustee_group = trustee_combined[trustee_combined[demographic] == group]

                    if len(delegate_group) > 0 and len(trustee_group) > 0:
                        delegate_yes_prop = ((delegate_group['vote'] == 'Yes') | (delegate_group['vote'] == 'yes')).mean()
                        trustee_yes_prop = ((trustee_group['vote'] == 'Yes') | (trustee_group['vote'] == 'yes')).mean()

                        agreement = 1 - abs(delegate_yes_prop - trustee_yes_prop)

                        all_results.append({
                            "demographic_group": group,
                            "model": model,
                            "agreement_rate": agreement,
                            "policy_idx": policy_idx
                        })

            except Exception as e:
                print(f"  Error with policy {policy_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

    results_df = pd.DataFrame(all_results)

    # Average across policies
    if len(results_df) > 0:
        results_df = results_df.groupby(
            ["demographic_group", "model"], as_index=False
        )["agreement_rate"].mean()

    return results_df


def plot_expert_agreement_panel(
    ax,
    data: pd.DataFrame,
    demographic: str,
    title: str
):
    """
    Plot top row panel: delegate/trustee agreement with expert consensus.

    Structure: One bar per demographic_group/condition showing average across models,
    with colored markers overlaid for individual model values.
    """
    # Replace "Black or African American" with "African American"
    data["demographic_group"] = data["demographic_group"].replace(
        "Black or African American", "African American"
    )
    groups = sorted(data["demographic_group"].unique())
    conditions = ["Delegate", "Trustee"]
    models = MODELS

    # Bar positioning with spacing
    n_groups = len(groups)

    bar_width = 0.4
    small_gap = 0.2  # Gap between Delegate and Trustee within a group
    large_gap = 0.8  # Gap between demographic groups

    # Calculate x positions
    x_positions = []
    tick_positions = []
    tick_labels = []

    current_x = 0
    for i, group in enumerate(groups):
        for j, condition in enumerate(conditions):
            if j > 0:
                current_x += small_gap  # Add small gap before Trustee

            x_positions.append(current_x)
            tick_positions.append(current_x)
            tick_labels.append(f"{group}\n{condition}")
            current_x += bar_width

        current_x += large_gap  # Add large gap after each demographic group

    # Plot bars (averaged across models)
    bar_idx = 0
    for i, group in enumerate(groups):
        for j, condition in enumerate(conditions):
            # Get values for all models
            model_vals = []
            for model in models:
                subset = data[
                    (data["demographic_group"] == group) &
                    (data["condition"] == condition) &
                    (data["model"] == model)
                ]
                if len(subset) > 0:
                    model_vals.append(subset["agreement_rate"].values[0])
                else:
                    model_vals.append(np.nan)

            # Calculate mean across models
            mean_val = np.nanmean(model_vals) if model_vals else 0

            # Plot bar with darker gray color for emphasis
            ax.bar(
                x_positions[bar_idx],
                mean_val,
                width=bar_width,
                color="gray",
                alpha=0.8,  # Darker bars
                zorder=1
            )

            # Overlay individual model markers (lighter)
            for k, (model, val) in enumerate(zip(models, model_vals)):
                if not np.isnan(val):
                    # Only add label on first occurrence for legend
                    label = MODEL_DISPLAY_NAMES[model] if i == 0 and j == 0 else None

                    ax.scatter(
                        x_positions[bar_idx],
                        val,
                        color=MODEL_COLORS[model],
                        s=80,  # Slightly smaller
                        label=label,
                        zorder=3,
                        alpha=0.6,  # Lighter markers
                        edgecolors='white',
                        linewidths=1.5
                    )

            bar_idx += 1

    # Formatting
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Agreement Rate with Expert")
    ax.set_ylim(0.4, 1.0)  # Start at 40%, end at 100%
    ax.set_title(title, fontsize=12)  # No bold
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)

    # Add vertical lines to separate demographic groups
    separator_x = []
    current_x = bar_width
    for i, group in enumerate(groups[:-1]):
        current_x += small_gap + bar_width  # Trustee bar
        current_x += large_gap / 2  # Middle of large gap
        separator_x.append(current_x)
        current_x += large_gap / 2 + bar_width  # Next delegate bar

    for x in separator_x:
        ax.axvline(x=x, color="black", linestyle="-", alpha=0.5, linewidth=1)


def plot_delegate_trustee_agreement_panel(
    ax,
    data: pd.DataFrame,
    demographic: str,
    title: str
):
    """
    Plot bottom row panel: delegate-trustee internal agreement.

    Structure: One bar per demographic_group showing average across models,
    with colored markers overlaid for individual model values.
    """
    # Replace "Black or African American" with "African American"
    data["demographic_group"] = data["demographic_group"].replace(
        "Black or African American", "African American"
    )
    groups = sorted(data["demographic_group"].unique())
    models = MODELS

    # Bar positioning
    n_groups = len(groups)
    bar_width = 0.5

    x_positions = np.arange(n_groups)

    # Plot bars (averaged across models)
    for i, group in enumerate(groups):
        # Get values for all models
        model_vals = []
        for model in models:
            subset = data[
                (data["demographic_group"] == group) &
                (data["model"] == model)
            ]
            if len(subset) > 0:
                model_vals.append(subset["agreement_rate"].values[0])
            else:
                model_vals.append(np.nan)

        # Calculate mean across models
        mean_val = np.nanmean(model_vals) if model_vals else 0

        # Plot bar with darker gray color for emphasis
        ax.bar(
            x_positions[i],
            mean_val,
            width=bar_width,
            color="gray",
            alpha=0.8,  # Darker bars
            zorder=1
        )

        # Overlay individual model markers (lighter)
        for model, val in zip(models, model_vals):
            if not np.isnan(val):
                # Only add label on first group for legend
                label = MODEL_DISPLAY_NAMES[model] if i == 0 else None

                ax.scatter(
                    x_positions[i],
                    val,
                    color=MODEL_COLORS[model],
                    s=80,  # Slightly smaller
                    label=label,
                    zorder=3,
                    alpha=0.6,  # Lighter markers
                    edgecolors='white',
                    linewidths=1.5
                )

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel("Delegate-Trustee Agreement Rate")
    ax.set_ylim(0.4, 1.0)  # Start at 40%, end at 100%
    # No title for bottom row
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)


def create_combined_demographic_plot(
    expert_consensus_policies: List[int],
    no_consensus_policies: List[int],
    prompt_nums: List[int],
    delegate_prompt_nums: List[int],
    trustee_prompt_nums: List[int],
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographics: List[str],
    alphas: List[float],
    alpha: float,
    figsize: Tuple[int, int] = (16, 12),
    output_file: str = None
):
    """
    Create combined 2x2 plot showing agreement by demographics across models.

    Top row: Agreement with expert consensus (delegate/trustee)
    Bottom row: Delegate-trustee agreement based on proportion differences (1 - |delegate_yes_prop - trustee_yes_prop|)
    Columns: One for each demographic (Political Affiliation, Race)

    Args:
        expert_consensus_policies: Policy indices with expert votes
        no_consensus_policies: Policy indices without expert votes
        prompt_nums: Prompt numbers for expert agreement (top row)
        delegate_prompt_nums: Delegate prompt numbers for bottom row
        trustee_prompt_nums: Trustee prompt numbers for bottom row
        trustee_type: "trustee_ls" or "trustee_lsd"
        bio_df: Biography dataframe with demographic info
        demographics: List of demographic columns to plot
        alphas: Alpha values for expert agreement calculation (top row)
        alpha: Single alpha value for bottom row trustee calculation
        figsize: Figure size
        output_file: Optional output file path
    """
    print("=" * 70)
    print("Creating Combined Demographic Agreement Visualization")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Process each demographic
    for col_idx, demographic in enumerate(demographics):
        print(f"\n{'='*70}")
        print(f"Processing demographic: {demographic}")
        print(f"{'='*70}")

        # Top row: Expert agreement
        print("\n--- Collecting expert agreement data ---")
        expert_data = collect_expert_agreement_data(
            policy_indices=expert_consensus_policies,
            prompt_nums=prompt_nums,
            models=MODELS,
            trustee_type=trustee_type,
            bio_df=bio_df,
            demographic=demographic,
            alphas=alphas
        )

        plot_expert_agreement_panel(
            axes[0, col_idx],
            expert_data,
            demographic,
            demographic  # Just the demographic name, no "Agreement with Expert Consensus"
        )

        # Bottom row: Delegate-trustee agreement (proportion-based)
        print("\n--- Collecting delegate-trustee agreement data ---")
        dt_data = collect_delegate_trustee_proportion_agreement_data(
            policy_indices=no_consensus_policies,
            delegate_prompt_nums=delegate_prompt_nums,
            trustee_prompt_nums=trustee_prompt_nums,
            models=MODELS,
            trustee_type=trustee_type,
            bio_df=bio_df,
            demographic=demographic,
            alpha=alpha
        )

        plot_delegate_trustee_agreement_panel(
            axes[1, col_idx],
            dt_data,
            demographic,
            None  # No title for bottom row
        )

    # Add legend (bottom right, outside the plots)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02),
               fontsize=10, frameon=True)

    # Remove individual legends
    for ax_row in axes:
        for ax in ax_row:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    # Overall title
    fig.suptitle(
        "Agreement Rates by Demographics Across Models",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nSaved to: {output_file}")

    plt.show()

    return fig


# Example usage
if __name__ == "__main__":
    # Load biographies
    bio_df = pd.read_json("rep_biographies.jsonl", lines=True)
    bio_df.rename(columns={"ID": "participant_id"}, inplace=True)

    # Configuration
    trustee_type = "trustee_ls"
    prompt_nums = [0, 1, 2]  # For expert agreement (top row)
    delegate_prompt_nums = [0, 1, 2, 3, 4]  # For bottom row
    trustee_prompt_nums = [0, 1, 2]  # For bottom row
    alphas = [1.0]  # Use alpha=1.0 for expert agreement (top row)
    alpha = 1.0  # Use alpha=1.0 for trustee calculations (bottom row)

    # Split policies by expert consensus availability
    # TODO: Programmatically determine which policies have expert consensus
    self_selected_policies = pd.read_json("../self_selected_policies_new.jsonl", lines=True)
    print(self_selected_policies.head())
    expert_consensus_policies = self_selected_policies[self_selected_policies['consensus'] == 'Yes']['id'].tolist()
    expert_consensus_policies = [policy - 1 for policy in expert_consensus_policies]
    no_consensus_policies = self_selected_policies[self_selected_policies['consensus'] == 'No']['id'].tolist()
    no_consensus_policies = [policy - 1 for policy in no_consensus_policies]
    print(f"Expert consensus policies: {len(expert_consensus_policies)}")
    print(f"No consensus policies: {len(no_consensus_policies)}")
    #raise Exception("Stop here")
    #expert_consensus_policies = list(range(20, 30))  # Policies with expert votes
    #no_consensus_policies = list(range(0, 20))       # Policies without expert votes

    demographics = ["Political Affiliation", "Race"]

    create_combined_demographic_plot(
        expert_consensus_policies=expert_consensus_policies,
        no_consensus_policies=no_consensus_policies,
        prompt_nums=prompt_nums,
        delegate_prompt_nums=delegate_prompt_nums,
        trustee_prompt_nums=trustee_prompt_nums,
        trustee_type=trustee_type,
        bio_df=bio_df,
        demographics=demographics,
        alphas=alphas,
        alpha=alpha,
        output_file="agreement_visuals/combined_demographic_agreement.png"
    )