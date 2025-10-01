#%%
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Dict, Tuple, Optional
import os

# Import utility functions
from vote_data_loader import load_policy_votes
from compare_delegates_trustees import calculate_weighted_vote, calculate_discounted_vote

bio_df = pd.read_json("rep_biographies.jsonl", lines=True)
bio_df.rename(columns={"ID": "participant_id"}, inplace=True)

#%%
def create_agreement_dataframe(
    policy_index: int,
    delegate_prompt_nums: List[int],
    trustee_prompt_nums: List[int],
    model: str,
    alpha: float,
    trustee_type: str = "trustee_ls",
    compare_expert: bool = False,
    bio_df: Optional[pd.DataFrame] = None,
    demographic: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a DataFrame with agreement rates for a single alpha value
    for different prompts. Supports splitting agreement rates by demographic.

    Args:
        policy_index (int): 0-based policy index
        delegate_prompt_nums (List[int]): List of delegate prompt numbers to analyze
        trustee_prompt_nums (List[int]): List of trustee prompt numbers to analyze
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        alpha (float): Alpha/sigma value for trustee calculations
        trustee_type (str): Either "trustee_ls", "trustee_lsd", or "both"
        compare_expert (bool): If True, compare to expert vote instead of model default
        bio_df (pd.DataFrame, optional): DataFrame of biographies with "id" column
        demographic (str, optional): Column in bio_df to split agreement by

    Returns:
        pd.DataFrame: DataFrame with agreement rates
    """

    result_data = {}

    # Determine which trustee types to process
    if trustee_type == "both":
        trustee_types_to_process = ["trustee_ls", "trustee_lsd"]
        print(f"Processing policy {policy_index + 1} with {len(trustee_prompt_nums)} trustee prompts (both LS and LSD) and {len(delegate_prompt_nums)} delegate prompts at alpha={alpha}...")
    else:
        trustee_types_to_process = [trustee_type]
        print(f"Processing policy {policy_index + 1} with {len(trustee_prompt_nums)} trustee prompts and {len(delegate_prompt_nums)} delegate prompts at alpha={alpha}...")

    # Load default vote for this policy (use first trustee type if "both")
    default_data = load_policy_votes(model, None, policy_index, trustee_prompt_nums[0])
    default_vote = (
        default_data["defaults"]["vote"].iloc[0]
        if len(default_data["defaults"]) > 0
        else "Unknown"
    )

    # Determine reference vote
    if compare_expert:
        expert_vote = _get_expert_vote(policy_index)
        if expert_vote is not None:
            reference_vote = expert_vote
            vote_type = "expert"
            print(f"Using expert vote for policy {policy_index + 1}: {expert_vote}")
        else:
            reference_vote = default_vote
            vote_type = "default"
            print(
                f"No expert vote found for policy {policy_index + 1}, using default: {default_vote}"
            )
    else:
        reference_vote = default_vote
        vote_type = "default"
        print(f"Using default model vote for policy {policy_index + 1}: {default_vote}")

    result_data["reference_vote"] = reference_vote
    result_data["vote_type"] = vote_type

    # Process trustee prompts
    for prompt_num in trustee_prompt_nums:
        print(f"  Processing trustee prompt {prompt_num}...")

        if demographic and bio_df is not None:
            # Trustee (split by demographic) - accumulate across trustee types
            group_rates_by_type = {}  # {group: [rate_from_type1, rate_from_type2]}

            for tt in trustee_types_to_process:
                print(f"    Processing trustee type {tt} prompt {prompt_num}...")
                try:
                    data = load_policy_votes(model, tt, policy_index, prompt_num)
                    rates = _calculate_trustee_agreement_rate_by_demo(
                        data["trustee"], alpha, tt, reference_vote, bio_df, demographic
                    )
                    for group, val in rates.items():
                        if group not in group_rates_by_type:
                            group_rates_by_type[group] = []
                        group_rates_by_type[group].append(val)
                except Exception as e:
                    print(f"    Error processing trustee type {tt} prompt {prompt_num}: {e}")
                    continue

            # Average across trustee types for each group

            for group, rates_list in group_rates_by_type.items():
                col_name = f"trustee_prompt_{prompt_num}_agreement_{group}"
                result_data[col_name] = np.nanmean(rates_list) if rates_list else np.nan

        else:
            # Trustee (overall) - accumulate across trustee types
            rates_by_type = []

            for tt in trustee_types_to_process:
                try:
                    data = load_policy_votes(model, tt, policy_index, prompt_num)
                    agreement_rate = _calculate_trustee_agreement_rate_by_demo(
                        data["trustee"], alpha, tt, reference_vote
                    )
                    rates_by_type.append(agreement_rate)
                except Exception as e:
                    print(f"    Error processing trustee type {tt} prompt {prompt_num}: {e}")
                    continue

            # Average across trustee types
            result_data[f"trustee_prompt_{prompt_num}_agreement"] = np.nanmean(rates_by_type) if rates_by_type else np.nan

    # Process delegate prompts
    for prompt_num in delegate_prompt_nums:
        print(f"  Processing delegate prompt {prompt_num}...")

        try:
            data = load_policy_votes(model, None, policy_index, prompt_num)
            if demographic and bio_df is not None:
                # Delegate (split by demographic)
                rates = _calculate_delegate_agreement_rate_by_demo(
                    data["delegate"], reference_vote, bio_df, demographic
                )
                for group, val in rates.items():
                    col_name = f"delegate_prompt_{prompt_num}_agreement_{group}"
                    result_data[col_name] = val

            else:
                # Delegate (overall)
                delegate_agreement = _calculate_delegate_agreement_rate_by_demo(
                    data["delegate"], reference_vote
                )
                result_data[f"delegate_prompt_{prompt_num}_agreement"] = delegate_agreement

        except Exception as e:
            print(f"  Error processing delegate prompt {prompt_num}: {e}")
            # Fill with NaN values
            if demographic and bio_df is not None:
                # If demographic mode, add NaNs for each group (unknown groups until first success)
                pass
            else:
                result_data[f"delegate_prompt_{prompt_num}_agreement"] = np.nan

    # Convert to DataFrame (single row)
    df = pd.DataFrame([result_data]) if result_data else pd.DataFrame()

    # Calculate mean agreement rates only if not splitting by demographic
    if not (demographic and bio_df is not None) and len(df) > 0:
        trustee_cols = [col for col in df.columns if col.startswith("trustee_prompt_")]
        delegate_cols = [col for col in df.columns if col.startswith("delegate_prompt_")]

        if trustee_cols:
            df["trustee_mean_agreement"] = df[trustee_cols].mean(axis=1)

        if delegate_cols:
            df["delegate_mean_agreement"] = df[delegate_cols].mean(axis=1)

    return df
def create_delegate_trustee_agreement_dataframe(
    policy_index: int,
    prompt_nums: List[int],
    model: str,
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographic: str,
    alphas: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Build DataFrame of delegate–trustee agreement by demographic, averaged across prompts.
    """
    if alphas is None:
        alphas = np.arange(0.0, 1.01, 0.1)

    result_data = {"alpha_sigma": alphas}

    # Storage for all prompt results
    group_storage: Dict[str, List[np.ndarray]] = {}

    for prompt_num in prompt_nums:
        try:
            data = load_policy_votes(model, trustee_type, policy_index, prompt_num)

            for alpha in alphas:
                rates = calculate_delegate_trustee_agreement_by_demo(
                    data["trustee"], data["delegate"], alpha, trustee_type, bio_df, demographic
                )
                for group, val in rates.items():
                    group_storage.setdefault(group, [[] for _ in alphas])
                    group_storage[group][list(alphas).index(alpha)].append(val)

        except Exception as e:
            print(f"  Error processing prompt {prompt_num}: {e}")
            continue

    # Compute mean across prompts
    for group, alpha_vals_list in group_storage.items():
        mean_vals = [np.nanmean(alpha_vals) if alpha_vals else np.nan for alpha_vals in alpha_vals_list]
        col_name = f"agreement_{group}"
        result_data[col_name] = mean_vals

    return pd.DataFrame(result_data)


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
def _calculate_trustee_agreement_rate_by_demo(
    trustee_data: pd.DataFrame,
    alpha: float,
    trustee_type: str,
    reference_vote: str,
    bio_df: Optional[pd.DataFrame] = None,
    demographic: Optional[str] = None
):
    votes = []

    for _, row in trustee_data.iterrows():
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
        

        votes.append({"participant_id": row['participant_id'], "vote": vote_result['vote']})
    if not votes:
        if bio_df is not None and demographic is not None:
            return pd.Series(dtype=float)
        else:
            return 0.0

    df_votes = pd.DataFrame(votes)

    # If demographic split is requested
    if bio_df is not None and demographic is not None:
        merged = df_votes.merge(bio_df, on="participant_id", how="left")
        grouped = merged.groupby(demographic)['vote'].apply(
            lambda v: (v == reference_vote).mean()
        )
        return grouped
    else:
        # Return overall agreement rate
        return (df_votes['vote'] == reference_vote).mean()
#%%
def _calculate_delegate_agreement_rate_by_demo(
    delegate_data: pd.DataFrame,
    reference_vote: str,
    bio_df: Optional[pd.DataFrame] = None,
    demographic: Optional[str] = None
):
    """Agreement rate between delegate votes and reference vote, optionally split by demographic group."""
    if len(delegate_data) == 0:
        if bio_df is not None and demographic is not None:
            return pd.Series(dtype=float)
        else:
            return 0.0

    # If demographic split is requested
    if bio_df is not None and demographic is not None:
        merged = delegate_data.merge(bio_df, on="participant_id", how="left")
        grouped = merged.groupby(demographic)['vote'].apply(
            lambda v: (v == reference_vote).mean()
        )
        return grouped
    else:
        # Return overall agreement rate
        return (delegate_data['vote'] == reference_vote).mean()

def calculate_delegate_trustee_agreement_by_demo(
    trustee_data: pd.DataFrame,
    delegate_data: pd.DataFrame,
    alpha: float,
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographic: str
) -> pd.Series:
    """
    Calculate agreement rate between delegate and trustee votes within each demographic group.
    """
    votes = []

    for _, row in trustee_data.iterrows():
        try:
            if trustee_type == "trustee_ls":
                entry = {
                    "yes_vote": {"short_util": row["yes_short_util"], "long_util": row["yes_long_util"]},
                    "no_vote": {"short_util": row["no_short_util"], "long_util": row["no_long_util"]}
                }
                vote_result = calculate_weighted_vote(entry, alpha)

            elif trustee_type == "trustee_lsd":
                vote_result = calculate_discounted_vote(entry, alpha)

            votes.append({"participant_id": row["participant_id"], "trustee_vote": vote_result["vote"]})
        except:
            continue

    if not votes or len(delegate_data) == 0:
        return pd.Series(dtype=float)

    trustee_df = pd.DataFrame(votes)
    delegate_df = delegate_data.rename(columns={"vote": "delegate_vote"})

    merged = trustee_df.merge(delegate_df, on="participant_id", how="inner")
    merged = merged.merge(bio_df, on="participant_id", how="left")

    merged["agree"] = (merged["trustee_vote"] == merged["delegate_vote"]).astype(int)

    grouped = merged.groupby(demographic)["agree"].mean()
    return grouped

#%%
def plot_agreement_by_demographic(
    agreement_df: pd.DataFrame,
    demographic: str,
    policy_index: int,
    trustee_type: str = "trustee_ls",
    compare_expert: bool = False,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True,
) -> None:
    """
    Plot agreement rates split by demographic group.
    Each demographic is a cluster with 2 bars: Delegate vs Trustee.

    Args:
        agreement_df (pd.DataFrame): Output from create_agreement_dataframe with demographic split
        demographic (str): Column used for demographic split (e.g., "Political Affiliation")
        policy_index (int): Policy index for labeling
        trustee_type (str): Trustee type ("trustee_ls" or "trustee_lsd")
        compare_expert (bool): Whether agreement is vs expert or default
        figsize (Tuple[int, int]): Figure size
        show_plot (bool): Whether to display plot
    """
    reference_vote = agreement_df.get("reference_vote", ["Unknown"])[0]
    vote_type = agreement_df.get("vote_type", ["default"])[0]

    # Collect data
    result_rows = []
    for col in agreement_df.columns:
        if not (col.startswith("trustee_prompt_") or col.startswith("delegate_prompt_")):
            continue
        if "_agreement_" in col:
            parts = col.split("_agreement_")
            group = parts[1]
            if col.startswith("trustee_"):
                val = agreement_df[col].mean()  # avg across alphas
                result_rows.append((group, "Trustee", val))
            elif col.startswith("delegate_"):
                val = agreement_df[col].iloc[0]  # constant across alphas
                result_rows.append((group, "Delegate", val))

    results_df = pd.DataFrame(result_rows, columns=[demographic, "Condition", "Agreement"])

    # Unique groups in demographic order
    groups = sorted(results_df[demographic].unique())
    conditions = ["Delegate", "Trustee"]

    x = np.arange(len(groups))  # positions for demographic groups
    width = 0.35

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot Delegate & Trustee side by side within each cluster
    for i, cond in enumerate(conditions):
        vals = [
            results_df.loc[
                (results_df[demographic] == g) & (results_df["Condition"] == cond),
                "Agreement",
            ].values[0]
            if not results_df.loc[
                (results_df[demographic] == g) & (results_df["Condition"] == cond)
            ].empty
            else np.nan
            for g in groups
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, label=cond)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(f"Agreement Rate with {'Expert' if compare_expert else 'Default'} Vote")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Agreement by {demographic} — Policy {policy_index+1}\nReference: {reference_vote} ({vote_type})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if show_plot:
        plt.show()

def plot_delegate_trustee_agreement_by_demo(
    df: pd.DataFrame,
    demographic: str,
    policy_index: int,
    trustee_type: str = "trustee_ls",
    figsize: Tuple[int, int] = (8, 6),
    show_plot: bool = True
):
    """
    Plot delegate–trustee agreement rates within demographics (averaged across prompts).
    """
    # Average across alphas
    mean_vals = {col: df[col].mean() for col in df.columns if col.startswith("agreement_")}

    groups = [col.replace("agreement_", "") for col in mean_vals.keys()]
    values = list(mean_vals.values())

    plt.figure(figsize=figsize)
    plt.bar(groups, values, color="skyblue")

    plt.ylabel("Delegate–Trustee Agreement Rate")
    plt.ylim(0, 1)
    plt.title(f"Delegate–Trustee Agreement by {demographic}\nPolicy {policy_index+1} (Averaged across prompts)",
              fontsize=13, fontweight="bold")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))
    plt.grid(axis="y", alpha=0.3)

    if show_plot:
        plt.show()

def save_all_policy_plots_to_pdf(
    policy_indices: List[int],
    prompt_nums: List[int],
    model: str,
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographic: str,
    alphas: List[float],
    compare_expert: bool = False,
    output_file: str = "agreement_by_demographic.pdf"
):
    """
    Generate and save agreement plots for all policies into a single PDF.

    Args:
        policy_indices (List[int]): Policies to process (e.g., range(30))
        prompt_nums (List[int]): Prompt numbers to analyze
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        trustee_type (str): Trustee type ("trustee_ls" or "trustee_lsd")
        bio_df (pd.DataFrame): Biography dataframe
        demographic (str): Demographic column name to split on
        compare_expert (bool): Use expert vote if available
        output_file (str): Path to save PDF
    """
    with PdfPages(output_file) as pdf:
        for policy_index in policy_indices:
            try:
                df = create_agreement_dataframe(
                    policy_index=policy_index,
                    prompt_nums=prompt_nums,
                    model=model,
                    trustee_type=trustee_type,
                    alphas=alphas,
                    compare_expert=compare_expert,
                    bio_df=bio_df,
                    demographic=demographic,
                )

                plt.figure()
                plot_agreement_by_demographic(
                    df,
                    demographic=demographic,
                    policy_index=policy_index,
                    trustee_type=trustee_type,
                    compare_expert=compare_expert,
                    show_plot=False  # don’t display interactively
                )
                pdf.savefig()   # save the current figure into PDF
                plt.close()     # close figure to free memory
                print(f"Saved policy {policy_index+1} to PDF")
            except Exception as e:
                print(f"Error with policy {policy_index+1}: {e}")

    print(f"All plots saved to {output_file}")

from matplotlib.backends.backend_pdf import PdfPages

def save_delegate_trustee_agreement_plots_to_pdf(
    policy_indices: List[int],
    prompt_nums: List[int],
    model: str,
    trustee_type: str,
    bio_df: pd.DataFrame,
    demographic: str,
    alphas: List[float],
    output_file: str = "delegate_trustee_agreement_by_demo.pdf"
):
    """
    Generate delegate–trustee agreement plots (by demographic) for multiple policies
    and save them into a single PDF.
    """
    with PdfPages(output_file) as pdf:
        for policy_index in policy_indices:
            try:
                df = create_delegate_trustee_agreement_dataframe(
                    policy_index=policy_index,
                    prompt_nums=prompt_nums,
                    model=model,
                    trustee_type=trustee_type,
                    bio_df=bio_df,
                    demographic=demographic,
                    alphas=alphas
                )

                plt.figure()
                plot_delegate_trustee_agreement_by_demo(
                    df,
                    demographic=demographic,
                    policy_index=policy_index,
                    trustee_type=trustee_type,
                    show_plot=False  # don’t show interactively
                )
                pdf.savefig()   # save current figure
                plt.close()     # free memory
                print(f"Saved policy {policy_index+1} to PDF")
            except Exception as e:
                print(f"Error with policy {policy_index+1}: {e}")

    print(f"All delegate–trustee agreement plots saved to {output_file}")

# # %%
# model = "claude-3-sonnet-v2"
# trustee_type = "trustee_ls"
# demographic = "Political Affiliation"
# # alphas = np.arange(0.0, 1.01, 0.1)
# alphas = [1]
# save_all_policy_plots_to_pdf(
#     policy_indices=range(20,30),          # all 30 policies
#     prompt_nums=[0, 1, 2],             # whichever prompts you use
#     model=model,
#     trustee_type=trustee_type,
#     bio_df=bio_df,
#     demographic=demographic,
#     alphas=alphas,
#     compare_expert=True,
#     output_file=f"agreement visuals/{model}/{trustee_type}/{demographic}_expert_agreement_alpha_1.pdf"
# )

# save_delegate_trustee_agreement_plots_to_pdf(
#     policy_indices=range(20),           # loop through first 20 policies
#     prompt_nums=[0, 1, 2],              # whichever prompts you want
#     model=model,
#     trustee_type=trustee_type,
#     bio_df=bio_df,
#     demographic=demographic,
#     alphas=alphas,
#     output_file=f"agreement visuals/{model}/{trustee_type}/{demographic}_delegate_trustee_agreement_alpha_1.pdf"
# )