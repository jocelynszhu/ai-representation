#%%
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from vote_data_loader import load_policy_votes
import json
import matplotlib.pyplot as plt

def plot_trustee_utilities(
    policy_index: int,
    trustee_type: str,
    model: str = "claude-3-sonnet-v2",
    prompt_num: int = 0,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot utilities for a given policy based on trustee type.

    For trustee_ls: Shows bar chart of short-term vs long-term utilities for Yes/No votes
    For trustee_lsd: Shows line chart of utilities across time periods for Yes/No votes

    Args:
        policy_index (int): 0-based policy index
        trustee_type (str): Either "trustee_ls" or "trustee_lsd"
        model (str): Model name (default: "claude-3-sonnet-v2")
        prompt_num (int): Prompt number (default: 0)
        figsize (tuple): Figure size (default: (12, 8))
        save_path (str, optional): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Load the data
    data = load_policy_votes(model, trustee_type, policy_index, prompt_num)
    trustee_df = data['trustee']

    # Load policy title
    policy_title = _get_policy_title(policy_index)

    if trustee_type == "trustee_ls":
        return _plot_trustee_ls_utilities(trustee_df, policy_index, policy_title, figsize, save_path)
    elif trustee_type == "trustee_lsd":
        return _plot_trustee_lsd_utilities(trustee_df, policy_index, policy_title, figsize, save_path)
    else:
        raise ValueError(f"Unsupported trustee_type: {trustee_type}")


def _plot_trustee_ls_utilities(
    df: pd.DataFrame,
    policy_index: int,
    policy_title: str,
    figsize: Tuple[int, int],
    save_path: Optional[str]
) -> plt.Figure:
    """Plot utilities for trustee_ls format (short-term vs long-term)."""

    # Calculate average utilities
    yes_short_avg = df['yes_short_util'].mean()
    yes_long_avg = df['yes_long_util'].mean()
    no_short_avg = df['no_short_util'].mean()
    no_long_avg = df['no_long_util'].mean()

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x = np.arange(2)  # Short-term, Long-term
    width = 0.35

    # Create bars
    yes_bars = ax.bar(x - width/2, [yes_short_avg, yes_long_avg], width,
                     label='Yes Vote', color='lightgreen', alpha=0.8)
    no_bars = ax.bar(x + width/2, [no_short_avg, no_long_avg], width,
                    label='No Vote', color='lightcoral', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Time Horizon')
    ax.set_ylabel('Average Utility')
    ax.set_title(f'Average Utilities by Time Horizon\nPolicy {policy_index + 1}: {policy_title}')
    ax.set_xticks(x)
    ax.set_xticklabels(['Short-term', 'Long-term'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [yes_bars, no_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def _plot_trustee_lsd_utilities(
    df: pd.DataFrame,
    policy_index: int,
    policy_title: str,
    figsize: Tuple[int, int],
    save_path: Optional[str]
) -> plt.Figure:
    """Plot utilities for trustee_lsd format (time periods)."""

    # Time periods
    periods = ["0_5_years", "5_10_years", "10_15_years", "15_20_years", "20_25_years", "25_30_years"]
    period_labels = ["0-5 years", "5-10 years", "10-15 years", "15-20 years", "20-25 years", "25-30 years"]

    # Calculate average utilities for each time period
    yes_utilities = []
    no_utilities = []

    for period in periods:
        yes_col = f'yes_{period}_score'
        no_col = f'no_{period}_score'

        if yes_col in df.columns and no_col in df.columns:
            yes_utilities.append(df[yes_col].mean())
            no_utilities.append(df[no_col].mean())
        else:
            yes_utilities.append(0)  # Default if column missing
            no_utilities.append(0)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines
    x = np.arange(len(periods))
    ax.plot(x, yes_utilities, marker='o', linewidth=2, markersize=8,
           label='Yes Vote', color='green', alpha=0.8)
    ax.plot(x, no_utilities, marker='s', linewidth=2, markersize=8,
           label='No Vote', color='red', alpha=0.8)

    # Customize the plot
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Average Utility Score')
    ax.set_title(f'Average Utilities Across Time Periods\nPolicy {policy_index + 1}: {policy_title}')
    ax.set_xticks(x)
    ax.set_xticklabels(period_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on points
    for i, (yes_val, no_val) in enumerate(zip(yes_utilities, no_utilities)):
        ax.annotate(f'{yes_val:.1f}', (i, yes_val), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{no_val:.1f}', (i, no_val), textcoords="offset points",
                   xytext=(0,-15), ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def _get_policy_title(policy_index: int) -> str:
    """Load policy title from self_selected_policies_new.json."""
    try:
        with open('../self_selected_policies_new.jsonl', 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == policy_index:
                    policy_data = json.loads(line.strip())
                    return policy_data.get('policy', f'Policy {policy_index + 1}')
        return f'Policy {policy_index + 1}'
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        return f'Policy {policy_index + 1}'


def plot_utilities_comparison(
    policy_indices: List[int],
    trustee_type: str,
    model: str = "claude-3-sonnet-v2",
    prompt_num: int = 0,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot utilities comparison across multiple policies.

    Args:
        policy_indices (List[int]): List of 0-based policy indices
        trustee_type (str): Either "trustee_ls" or "trustee_lsd"
        model (str): Model name (default: "claude-3-sonnet-v2")
        prompt_num (int): Prompt number (default: 0)
        figsize (tuple): Figure size (default: (15, 10))
        save_path (str, optional): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """
    n_policies = len(policy_indices)

    if trustee_type == "trustee_ls":
        fig, axes = plt.subplots(1, n_policies, figsize=figsize, sharey=True)
        if n_policies == 1:
            axes = [axes]

        for i, policy_idx in enumerate(policy_indices):
            data = load_policy_votes(model, trustee_type, policy_idx, prompt_num)
            df = data['trustee']
            policy_title = _get_policy_title(policy_idx)

            # Calculate averages
            yes_short_avg = df['yes_short_util'].mean()
            yes_long_avg = df['yes_long_util'].mean()
            no_short_avg = df['no_short_util'].mean()
            no_long_avg = df['no_long_util'].mean()

            # Plot
            x = np.arange(2)
            width = 0.35
            axes[i].bar(x - width/2, [yes_short_avg, yes_long_avg], width,
                       label='Yes Vote', color='lightgreen', alpha=0.8)
            axes[i].bar(x + width/2, [no_short_avg, no_long_avg], width,
                       label='No Vote', color='lightcoral', alpha=0.8)

            axes[i].set_title(f'Policy {policy_idx + 1}\n{policy_title[:30]}...' if len(policy_title) > 30 else f'Policy {policy_idx + 1}\n{policy_title}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(['Short', 'Long'])
            axes[i].grid(True, alpha=0.3)

            if i == 0:
                axes[i].set_ylabel('Average Utility')
                axes[i].legend()

    elif trustee_type == "trustee_lsd":
        fig, axes = plt.subplots(1, n_policies, figsize=figsize, sharey=True)
        if n_policies == 1:
            axes = [axes]

        periods = ["0_5_years", "5_10_years", "10_15_years", "15_20_years", "20_25_years", "25_30_years"]
        period_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30"]

        for i, policy_idx in enumerate(policy_indices):
            data = load_policy_votes(model, trustee_type, policy_idx, prompt_num)
            df = data['trustee']
            policy_title = _get_policy_title(policy_idx)

            yes_utilities = []
            no_utilities = []

            for period in periods:
                yes_col = f'yes_{period}_score'
                no_col = f'no_{period}_score'

                if yes_col in df.columns and no_col in df.columns:
                    yes_utilities.append(df[yes_col].mean())
                    no_utilities.append(df[no_col].mean())
                else:
                    yes_utilities.append(0)
                    no_utilities.append(0)

            x = np.arange(len(periods))
            axes[i].plot(x, yes_utilities, marker='o', label='Yes Vote', color='green')
            axes[i].plot(x, no_utilities, marker='s', label='No Vote', color='red')

            axes[i].set_title(f'Policy {policy_idx + 1}\n{policy_title[:30]}...' if len(policy_title) > 30 else f'Policy {policy_idx + 1}\n{policy_title}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(period_labels, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)

            if i == 0:
                axes[i].set_ylabel('Average Utility Score')
                axes[i].legend()

    plt.suptitle(f'Utility Comparison Across Policies ({trustee_type.upper()})')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

#%%
fig3 = plot_utilities_comparison(
    policy_indices=[2],
    trustee_type="trustee_lsd",
    model="gpt-4o",
    prompt_num=1
)
plt.show()
#%%
# Example usage
if __name__ == "__main__":
    # Example 1: Single policy with trustee_ls
    # fig1 = plot_trustee_utilities(
    #     policy_index=0,
    #     trustee_type="trustee_ls",
    #     model="claude-3-sonnet-v2",
    #     prompt_num=0
    # )
    # plt.show()

    # # Example 2: Single policy with trustee_lsd
    # fig2 = plot_trustee_utilities(
    #     policy_index=4,
    #     trustee_type="trustee_lsd",
    #     model="claude-3-sonnet-v2",
    #     prompt_num=0
    # )
    # plt.show()

    # Example 3: Compare multiple policies
