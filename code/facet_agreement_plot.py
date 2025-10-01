#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
import os
from typing import List, Optional, Tuple
from matplotlib.lines import Line2D
from agreement_plotting import plot_mean_across_policies

def create_facet_agreement_plot(
    policy_indices: List[int],
    delegate_prompt_nums: List[int],
    trustee_prompt_nums: List[int],
    figsize: Tuple[int, int] = (22, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a facet plot comparing agreement with model defaults vs expert consensus.

    For single policy: Creates 1x2 plot (one row) based on whether policy has expert consensus
    For multiple policies: Creates 2x2 plot comparing model defaults (top) vs expert consensus (bottom)

    Args:
        policy_indices (List[int]): List of 0-based policy indices to analyze
        delegate_prompt_nums (List[int]): List of delegate prompt numbers
        trustee_prompt_nums (List[int]): List of trustee prompt numbers
        figsize (Tuple[int, int]): Overall figure size (default: (22, 12))
        save_path (Optional[str]): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """

    # Detect single policy mode
    single_policy_mode = len(policy_indices) == 1

    # Load policy data if single policy
    policy_statement = None
    has_expert_vote = False
    is_consensus = None
    expert_vote = None
    if single_policy_mode:
        policies_df = pd.read_json("../self_selected_policies_new.jsonl", lines=True)
        policy_data = policies_df.iloc[policy_indices[0]]
        policy_statement = policy_data['statement']
        print(policy_data)
        is_consensus = policy_data.get('consensus', 'Unknown')
        expert_vote = policy_data.get('expert_vote')
        has_expert_vote = pd.notna(expert_vote)

    # Create subplot grid based on mode
    if single_policy_mode:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]//2), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    # Model names and configurations - organized as Claude models (left) and GPT models (right)
    model_groups = [
        # Left column: Claude models
        {
            "models": ["claude-3-sonnet-v2", "claude-3-haiku-v2-mini"],
            "display_names": ["Claude 3 Sonnet", "Claude 3 Haiku"],
            "col": 0,
            "title": "Claude Models"
        },
        # Right column: GPT-4o models
        {
            "models": ["gpt-4o-mini", "gpt-4o"],
            "display_names": ["GPT-4o-mini", "GPT-4o"],
            "col": 1,
            "title": "GPT-4o Models"
        }
    ]

    # Define subplot configurations
    if single_policy_mode:
        # Single row based on expert vote availability
        if has_expert_vote:
            configs = [{"row": 0, "consensus_filter": None, "compare_expert": True, "ylabel": "Agreement w/ Expert Consensus"}]
        else:
            configs = [{"row": 0, "consensus_filter": None, "compare_expert": False, "ylabel": "Agreement w/ Model Default"}]
    else:
        # Two rows for multiple policies
        configs = [
            # Top row: Agreement with Model Defaults
            {"row": 0, "consensus_filter": "No", "compare_expert": False, "ylabel": "Agreement w/ Model Default"},
            # Bottom row: Agreement with Expert Consensus
            {"row": 1, "consensus_filter": "Yes", "compare_expert": True, "ylabel": "Agreement w/ Expert Consensus"}
        ]

    # Colors and styles
    trustee_color_dark = '#1f77b4'  # Dark blue
    trustee_color_light = '#87ceeb'  # Light blue
    delegate_color_dark = '#d62728'  # Dark red
    delegate_color_light = '#ff7f7f'  # Light red

    # Store data for legend creation
    legend_elements = []
    legend_created = False

    # Parameter range
    alphas = np.arange(0.0, 1.01, 0.1)

    # Define colors for different models
    # Mean line colors
    trustee_colors_by_model = {
        "claude-3-sonnet-v2": '#000000',      # Black
        "claude-3-haiku-v2-mini": '#0000FF',  # Blue
        "gpt-4o-mini": '#FFA500',             # Orange
        "gpt-4o": '#FF0000'                   # Red
    }

    delegate_colors_by_model = {
        "claude-3-sonnet-v2": '#000000',      # Black
        "claude-3-haiku-v2-mini": '#0000FF',  # Blue
        "gpt-4o-mini": '#FFA500',             # Orange
        "gpt-4o": '#FF0000'                   # Red
    }

    # Individual prompt colors (lighter versions)
    individual_prompt_colors = {
        "claude-3-sonnet-v2": '#808080',      # Grey
        "claude-3-haiku-v2-mini": '#ADD8E6',  # Light blue
        "gpt-4o-mini": '#FFD580',             # Light orange
        "gpt-4o": '#FFB6C1'                   # Pink
    }

    model_reference_votes = {}
    # Process each subplot
    for config in configs:
        row = config["row"]

        for group in model_groups:
            col = group["col"]
            # Handle axes indexing based on mode
            if single_policy_mode:
                ax = axes[col]
            else:
                ax = axes[row, col]

            for model, display_name in zip(group["models"], group["display_names"]):
                # Get data using plot_mean_across_policies
                df, reference_vote = plot_mean_across_policies(
                    policy_indices=policy_indices,
                    delegate_prompt_nums=delegate_prompt_nums,
                    trustee_prompt_nums=trustee_prompt_nums,
                    model=model,
                    trustee_type="both",
                    consensus_filter=config["consensus_filter"],
                    compare_expert=config["compare_expert"],
                    show_plot=False
                )
                model_reference_votes[display_name] = reference_vote
                # Plot individual trustee_ls prompts (thin, light, model-specific color)
                for prompt_num in trustee_prompt_nums:
                    col_name = f'trustee_ls_prompt_{prompt_num}_mean'
                    if col_name in df.columns:
                        ax.plot(alphas, df[col_name],
                               color=individual_prompt_colors[model], linewidth=1, alpha=0.5,
                               linestyle='-', zorder=1)

                # Plot individual trustee_lsd prompts (thin, light, model-specific color)
                for prompt_num in trustee_prompt_nums:
                    col_name = f'trustee_lsd_prompt_{prompt_num}_mean'
                    if col_name in df.columns:
                        ax.plot(alphas, df[col_name],
                               color=individual_prompt_colors[model], linewidth=1, alpha=0.5,
                               linestyle='-', zorder=1)

                # Plot individual delegate prompts (thin, light, dashed, model-specific color)
                for prompt_num in delegate_prompt_nums:
                    col_name = f'delegate_prompt_{prompt_num}_mean'
                    if col_name in df.columns:
                        ax.plot(alphas, df[col_name],
                               color=individual_prompt_colors[model], linewidth=1, alpha=0.5,
                               linestyle=(0, (5, 5)), zorder=1)

                # Plot trustee overall mean for this model (thick line with model-specific color)
                if 'trustee_overall_mean' in df.columns:
                    line = ax.plot(alphas, df['trustee_overall_mean'],
                                 color=trustee_colors_by_model[model], linewidth=3, alpha=1.0,
                                 linestyle='-', zorder=3)
                    # Only add to legend once per model (during first row processing)
                    if row == 0 and display_name not in [label for _, label in legend_elements]:
                        legend_elements.append((line[0], display_name))

                # Plot delegate overall mean for this model (thick dashed line with model-specific color)
                if 'delegate_overall_mean' in df.columns:
                    ax.plot(alphas, df['delegate_overall_mean'],
                           color=delegate_colors_by_model[model], linewidth=3, alpha=1.0,
                           linestyle=(0, (5, 5)), zorder=3)

            # Format subplot
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            if single_policy_mode:
                ax.set_ylim(0.3, 1)
            else:
                ax.set_ylim(0.5, 1)

            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            # Set titles (always show in single policy mode, otherwise only top row)
            if single_policy_mode or row == 0:
                ax.set_title(group["title"], fontsize=12)

            # Set y-axis labels for leftmost column
            if col == 0:
                ax.set_ylabel(config["ylabel"], fontsize=10)

            # Set x-axis labels (always show in single policy mode, otherwise only bottom row)
            if single_policy_mode or row == 1:
                ax.set_xlabel("Long Term Weight", fontsize=10)

    # Create Condition legend (bottom left)
    condition_handles = [
        Line2D([0], [0], color='darkgrey', linewidth=3, linestyle='--'),
        Line2D([0], [0], color='darkgrey', linewidth=3, linestyle=(0, (5, 5)))
    ]
    condition_labels = ["Trustee", "Delegate"]

    # Adjust legend position based on mode
    legend_y = -0.15 if single_policy_mode else 0.005

    legend1 = fig.legend(condition_handles, condition_labels,
                        loc="lower left", bbox_to_anchor=(0.05, legend_y),
                        fontsize=12, frameon=True, title="Condition",
                        ncol=2, borderaxespad=0, handlelength=2, handleheight=1.5)

    # Create Model legend (bottom center-right)
    if legend_elements:
        model_handles, model_labels = zip(*legend_elements)
        legend2 = fig.legend(model_handles, model_labels,
                           loc="lower left", bbox_to_anchor=(0.35, legend_y),
                           fontsize=12, frameon=True, title="Model",
                           ncol=4, borderaxespad=0, handlelength=2, handleheight=1.5)

    # Add text boxes - policy statement for single policy, topics for multiple policies
    if single_policy_mode:
        # Wrap policy statement text
        wrapped_policy = "\n".join(textwrap.wrap(policy_statement, width=28))
        fig.text(0.78, 0.62, wrapped_policy,
                fontsize=10, va='center', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=0.5))

        # Display the reference vote for each model below the wrapped policy statement
        # model_reference_votes is expected to be a dict: {model_name: reference_vote}
        if isinstance(model_reference_votes, dict):
            # Format as "Model: Vote" per line
            ref_vote_lines = [f"{model}: {vote}" for model, vote in model_reference_votes.items()]
            default_or_expert = "Model Default" if is_consensus == "No" else "Expert Consensus"
            if is_consensus == "No":
                ref_vote_lines = [f"{model}: {vote}" for model, vote in model_reference_votes.items()]
                ref_vote_text = f"{default_or_expert} Vote\n" + "\n".join(ref_vote_lines)
            else:
                ref_vote_text = f"Expert Consensus: {expert_vote}"
        else:
            # Fallback if not a dict (for backward compatibility)
            ref_vote_text = f"{default_or_expert} Vote {model_reference_votes}"

        fig.text(0.78, 0.46, ref_vote_text,
                 fontsize=10, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=0.5))
        # Set suptitle based on expert vote
        if has_expert_vote:
            fig.suptitle(f"Agreement with Expert Consensus",
                        fontsize=16, y=1.00)
        else:
            fig.suptitle(f"Agreement with Model Default",
                        fontsize=16, y=1.00)
    else:
        # Top row: No consensus topics (Model Defaults)
        no_consensus_topics = [
            "Minimum Wage",
            "Abortion",
            "Race/Gender in Hiring",
            "Universal Healthcare",
            "Sex Education",
            "Eat Less Meat",
            "Immigration",
            "Crime Sentences",
            "Government Pension",
            "Housing for the Homeless"
        ]
        no_consensus_text = "\n".join(no_consensus_topics)
        fig.text(1.02, 0.72, no_consensus_text,
                fontsize=12, va='center', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=0.5))

        # Bottom row: Expert consensus topics
        expert_topics = [
            "GMOs Safe",
            "Childhood Vaccination",
            "Free Trade",
            "Water Fluoridation",
            "Limiting Carbon Emissions"
        ]
        expert_text = "\n".join(expert_topics)
        fig.text(1.02, 0.33, expert_text,
                fontsize=12, va='center', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=0.5))

        # Set overall title
        fig.suptitle("Agreement with Model Defaults and Expert Consensus",
                    fontsize=16, y=1.00)

    # Adjust layout based on mode
    plt.tight_layout()
    if single_policy_mode:
        plt.subplots_adjust(top=0.88, right=0.75, bottom=0.12, left=0.06)
    else:
        plt.subplots_adjust(top=0.92, right=0.90, bottom=0.14, left=0.06)

    # Save if path provided
    if save_path:
        # For single policy mode, use special folder structure
        if single_policy_mode:
            policy_idx = policy_indices[0]
            save_dir = "../data/plots/line_agreement_plots"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"policy_{policy_idx}.png")

        plt.savefig(save_path, dpi=500, bbox_inches='tight')

    return fig


# Example usage
if __name__ == "__main__":
    # Example: Create facet plot for all 30 policies
    fig = create_facet_agreement_plot(
        #policy_indices=list(range(30)),
        policy_indices = [29],
        delegate_prompt_nums=[0, 1, 2, 3, 4],
        trustee_prompt_nums=[0, 1, 2],
        figsize=(12, 8),
        save_path="../data/plots/facet_agreement_plot.png"
    )
    # plt.show()  # Commented out - plot is saved instead
# %%
