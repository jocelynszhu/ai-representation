#%%
import matplotlib.pyplot as plt
import numpy as np
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
    Create a 2x2 facet plot comparing agreement with model defaults vs expert consensus.

    Args:
        policy_indices (List[int]): List of 0-based policy indices to analyze
        delegate_prompt_nums (List[int]): List of delegate prompt numbers
        trustee_prompt_nums (List[int]): List of trustee prompt numbers
        figsize (Tuple[int, int]): Overall figure size (default: (22, 12))
        save_path (Optional[str]): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """

    # Create 2x2 subplot grid
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

    # Process each subplot
    for config in configs:
        row = config["row"]

        for group in model_groups:
            col = group["col"]
            ax = axes[row, col]

            for model, display_name in zip(group["models"], group["display_names"]):
                # Get data using plot_mean_across_policies
                df = plot_mean_across_policies(
                    policy_indices=policy_indices,
                    delegate_prompt_nums=delegate_prompt_nums,
                    trustee_prompt_nums=trustee_prompt_nums,
                    model=model,
                    trustee_type="both",
                    consensus_filter=config["consensus_filter"],
                    compare_expert=config["compare_expert"],
                    show_plot=False
                )

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
            ax.set_ylim(0.5, 1)

            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            # Set titles for top row
            if row == 0:
                ax.set_title(group["title"], fontsize=12)

            # Set y-axis labels for leftmost column
            if col == 0:
                ax.set_ylabel(config["ylabel"], fontsize=10)

            # Set x-axis labels for bottom row
            if row == 1:
                ax.set_xlabel("Long Term Weight", fontsize=10)

    # Create Condition legend (bottom left)
    condition_handles = [
        Line2D([0], [0], color='darkgrey', linewidth=3, linestyle='--'),
        Line2D([0], [0], color='darkgrey', linewidth=3, linestyle=(0, (5, 5)))
    ]
    condition_labels = ["Trustee", "Delegate"]

    legend1 = fig.legend(condition_handles, condition_labels,
                        loc="lower left", bbox_to_anchor=(0.05, 0.005),
                        fontsize=12, frameon=True, title="Condition",
                        ncol=2, borderaxespad=0, handlelength=2, handleheight=1.5)

    # Create Model legend (bottom center-right)
    if legend_elements:
        model_handles, model_labels = zip(*legend_elements)
        legend2 = fig.legend(model_handles, model_labels,
                           loc="lower left", bbox_to_anchor=(0.35, 0.005),
                           fontsize=12, frameon=True, title="Model",
                           ncol=4, borderaxespad=0, handlelength=2, handleheight=1.5)

    # Add topic text boxes
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

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.90, bottom=0.14, left=0.06)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')

    return fig


# Example usage
if __name__ == "__main__":
    # Example: Create facet plot for all 30 policies
    fig = create_facet_agreement_plot(
        policy_indices=list(range(30)),
        delegate_prompt_nums=[0, 1, 2, 3, 4],
        trustee_prompt_nums=[0, 1, 2],
        figsize=(12, 8),
        save_path="../data/plots/facet_agreement_plot.png"
    )
    # plt.show()  # Commented out - plot is saved instead
# %%
