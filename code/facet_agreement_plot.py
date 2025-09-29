#%%
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
from agreement_plotting import plot_mean_across_policies

def create_facet_agreement_plot(
    policy_indices: List[int],
    delegate_prompt_nums: List[int],
    trustee_prompt_nums: List[int],
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a 2x2 facet plot comparing agreement with model defaults vs expert consensus.

    Args:
        policy_indices (List[int]): List of 0-based policy indices to analyze
        delegate_prompt_nums (List[int]): List of delegate prompt numbers
        trustee_prompt_nums (List[int]): List of trustee prompt numbers
        figsize (Tuple[int, int]): Overall figure size (default: (16, 12))
        save_path (Optional[str]): Path to save the plot

    Returns:
        plt.Figure: The matplotlib figure object
    """

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    # Model names and configurations
    models = ["claude-3-sonnet-v2", "gpt-4o"]
    model_display_names = ["Claude 3 Sonnet", "GPT-4o"]

    # Define subplot configurations
    configs = [
        # Top row: Agreement with Model Defaults
        {"row": 0, "consensus_filter": "No", "compare_expert": False, "ylabel": "Agreement Rate with Model Default"},
        # Bottom row: Agreement with Expert Consensus
        {"row": 1, "consensus_filter": "Yes", "compare_expert": True, "ylabel": "Agreement Rate with Expert Consensus"}
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

    # Process each subplot
    for config in configs:
        row = config["row"]

        for col, (model, model_display) in enumerate(zip(models, model_display_names)):
            ax = axes[row, col]

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

            # Plot individual trustee_ls prompts (thin, light blue)
            for prompt_num in trustee_prompt_nums:
                col_name = f'trustee_ls_prompt_{prompt_num}_mean'
                if col_name in df.columns:
                    ax.plot(alphas, df[col_name],
                           color=trustee_color_light, linewidth=1, alpha=0.8,
                           linestyle='-', zorder=1)

            # Plot individual trustee_lsd prompts (thin, light blue)
            for prompt_num in trustee_prompt_nums:
                col_name = f'trustee_lsd_prompt_{prompt_num}_mean'
                if col_name in df.columns:
                    ax.plot(alphas, df[col_name],
                           color=trustee_color_light, linewidth=1, alpha=0.8,
                           linestyle='-', zorder=1)

            # Plot individual delegate prompts (thin, light red, dashed)
            for prompt_num in delegate_prompt_nums:
                col_name = f'delegate_prompt_{prompt_num}_mean'
                if col_name in df.columns:
                    ax.plot(alphas, df[col_name],
                           color=delegate_color_light, linewidth=1, alpha=0.8,
                           linestyle=(0, (5, 5)), zorder=1)  # Custom dash pattern

            # Plot trustee overall mean (thick, dark blue)
            if 'trustee_overall_mean' in df.columns:
                line = ax.plot(alphas, df['trustee_overall_mean'],
                             color=trustee_color_dark, linewidth=3, alpha=1.0,
                             linestyle='-', zorder=3)
                if not legend_created:
                    legend_elements.append((line[0], "Trustee Mean"))

            # Plot delegate overall mean (thick, dark red, dashed)
            if 'delegate_overall_mean' in df.columns:
                line = ax.plot(alphas, df['delegate_overall_mean'],
                             color=delegate_color_dark, linewidth=3, alpha=1.0,
                             linestyle=(0, (5, 5)), zorder=3)  # Custom dash pattern
                if not legend_created:
                    legend_elements.append((line[0], "Delegate Mean"))

            # Format subplot
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.4, 1)

            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

            # Set titles for top row
            if row == 0:
                ax.set_title(model_display, fontsize=14)

            # Set y-axis labels for leftmost column
            if col == 0:
                ax.set_ylabel(config["ylabel"], fontsize=12)

            # Set x-axis labels for bottom row
            if row == 1:
                ax.set_xlabel("Long Term Weight", fontsize=12)

            # Mark that we've created legend elements
            if not legend_created:
                legend_created = True

    # Create shared legend in upper right
    if legend_elements:
        # Position legend in upper right area
        handles, labels = zip(*legend_elements)

        # Reorder legend elements for better organization
        # We want: Trustee Mean, Delegate Mean, individual trustee types
        ordered_handles = []
        ordered_labels = []

        # Add means first
        for i, label in enumerate(labels):
            if "Mean" in label:
                ordered_handles.append(handles[i])
                ordered_labels.append(labels[i])

        # Add individual types
        for i, label in enumerate(labels):
            if "Mean" not in label:
                ordered_handles.append(handles[i])
                ordered_labels.append(labels[i])

        fig.legend(ordered_handles, ordered_labels,
                  bbox_to_anchor=(0.98, 0.98), loc='upper right',
                  fontsize=11, frameon=True, fancybox=True, shadow=True)

    # Set overall title
    fig.suptitle("Agreement with Model Defaults and Expert Consensus",
                fontsize=16, y=0.95)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.85)

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
#%%
fig = create_facet_agreement_plot(
    policy_indices=list(range(30)),
    delegate_prompt_nums=[0, 1, 2, 3, 4],
    trustee_prompt_nums=[0, 1, 2],
    figsize=(16, 12)
)
plt.show()

# # Example usage
# if __name__ == "__main__":
#     # Example: Create facet plot for policies 0-9 with different prompt configurations
#     fig = create_facet_agreement_plot(
#         policy_indices=list(range(10)),
#         delegate_prompt_nums=[0, 1, 2, 3, 4],
#         trustee_prompt_nums=[0, 1, 2],
#         figsize=(16, 12)
#     )
#     plt.show()
# %%
