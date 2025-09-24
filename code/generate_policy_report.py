#!/usr/bin/env python3
"""
Policy Disagreement Report Generator
Generates PDF reports with 2x2 grid layouts showing disagreement plots for multiple policies.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys

# Import the comparison functions from compare_delegates_trustees.py
from compare_delegates_trustees import plot_disagreement_by_delegate_prompts, plot_all_policies_overview

def generate_policy_report(model, policies_list, delegate_prompt_nums, trustee_prompt_num=0,
                          trustee_format='trustee_ls', output_file=None, policies_per_page=4):
    """
    Generate a PDF report with disagreement plots for multiple policies in 2x2 grid layout.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2" or "gpt-4o")
        policies_list (list): List of policy indices to analyze (0-based)
        delegate_prompt_nums (list): List of delegate prompt numbers to compare
        trustee_prompt_num (int): Trustee prompt number to use for all comparisons
        trustee_format (str): Trustee data format ('trustee_ls' or 'trustee_lsd')
        output_file (str): Output PDF filename (default: auto-generated)
        policies_per_page (int): Number of policies per page (default: 4 for 2x2 grid)

    Returns:
        str: Path to generated PDF file
    """

    # Load policy statements
    policies_df = pd.read_json("../self_selected_policies.jsonl", lines=True)

    # Generate default filename if not provided
    if output_file is None:
        output_file = f"../data/plots/{model}_{trustee_prompt_num}_{trustee_format}.pdf"
    elif not output_file.startswith('../data/plots/') and not os.path.isabs(output_file):
        # If relative path provided, put it in plots directory
        output_file = f"../data/plots/{output_file}"

    # Ensure plots directory exists
    os.makedirs("../data/plots", exist_ok=True)

    print(f"Generating PDF report: {output_file}")
    print(f"Model: {model}")
    print(f"Policies: {policies_list}")
    print(f"Delegate prompts: {delegate_prompt_nums}")
    print(f"Trustee prompt: {trustee_prompt_num}")
    print(f"Trustee format: {trustee_format}")
    print(f"Layout: {policies_per_page} policies per page")
    print("=" * 80)

    # Calculate grid dimensions for 2x2 layout
    if policies_per_page == 4:
        rows, cols = 2, 2
    else:
        # For other layouts, calculate square-ish grid
        rows = int(np.ceil(np.sqrt(policies_per_page)))
        cols = int(np.ceil(policies_per_page / rows))

    # Create PDF
    with matplotlib.backends.backend_pdf.PdfPages(output_file) as pdf:

        # Generate overview plot as first page
        print(f"\nGenerating overview page...")
        try:
            # Create full-page figure for overview
            fig, ax = plt.subplots(1, 1, figsize=(16, 12))

            # Generate overview data (suppress display)
            overview_results = plot_all_policies_overview(
                model, policies_list, delegate_prompt_nums, trustee_prompt_num, trustee_format, show_plot=False
            )

            if overview_results and 'all_curves' in overview_results:
                weights = overview_results['weights']
                all_curves = overview_results['all_curves']
                overall_mean = overview_results['overall_mean']
                successful_combinations = overview_results['successful_combinations']

                # Plot individual curves as very light red lines
                for i, curve in enumerate(all_curves):
                    policy_idx, delegate_idx = successful_combinations[i]
                    plt.plot(weights, curve, color='#ff9999', alpha=0.2, linewidth=0.5)

                # Plot overall mean as thick black line
                plt.plot(weights, overall_mean, color='black', linewidth=4, label='Overall Mean', alpha=0.9)

                # Add legend entry for individual lines
                plt.plot([], [], color='#ff9999', alpha=0.2, linewidth=0.5, label='Individual Policy-Prompt Combinations')

                # Format plot
                plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=14)
                plt.ylabel('Disagreement Rate', fontsize=14)
                plt.title(f'Disagreement Patterns Overview - All Policies and Delegate Prompts\n{model}, Trustee Prompt {trustee_prompt_num}, {len(all_curves)} combinations',
                         fontsize=16, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                plt.ylim(min(overall_mean) - .05, max(overall_mean) + .05)
                plt.xlim(0, 2)
                plt.legend(loc='upper right', fontsize=12)

                # Format y-axis as percentages
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

                print(f"  ✓ Overview completed with {len(all_curves)} policy-delegate combinations")
            else:
                # Create error message if no data
                plt.text(0.5, 0.5, 'No data available for overview plot',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=16, color='red')
                plt.title('Overview Plot (No Data Available)', fontsize=16, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                print(f"  ⚠ No data available for overview")

            # Save overview page
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  ✓ Overview page completed")

        except Exception as e:
            print(f"  ✗ Error generating overview: {e}")
            # Create error page
            fig, ax = plt.subplots(1, 1, figsize=(16, 12))
            plt.text(0.5, 0.5, f'Error generating overview plot:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='red')
            plt.title('Overview Plot (Error)', fontsize=16, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)

        # Process policies in batches for detailed 2x2 grids
        for page_start in range(0, len(policies_list), policies_per_page):
            page_policies = policies_list[page_start:page_start + policies_per_page]

            print(f"\nGenerating page {page_start//policies_per_page + 2} with policies: {[p+1 for p in page_policies]}")

            # Create figure for this page
            fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
            if policies_per_page == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Process each policy on this page
            for i, policy_index in enumerate(page_policies):
                print(f"  Processing policy {policy_index + 1}...")

                try:
                    # Set current axes
                    plt.sca(axes[i])

                    # Generate the disagreement plot (suppress display)
                    results = plot_disagreement_by_delegate_prompts(
                        model, policy_index, delegate_prompt_nums,
                        trustee_prompt_num, trustee_format, show_plot=False
                    )

                    # Plot the results manually on current axes
                    weights = np.arange(0.0, 1.01, 0.1) # changed from 2.01 to 1.01
                    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                    line_styles = ['--', '--', '--', '--']

                    mean_disagreement_rates = []
                    for j, delegate_prompt_num in enumerate(delegate_prompt_nums):
                        if delegate_prompt_num in results:
                            results_df = results[delegate_prompt_num]
                            color = colors[j % len(colors)]
                            line_style = line_styles[j % len(line_styles)]

                            plt.plot(results_df['long_term_weight'], results_df['disagreement_rate'],
                                   color=color, linestyle=line_style, linewidth=1.5,
                                   markersize=4, marker='o',
                                   label=f'Delegate Prompt {delegate_prompt_num}', alpha=0.7)

                            mean_disagreement_rates.append(results_df['disagreement_rate'].values)

                    # Calculate and plot mean
                    if mean_disagreement_rates:
                        mean_disagreement_rates = np.array(mean_disagreement_rates)
                        mean_across_prompts = np.nanmean(mean_disagreement_rates, axis=0)

                        plt.plot(weights, mean_across_prompts,
                               color='black', linewidth=3, label='Mean Disagreement', alpha=0.9)

                    # Add trustee policy support line if available
                    if 'trustee_support' in results:
                        support_data = results['trustee_support']
                        plt.plot(support_data['weights'], support_data['support_rates'],
                               color='purple', linewidth=2, linestyle='-',
                               marker='s', markersize=3, label='Trustee Policy Support', alpha=0.8)

                    # Get policy statement for title
                    policy_statement = policies_df.iloc[policy_index]['statement']
                    # Truncate title if too long (wrap at ~60 characters)
                    if len(policy_statement) > 60:
                        policy_title = policy_statement[:57] + "..."
                    else:
                        policy_title = policy_statement

                    # Format subplot
                    plt.title(policy_title, fontsize=10, fontweight='bold', wrap=True, pad=10)
                    plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=10)
                    plt.ylabel('Rate', fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.ylim(0, 1)  # Set to 0-100% to accommodate both disagreement and support rates
                    plt.xlim(0, 1) # changed from 2 to 1

                    # Format y-axis as percentages
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

                    # Add legend only to first subplot to avoid clutter
                    if i == 0:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

                    print(f"    ✓ Policy {policy_index + 1} completed")

                except Exception as e:
                    print(f"    ✗ Error processing policy {policy_index + 1}: {e}")

                    # Get policy statement for error title
                    try:
                        policy_statement = policies_df.iloc[policy_index]['statement']
                        if len(policy_statement) > 40:
                            error_title = policy_statement[:37] + "... (Error)"
                        else:
                            error_title = f"{policy_statement} (Error)"
                    except:
                        error_title = f'Policy {policy_index + 1} (Error)'

                    # Create empty plot with error message
                    plt.sca(axes[i])
                    plt.text(0.5, 0.5, f'Error: Policy {policy_index + 1}\n{str(e)}',
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, color='red')
                    plt.title(error_title, fontsize=10, color='red')
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])

            # Hide unused subplots
            for i in range(len(page_policies), len(axes)):
                axes[i].set_visible(False)

            # Add page title
            fig.suptitle(f'Disagreement Patterns by Delegate Prompt Type\\n{model}, Trustee Prompt {trustee_prompt_num}',
                        fontsize=16, fontweight='bold', y=0.95)

            # Adjust layout and save page
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)

            print(f"  ✓ Page {page_start//policies_per_page + 2} completed")

    print("=" * 80)
    print(f"✅ PDF report generated successfully: {output_file}")

    # Print file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"📄 File size: {file_size:.2f} MB")

    return output_file

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Generate PDF report of disagreement patterns across policies')

    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='Model name (claude-3-sonnet-v2 or gpt-4o) (default: claude-3-sonnet-v2)')
    parser.add_argument('--policies', type=str, default='0-19',
                       help='Policy range (e.g., "0-19" or "0,1,2,3") (default: 0-19)')
    parser.add_argument('--delegate-prompts', type=str, default='0,1,2,3',
                       help='Delegate prompt numbers (comma-separated) (default: 0,1,2,3)')
    parser.add_argument('--trustee-prompt', type=int, default=0,
                       help='Trustee prompt number (default: 0)')
    parser.add_argument('--trustee-format', type=str, default='trustee_ls',
                       choices=['trustee_ls', 'trustee_lsd'],
                       help='Trustee data format (default: trustee_ls)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDF filename (default: auto-generated)')
    parser.add_argument('--policies-per-page', type=int, default=4,
                       help='Number of policies per page (default: 4 for 2x2 grid)')

    args = parser.parse_args()

    # Parse policies list
    try:
        if '-' in args.policies:
            start, end = map(int, args.policies.split('-'))
            policies_list = list(range(start, end + 1))
        else:
            policies_list = [int(p.strip()) for p in args.policies.split(',')]
    except ValueError:
        print(f"Error: Invalid policies format '{args.policies}'. Use '0-19' or '0,1,2,3'")
        sys.exit(1)

    # Parse delegate prompts
    try:
        delegate_prompt_nums = [int(p.strip()) for p in args.delegate_prompts.split(',')]
    except ValueError:
        print(f"Error: Invalid delegate prompts format '{args.delegate_prompts}'. Use '0,1,2,3'")
        sys.exit(1)

    # Validate arguments
    if not policies_list:
        print("Error: No policies specified")
        sys.exit(1)

    if not delegate_prompt_nums:
        print("Error: No delegate prompts specified")
        sys.exit(1)

    try:
        # Generate report
        output_file = generate_policy_report(
            args.model, policies_list, delegate_prompt_nums,
            args.trustee_prompt, args.trustee_format, args.output, args.policies_per_page
        )

        print(f"\n🎉 Report generation completed successfully!")
        print(f"📁 Output file: {os.path.abspath(output_file)}")

    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()