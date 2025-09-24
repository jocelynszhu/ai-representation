#!/usr/bin/env python3
"""
Agreement Rate Chart Generator
Generates PDF reports comparing trustee agreement rates vs delegate agreement rates for multiple policies.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys
import json

# Import utility functions from existing modules
from compare_delegates_trustees import calculate_weighted_vote, calculate_discounted_vote


def calculate_trustee_agreement_rate(model, policy_index, weights, trustee_prompt_num=0, trustee_format='trustee_ls'):
    """
    Calculate trustee agreement rate (proportion voting "Yes") across different weight parameters.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2" or "gpt-4o")
        policy_index (int): Policy index (0-based)
        weights (list): List of weight parameters to analyze
        trustee_prompt_num (int): Trustee prompt number
        trustee_format (str): Trustee data format ('trustee_ls' or 'trustee_lsd')

    Returns:
        list: Agreement rates for each weight parameter
    """
    print(f"  Calculating trustee agreement rates for policy {policy_index + 1}...")

    # Construct trustee file path
    trustee_file = f"../data/{trustee_format}/{model}/prompt-{trustee_prompt_num}/t_policy_{policy_index+1}_votes.jsonl"    
    #trustee_file = f"../data/{model}_policy_{policy_index}_prompt_{trustee_prompt_num}_{trustee_format}.jsonl"

    if not os.path.exists(trustee_file):
        print(f"  Warning: Trustee file not found: {trustee_file}")
        return [np.nan] * len(weights)

    agreement_rates = []

    for weight in weights:
        try:
            # Load and process trustee data for this weight
            trustee_votes = []

            with open(trustee_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())

                        # Calculate vote based on format
                        if trustee_format == 'trustee_ls':
                            vote_result = calculate_weighted_vote(data, weight)
                        elif trustee_format == 'trustee_lsd':
                            vote_result = calculate_discounted_vote(data, weight)
                        else:
                            raise ValueError(f"Unsupported trustee format: {trustee_format}")

                        trustee_votes.append(vote_result['vote'])

                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        continue

            if trustee_votes:
                # Calculate agreement rate (proportion voting "Yes")
                yes_votes = sum(1 for vote in trustee_votes if vote == 'Yes')
                agreement_rate = yes_votes / len(trustee_votes)
                agreement_rates.append(agreement_rate)
            else:
                agreement_rates.append(np.nan)

        except Exception as e:
            print(f"  Error calculating trustee agreement at weight {weight:.2f}: {e}")
            agreement_rates.append(np.nan)

    return agreement_rates


def calculate_delegate_agreement_rate(model, policy_index, weights, delegate_prompt_nums, trustee_format='trustee_ls'):
    """
    Calculate delegate agreement rate (average proportion voting "Yes") across delegate prompts and weights.

    Args:
        model (str): Model name
        policy_index (int): Policy index (0-based)
        weights (list): List of weight parameters to analyze
        delegate_prompt_nums (list): List of delegate prompt numbers
        trustee_format (str): Format to use for weight calculation consistency

    Returns:
        dict: Agreement rates by delegate prompt, plus overall mean
    """
    print(f"  Calculating delegate agreement rates for policy {policy_index + 1}...")

    all_delegate_results = {}

    for delegate_prompt_num in delegate_prompt_nums:
        # Construct delegate file path
        delegate_file = f"../data/delegate/{model}/prompt-{delegate_prompt_num}/d_policy_{policy_index+1}_votes.jsonl"
        #delegate_file = f"../data/{model}_policy_{policy_index}_prompt_{delegate_prompt_num}_delegate.jsonl"

        if not os.path.exists(delegate_file):
            print(f"    Warning: Delegate file not found: {delegate_file}")
            all_delegate_results[delegate_prompt_num] = [np.nan] * len(weights)
            continue

        agreement_rates = []

        for weight in weights:
            try:
                # Load and process delegate data for this weight
                delegate_votes = []

                with open(delegate_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            delegate_votes.append(data['vote'])

                        except (json.JSONDecodeError, KeyError, Exception) as e:
                            continue

                if delegate_votes:
                    # Calculate agreement rate (proportion voting "Yes")
                    yes_votes = sum(1 for vote in delegate_votes if vote == 'Yes')
                    agreement_rate = yes_votes / len(delegate_votes)
                    agreement_rates.append(agreement_rate)
                else:
                    agreement_rates.append(np.nan)

            except Exception as e:
                print(f"    Error calculating delegate agreement for prompt {delegate_prompt_num} at weight {weight:.2f}: {e}")
                agreement_rates.append(np.nan)
        print(f"    Delegate {delegate_prompt_num} agreement rates: {agreement_rates}")
        all_delegate_results[delegate_prompt_num] = agreement_rates

    # Calculate mean across all delegate prompts
    all_rates = []
    for rates in all_delegate_results.values():
        if not all(np.isnan(r) for r in rates):
            all_rates.append(rates)

    if all_rates:
        mean_rates = np.nanmean(all_rates, axis=0)
        all_delegate_results['mean'] = mean_rates.tolist()
    else:
        all_delegate_results['mean'] = [np.nan] * len(weights)

    return all_delegate_results


def plot_agreement_comparison(model, policy_index, delegate_prompt_nums, trustee_prompt_num=0,
                             trustee_format='trustee_ls', show_plot=True):
    """
    Generate agreement rate comparison plot for a single policy.

    Args:
        model (str): Model name
        policy_index (int): Policy index (0-based)
        delegate_prompt_nums (list): List of delegate prompt numbers
        trustee_prompt_num (int): Trustee prompt number
        trustee_format (str): Trustee data format
        show_plot (bool): Whether to display the plot

    Returns:
        dict: Results data for trustee and delegate agreement rates
    """
    # Define weight range
    weights = np.arange(0.0, 2.01, 0.1)

    print(f"Processing policy {policy_index + 1} agreement rates...")

    # Calculate trustee agreement rates
    trustee_agreement_rates = calculate_trustee_agreement_rate(
        model, policy_index, weights, trustee_prompt_num, trustee_format
    )

    # Calculate delegate agreement rates
    delegate_agreement_results = calculate_delegate_agreement_rate(
        model, policy_index, weights, delegate_prompt_nums, trustee_format
    )

    # Create plot if requested
    if show_plot:
        plt.figure(figsize=(12, 8))

        # Plot trustee agreement rate
        plt.plot(weights, trustee_agreement_rates,
                color='blue', linewidth=3, marker='o', markersize=4,
                label=f'Trustee Agreement Rate (Prompt {trustee_prompt_num})', alpha=0.8)

        # Plot individual delegate prompt agreement rates
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, delegate_prompt_num in enumerate(delegate_prompt_nums):
            if delegate_prompt_num in delegate_agreement_results:
                rates = delegate_agreement_results[delegate_prompt_num]
                color = colors[i % len(colors)]
                plt.plot(weights, rates,
                        color=color, linewidth=2, linestyle='--', marker='s', markersize=3,
                        label=f'Delegate Agreement Rate (Prompt {delegate_prompt_num})', alpha=0.6)

        # Plot mean delegate agreement rate
        if 'mean' in delegate_agreement_results:
            plt.plot(weights, delegate_agreement_results['mean'],
                    color='red', linewidth=4, marker='D', markersize=4,
                    label='Mean Delegate Agreement Rate', alpha=0.9)

        # Load policy statement for title
        try:
            policies_df = pd.read_json("../self_selected_policies.jsonl", lines=True)
            policy_statement = policies_df.iloc[policy_index]['statement']
            if len(policy_statement) > 80:
                policy_title = policy_statement[:77] + "..."
            else:
                policy_title = policy_statement
        except:
            policy_title = f'Policy {policy_index + 1}'

        # Format plot
        plt.title(f'Agreement Rates Comparison: {policy_title}\\n{model}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=12)
        plt.ylabel('Agreement Rate (Proportion Voting "Yes")', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, 2)

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

        if show_plot:
            plt.show()

    # Return results
    results = {
        'weights': weights.tolist(),
        'trustee_agreement_rates': trustee_agreement_rates,
        'delegate_agreement_results': delegate_agreement_results
    }

    return results


def plot_all_policies_agreement_overview(model, policies_list, delegate_prompt_nums, trustee_prompt_num=0,
                                        trustee_format='trustee_ls', show_plot=True):
    """
    Generate overview plot showing mean agreement rates across all policies.

    Args:
        model (str): Model name
        policies_list (list): List of policy indices
        delegate_prompt_nums (list): List of delegate prompt numbers
        trustee_prompt_num (int): Trustee prompt number
        trustee_format (str): Trustee data format
        show_plot (bool): Whether to display the plot

    Returns:
        dict: Overview results with mean curves and data
    """
    weights = np.arange(0.0, 2.01, 0.1)

    print(f"Calculating overview agreement rates across {len(policies_list)} policies...")

    # Collect all trustee agreement curves
    all_trustee_curves = []
    all_delegate_curves = []
    successful_policies = []

    for policy_index in policies_list:
        try:
            print(f"Processing policy {policy_index + 1} for overview...", end=" ")

            # Calculate trustee agreement rates
            trustee_rates = calculate_trustee_agreement_rate(
                model, policy_index, weights, trustee_prompt_num, trustee_format
            )

            # Calculate delegate agreement rates
            delegate_results = calculate_delegate_agreement_rate(
                model, policy_index, weights, delegate_prompt_nums, trustee_format
            )

            # Check if we have valid data
            if not all(np.isnan(r) for r in trustee_rates) and 'mean' in delegate_results:
                delegate_mean_rates = delegate_results['mean']
                if not all(np.isnan(r) for r in delegate_mean_rates):
                    all_trustee_curves.append(np.array(trustee_rates))
                    all_delegate_curves.append(np.array(delegate_mean_rates))
                    successful_policies.append(policy_index)
                    print("✓")
                else:
                    print("(no delegate data)")
            else:
                print("(no trustee data)")

        except Exception as e:
            print(f"(error: {e})")

    if not all_trustee_curves or not all_delegate_curves:
        print("No valid data found for overview")
        return None

    # Calculate overall means
    trustee_overall_mean = np.nanmean(all_trustee_curves, axis=0)
    delegate_overall_mean = np.nanmean(all_delegate_curves, axis=0)

    # Create plot if requested
    if show_plot:
        plt.figure(figsize=(16, 12))

        # Plot individual trustee curves as light blue lines
        for i, curve in enumerate(all_trustee_curves):
            policy_idx = successful_policies[i]
            plt.plot(weights, curve, color='#add8e6', alpha=0.3, linewidth=0.8)

        # Plot individual delegate curves as light red lines
        for i, curve in enumerate(all_delegate_curves):
            policy_idx = successful_policies[i]
            plt.plot(weights, curve, color='#ffcccb', alpha=0.3, linewidth=0.8)

        # Plot overall means as thick lines
        plt.plot(weights, trustee_overall_mean, color='blue', linewidth=4,
               label='Mean Trustee Agreement Rate', alpha=0.9)
        plt.plot(weights, delegate_overall_mean, color='red', linewidth=4,
               label='Mean Delegate Agreement Rate', alpha=0.9)

        # Add legend entries for individual lines
        plt.plot([], [], color='#add8e6', alpha=0.3, linewidth=0.8,
               label='Individual Policy Trustee Rates')
        plt.plot([], [], color='#ffcccb', alpha=0.3, linewidth=0.8,
               label='Individual Policy Delegate Rates')

        # Format plot
        plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=14)
        plt.ylabel('Agreement Rate', fontsize=14)
        plt.title(f'Agreement Rates Overview - All Policies\\n{model}, Trustee Prompt {trustee_prompt_num}, {len(successful_policies)} policies',
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, 2)
        plt.legend(loc='upper right', fontsize=12)

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        if show_plot:
            plt.show()

    return {
        'weights': weights.tolist(),
        'trustee_overall_mean': trustee_overall_mean.tolist(),
        'delegate_overall_mean': delegate_overall_mean.tolist(),
        'all_trustee_curves': all_trustee_curves,
        'all_delegate_curves': all_delegate_curves,
        'successful_policies': successful_policies
    }


def generate_agreement_report(model, policies_list, delegate_prompt_nums, trustee_prompt_num=0,
                             trustee_format='trustee_ls', output_file=None, policies_per_page=4):
    """
    Generate a PDF report with agreement rate comparison plots for multiple policies.

    Args:
        model (str): Model name
        policies_list (list): List of policy indices to analyze (0-based)
        delegate_prompt_nums (list): List of delegate prompt numbers
        trustee_prompt_num (int): Trustee prompt number
        trustee_format (str): Trustee data format
        output_file (str): Output PDF filename (default: auto-generated)
        policies_per_page (int): Number of policies per page (default: 4 for 2x2 grid)

    Returns:
        str: Path to generated PDF file
    """
    # Generate default filename if not provided
    if output_file is None:
        output_file = f"../data/plots/{model}_agreement_rates_{trustee_prompt_num}_{trustee_format}.pdf"
    elif not output_file.startswith('../data/plots/') and not os.path.isabs(output_file):
        output_file = f"../data/plots/{output_file}"

    # Ensure plots directory exists
    os.makedirs("../data/plots", exist_ok=True)

    print(f"Generating Agreement Rates PDF report: {output_file}")
    print(f"Model: {model}")
    print(f"Policies: {policies_list}")
    print(f"Delegate prompts: {delegate_prompt_nums}")
    print(f"Trustee prompt: {trustee_prompt_num}")
    print(f"Trustee format: {trustee_format}")
    print(f"Layout: {policies_per_page} policies per page")
    print("=" * 80)

    # Calculate grid dimensions
    if policies_per_page == 4:
        rows, cols = 2, 2
    else:
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
            overview_results = plot_all_policies_agreement_overview(
                model, policies_list, delegate_prompt_nums, trustee_prompt_num, trustee_format, show_plot=False
            )

            if overview_results:
                weights = overview_results['weights']
                trustee_mean = overview_results['trustee_overall_mean']
                delegate_mean = overview_results['delegate_overall_mean']
                all_trustee_curves = overview_results['all_trustee_curves']
                all_delegate_curves = overview_results['all_delegate_curves']
                successful_policies = overview_results['successful_policies']

                # Plot individual trustee curves as light blue lines
                for i, curve in enumerate(all_trustee_curves):
                    plt.plot(weights, curve, color='#add8e6', alpha=0.25, linewidth=0.6)

                # Plot individual delegate curves as light red lines
                for i, curve in enumerate(all_delegate_curves):
                    plt.plot(weights, curve, color='#ffcccb', alpha=0.25, linewidth=0.6)

                # Plot overall means as thick lines
                plt.plot(weights, trustee_mean, color='blue', linewidth=4,
                       label='Mean Trustee Agreement Rate', alpha=0.9)
                plt.plot(weights, delegate_mean, color='red', linewidth=4,
                       label='Mean Delegate Agreement Rate', alpha=0.9)

                # Add legend entries for individual lines
                plt.plot([], [], color='#add8e6', alpha=0.25, linewidth=0.6,
                       label='Individual Policy Trustee Rates')
                plt.plot([], [], color='#ffcccb', alpha=0.25, linewidth=0.6,
                       label='Individual Policy Delegate Rates')

                # Format plot
                plt.xlabel('Weight Parameter (Long-term Weight / Sigma)', fontsize=14)
                plt.ylabel('Agreement Rate', fontsize=14)
                plt.title(f'Agreement Rates Overview - All Policies\\n{model}, Trustee Prompt {trustee_prompt_num}, {len(successful_policies)} policies',
                         fontsize=16, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1)
                plt.xlim(0, 2)
                plt.legend(loc='upper right', fontsize=12)

                # Format y-axis as percentages
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

                print(f"  ✓ Overview completed with {len(successful_policies)} policies")
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
            plt.text(0.5, 0.5, f'Error generating overview plot:\\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='red')
            plt.title('Overview Plot (Error)', fontsize=16, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)

        # Process policies in batches for 2x2 grids
        for page_start in range(0, len(policies_list), policies_per_page):
            page_policies = policies_list[page_start:page_start + policies_per_page]

            print(f"\\nGenerating page {page_start//policies_per_page + 2} with policies: {[p+1 for p in page_policies]}")

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

                    # Generate agreement rate data (suppress display)
                    results = plot_agreement_comparison(
                        model, policy_index, delegate_prompt_nums,
                        trustee_prompt_num, trustee_format, show_plot=False
                    )

                    weights = results['weights']
                    trustee_rates = results['trustee_agreement_rates']
                    delegate_results = results['delegate_agreement_results']

                    # Plot trustee agreement rate
                    plt.plot(weights, trustee_rates,
                           color='blue', linewidth=2.5, marker='o', markersize=3,
                           label=f'Trustee (Prompt {trustee_prompt_num})', alpha=0.8)

                    # Plot mean delegate agreement rate
                    if 'mean' in delegate_results:
                        plt.plot(weights, delegate_results['mean'],
                               color='red', linewidth=2.5, marker='D', markersize=3,
                               label='Mean Delegate', alpha=0.8)

                    # Plot individual delegate prompts (lighter)
                    colors = ['lightcoral', 'lightgreen', 'orange', 'plum']
                    for j, delegate_prompt_num in enumerate(delegate_prompt_nums):
                        if delegate_prompt_num in delegate_results:
                            rates = delegate_results[delegate_prompt_num]
                            color = colors[j % len(colors)]
                            plt.plot(weights, rates,
                                   color=color, linewidth=1, linestyle=':', marker='s', markersize=2,
                                   label=f'Delegate {delegate_prompt_num}', alpha=0.5)

                    # Get policy statement for title
                    try:
                        policies_df = pd.read_json("../self_selected_policies.jsonl", lines=True)
                        policy_statement = policies_df.iloc[policy_index]['statement']
                        if len(policy_statement) > 50:
                            policy_title = policy_statement[:47] + "..."
                        else:
                            policy_title = policy_statement
                    except:
                        policy_title = f'Policy {policy_index + 1}'

                    # Format subplot
                    plt.title(policy_title, fontsize=10, fontweight='bold', pad=8)
                    plt.xlabel('Weight Parameter', fontsize=9)
                    plt.ylabel('Agreement Rate', fontsize=9)
                    plt.grid(True, alpha=0.3)
                    plt.ylim(0, 1)
                    plt.xlim(0, 2)

                    # Format y-axis as percentages
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

                    # Add legend only to first subplot to avoid clutter
                    if i == 0:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

                    print(f"    ✓ Policy {policy_index + 1} completed")

                except Exception as e:
                    print(f"    ✗ Error processing policy {policy_index + 1}: {e}")

                    # Create error plot
                    plt.sca(axes[i])
                    plt.text(0.5, 0.5, f'Error: Policy {policy_index + 1}\\n{str(e)}',
                           ha='center', va='center', transform=axes[i].transAxes,
                           fontsize=10, color='red')
                    plt.title(f'Policy {policy_index + 1} (Error)', fontsize=10, color='red')
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])

            # Hide unused subplots
            for i in range(len(page_policies), len(axes)):
                axes[i].set_visible(False)

            # Add page title
            fig.suptitle(f'Agreement Rates: Trustees vs Delegates\\n{model}, Trustee Prompt {trustee_prompt_num}',
                        fontsize=16, fontweight='bold', y=0.95)

            # Adjust layout and save page
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)

            print(f"  ✓ Page {page_start//policies_per_page + 2} completed")

    print("=" * 80)
    print(f"✅ Agreement rates PDF report generated successfully: {output_file}")

    # Print file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"📄 File size: {file_size:.2f} MB")

    return output_file


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Generate PDF report of agreement rate patterns across policies')

    parser.add_argument('--model', type=str, default='claude-3-sonnet-v2',
                       help='Model name (claude-3-sonnet-v2 or gpt-4o) (default: gpt-4o)')
    parser.add_argument('--policies', type=str, default='0-19',
                       help='Policy range (e.g., "0-19" or "0,1,2,3") (default: 0-19)')
    parser.add_argument('--delegate-prompts', type=str, default='0,1,2,3',
                       help='Delegate prompt numbers (comma-separated) (default: 0,1,2,3)')
    parser.add_argument('--trustee-prompt', type=int, default=0,
                       help='Trustee prompt number (default: 0)')
    parser.add_argument('--trustee-format', type=str, default='trustee_lsd',
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
        output_file = generate_agreement_report(
            args.model, policies_list, delegate_prompt_nums,
            args.trustee_prompt, args.trustee_format, args.output, args.policies_per_page
        )

        print(f"\\n🎉 Agreement rates report generation completed successfully!")
        print(f"📁 Output file: {os.path.abspath(output_file)}")

    except KeyboardInterrupt:
        print(f"\\n⚠ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()