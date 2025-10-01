#!/usr/bin/env python3
"""
Create a table of agreement rates for all policies across models and conditions.

Rows: Policies (30 total)
Columns: MultiIndex with Model (level 0) and Condition (Delegate/Trustee, level 1)
Values: Agreement rate with model default or expert consensus
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from agreement_demographics import create_agreement_dataframe

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

def create_agreement_table(
    delegate_prompt_nums: List[int] = [0, 1, 2, 3, 4],
    trustee_prompt_nums: List[int] = [0, 1, 2],
    alpha: float = 1.0,
    trustee_type: str = "both"
) -> pd.DataFrame:
    """
    Create agreement table for all policies.

    Args:
        delegate_prompt_nums: List of delegate prompt numbers
        trustee_prompt_nums: List of trustee prompt numbers
        alpha: Alpha value for trustee calculations
        trustee_type: "both" to average LS and LSD, or specific type

    Returns:
        DataFrame with MultiIndex columns (Model, Condition)
    """
    # Load policies
    policies_df = pd.read_json("../self_selected_policies_new.jsonl", lines=True)
    num_policies = len(policies_df)

    print(f"Creating agreement table for {num_policies} policies...")
    print(f"Models: {len(MODELS)}")
    print(f"Delegate prompts: {delegate_prompt_nums}")
    print(f"Trustee prompts: {trustee_prompt_nums}")
    print(f"Alpha: {alpha}, Trustee type: {trustee_type}")
    print("=" * 70)

    # Build data structure
    # Structure: {policy_idx: {model: {"Delegate": rate, "Trustee": rate}}}
    data = {}

    for policy_idx in range(num_policies):
        policy_data = policies_df.iloc[policy_idx]
        policy_statement = policy_data['statement']
        has_expert_vote = pd.notna(policy_data.get('expert_vote'))

        print(f"\nPolicy {policy_idx + 1}/{num_policies}: {policy_statement[:60]}...")
        print(f"  Has expert consensus: {has_expert_vote}")

        data[policy_idx] = {}

        for model in MODELS:
            print(f"  Processing model: {model}")

            # Get agreement dataframe
            df = create_agreement_dataframe(
                policy_index=policy_idx,
                delegate_prompt_nums=delegate_prompt_nums,
                trustee_prompt_nums=trustee_prompt_nums,
                model=model,
                alpha=alpha,
                trustee_type=trustee_type,
                compare_expert=has_expert_vote,
                bio_df=None,  # No demographic split
                demographic=None
            )

            # Extract agreement rates
            delegate_rate = df['delegate_mean_agreement'].iloc[0] if 'delegate_mean_agreement' in df.columns else np.nan
            trustee_rate = df['trustee_mean_agreement'].iloc[0] if 'trustee_mean_agreement' in df.columns else np.nan

            data[policy_idx][model] = {
                "Delegate": delegate_rate,
                "Trustee": trustee_rate
            }

    print("\n" + "=" * 70)
    print("Building DataFrame...")

    # Convert to DataFrame with MultiIndex columns
    # Create MultiIndex columns
    columns = pd.MultiIndex.from_product(
        [[MODEL_DISPLAY_NAMES[m] for m in MODELS], ["Delegate", "Trustee"]],
        names=["Model", "Condition"]
    )

    # Build rows
    rows = []
    policy_statements = []

    for policy_idx in range(num_policies):
        policy_statement = policies_df.iloc[policy_idx]['statement']
        policy_statements.append(policy_statement)

        row = []
        for model in MODELS:
            row.append(data[policy_idx][model]["Delegate"])
            row.append(data[policy_idx][model]["Trustee"])

        rows.append(row)

    # Create DataFrame
    result_df = pd.DataFrame(rows, columns=columns, index=policy_statements)
    result_df.index.name = "Policy"

    # Round all values to 2 decimal places
    result_df = result_df.round(2)

    return result_df


if __name__ == "__main__":
    # Create the table
    agreement_table = create_agreement_table(
        delegate_prompt_nums=[0, 1, 2, 3, 4],
        trustee_prompt_nums=[0, 1, 2],
        alpha=1.0,
        trustee_type="both"
    )

    # Save to Excel
    output_file = "../data/agreement_table.xlsx"
    agreement_table.to_excel(output_file)
    print(f"\n✓ Agreement table saved to: {output_file}")

    # Display summary
    print("\nTable shape:", agreement_table.shape)
    print("\nFirst few rows:")
    print(agreement_table.head())
