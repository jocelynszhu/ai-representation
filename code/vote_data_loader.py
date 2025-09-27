#%%
import json
import pandas as pd
import os
from typing import Dict, Tuple, Optional, Union
#%%

def load_policy_votes(
    model: str,
    trustee_type: str,
    policy_index: int,
    prompt_num: int = 0
) -> Dict[str, pd.DataFrame]:
    """
    Load and combine voting data for a specific policy across different vote types.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2", "gpt-4o")
        trustee_type (str): Either "trustee_ls" or "trustee_lsd"
        policy_index (int): 0-based policy index (will be converted to 1-based for file naming)
        prompt_num (int): Prompt number (default 0)

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'trustee', 'delegate', and 'defaults' DataFrames

    Raises:
        FileNotFoundError: If required data files are not found
        ValueError: If trustee_type is not valid or data format is unexpected
    """
    # Validate trustee_type
    if trustee_type not in ["trustee_ls", "trustee_lsd"]:
        raise ValueError(f"trustee_type must be 'trustee_ls' or 'trustee_lsd', got: {trustee_type}")

    # Convert 0-based index to 1-based for file naming
    policy_id = policy_index + 1

    # Construct file paths
    base_path = "../data"

    # Handle different directory structures for trustee data
    if trustee_type == "trustee_lsd":
        # trustee_lsd uses a different structure
        trustee_file = f"{base_path}/{trustee_type}/{model}/prompt-{prompt_num}/t_policy_{policy_id}_votes.jsonl"
    else:
        # trustee_ls uses self_selected_policies_new structure
        trustee_file = f"{base_path}/{trustee_type}/{model}/self_selected_policies_new/prompt-{prompt_num}/t_policy_{policy_id}_votes.jsonl"

    delegate_file = f"{base_path}/delegate/{model}/self_selected_policies_new/prompt-{prompt_num}/d_policy_{policy_id}_votes.jsonl"
    defaults_file = f"{base_path}/defaults/self_selected_policies_new/{model}.jsonl"

    # Load and process each dataset
    result = {}

    # Load trustee data
    result['trustee'] = _load_trustee_data(trustee_file, trustee_type)

    # Load delegate data
    result['delegate'] = _load_delegate_data(delegate_file)

    # Load defaults data
    result['defaults'] = _load_defaults_data(defaults_file, policy_index)

    return result


def _load_trustee_data(file_path: str, trustee_type: str) -> pd.DataFrame:
    """Load and process trustee voting data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trustee file not found: {file_path}")

    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                processed_entry = _process_trustee_entry(entry, trustee_type)
                data.append(processed_entry)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing trustee data at line {line_num}: {e}")
                continue

    return pd.DataFrame(data)


def _load_delegate_data(file_path: str) -> pd.DataFrame:
    """Load and process delegate voting data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Delegate file not found: {file_path}")

    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                processed_entry = {
                    'participant_id': entry['id'],
                    'vote': entry['vote'],
                    'reason': entry['reason']
                }
                data.append(processed_entry)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing delegate data at line {line_num}: {e}")
                continue

    return pd.DataFrame(data)


def _load_defaults_data(file_path: str, policy_index: int) -> pd.DataFrame:
    """Load and process default voting data for a specific policy."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Defaults file not found: {file_path}")

    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                # Filter for the specific policy (defaults file contains all policies)
                if entry.get('id') == policy_index + 1:  # Convert to 1-based for comparison
                    processed_entry = {
                        'participant_id': entry['id'],
                        'vote': entry['vote'],
                        'reason': entry['reason']
                    }
                    data.append(processed_entry)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing defaults data at line {line_num}: {e}")
                continue

    return pd.DataFrame(data)


def _process_trustee_entry(entry: Dict, trustee_type: str) -> Dict:
    """Process a single trustee entry based on the trustee type."""
    if trustee_type == "trustee_ls":
        return _process_trustee_ls_entry(entry)
    elif trustee_type == "trustee_lsd":
        return _process_trustee_lsd_entry(entry)
    else:
        raise ValueError(f"Unknown trustee_type: {trustee_type}")


def _process_trustee_ls_entry(entry: Dict) -> Dict:
    """Process trustee_ls format entry."""
    return {
        'participant_id': entry['id'],
        'yes_short_util': entry['yes_vote']['short_util'],
        'yes_long_util': entry['yes_vote']['long_util'],
        'yes_reasoning': entry['yes_vote']['reasoning'],
        'no_short_util': entry['no_vote']['short_util'],
        'no_long_util': entry['no_vote']['long_util'],
        'no_reasoning': entry['no_vote']['reasoning']
    }


def _process_trustee_lsd_entry(entry: Dict) -> Dict:
    """Process trustee_lsd format entry."""
    processed = {'participant_id': entry['id']}

    # Extract time period scores for yes vote
    for period in ["0-5 years", "5-10 years", "10-15 years", "15-20 years", "20-25 years", "25-30 years"]:
        period_key = period.replace("-", "_").replace(" ", "_")
        processed[f'yes_{period_key}_score'] = entry['yes'][period]['score']
        processed[f'yes_{period_key}_rationale'] = entry['yes'][period]['rationale']
        processed[f'no_{period_key}_score'] = entry['no'][period]['score']
        processed[f'no_{period_key}_rationale'] = entry['no'][period]['rationale']

    return processed


# Utility function to get summary statistics
def get_vote_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Get summary statistics for loaded voting data."""
    summary = {}

    for vote_type, df in data.items():
        if len(df) > 0:
            summary[vote_type] = {
                'total_participants': len(df),
                'unique_participants': df['participant_id'].nunique() if 'participant_id' in df.columns else 'N/A'
            }

            # Add vote distribution if vote column exists
            if 'vote' in df.columns:
                vote_counts = df['vote'].value_counts().to_dict()
                summary[vote_type]['vote_distribution'] = vote_counts
        else:
            summary[vote_type] = {'total_participants': 0}

    return summary
#%%

#%%
# data_ls = load_policy_votes(
#     model="claude-3-sonnet-v2",
#     trustee_type="trustee_ls",
#     policy_index=0,
#     prompt_num=0
# )
# #%%
# data_ls['defaults']
#%%
# # Example usage
# if __name__ == "__main__":
#     # Load data for policy index 0 with trustee_ls format
#     print("Loading policy data...")

#     try:
#         # Example 1: Trustee LS format
#         data_ls = load_policy_votes(
#             model="claude-3-sonnet-v2",
#             trustee_type="trustee_ls",
#             policy_index=0,
#             prompt_num=0
#         )

#         print("Trustee LS data loaded successfully:")
#         summary = get_vote_summary(data_ls)
#         for vote_type, stats in summary.items():
#             print(f"  {vote_type}: {stats}")

#         print(f"\nTrustee data columns: {list(data_ls['trustee'].columns)}")

#         # Example 2: Trustee LSD format
#         data_lsd = load_policy_votes(
#             model="claude-3-sonnet-v2",
#             trustee_type="trustee_lsd",
#             policy_index=4,  # Using policy index 4 as policy_5 exists
#             prompt_num=0
#         )

#         print("\nTrustee LSD data loaded successfully:")
#         summary = get_vote_summary(data_lsd)
#         for vote_type, stats in summary.items():
#             print(f"  {vote_type}: {stats}")

#     except Exception as e:
#         print(f"Error: {e}")