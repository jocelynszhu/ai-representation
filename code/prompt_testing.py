# %%
# Prompt Testing Script
# Test prompts from `prompts_long_short.json` on specific policies and users.

# %%
import json
import pandas as pd
import os
import anthropic
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# %%
# Configuration - modify these to test different combinations
policy_index = 0  # Which policy to test (0-based index)
user_index = 0    # Which user profile to test (0-based index)
prompt_type = "trustee_ls"  # "delegate" or "trustee"

# %%
# Load Data

# Load prompts
with open('../prompts_long_short_discount.json', 'r') as f:
    prompts = json.load(f)

# Load user profiles
written_profiles = pd.read_json("gpt-4o_written_profiles.jsonl", encoding='cp1252', lines=True)

# Load policies
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

print(f"Loaded {len(prompts)} prompt sets")
print(f"Loaded {len(written_profiles)} user profiles")
print(f"Loaded {len(policies)} policies")

# %%
# Preview Selected Data

# Show selected policy
selected_policy = policies.iloc[policy_index].statement
print(f"Selected Policy (index {policy_index}):")
print(selected_policy)
print()

# Show selected user profile (truncated)
selected_user = written_profiles.iloc[user_index]
print(f"Selected User (ID {selected_user['ID']}):")
print(selected_user['Profile'][:200] + "...")
print()

# Show selected prompt
selected_prompt = prompts["0"][prompt_type]
print(f"Selected Prompt ({prompt_type}):")
print(selected_prompt)

# %%
# Claude API Function
def run_claude(prompt, profile, policy):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [
        {
            "role": "user",
            "content": f"{prompt.format(bio=profile)}\n\nPolicy proposal: {policy}"
        }
    ]

    system = prompt.format(bio=profile)

    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        messages=messages,
        system=system,
        temperature=0.0
    )

    return response.content[0].text

# %%
# Run Test

# Run the test
response = run_claude(selected_prompt, selected_user['Profile'], selected_policy)

print("Claude Response:")
print("=" * 50)
print(response)
print("=" * 50)

# %%
# Parse JSON Response (if applicable)

# Try to parse JSON response
try:
    # Extract JSON from response
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    json_str = response[json_start:json_end]

    parsed_response = json.loads(json_str)

    print("Parsed JSON Response:")
    print(json.dumps(parsed_response, indent=2))

except (json.JSONDecodeError, ValueError) as e:
    print(f"Could not parse JSON: {e}")
    print("Raw response shown above.")

# %%
# Utility Weighting Function
def calculate_weighted_vote(parsed_response, short_term_weight):
    """
    Calculate weighted utility scores and determine vote.

    Args:
        parsed_response (dict): Parsed JSON response with yes_vote/no_vote utilities
        short_term_weight (float): Weight for short-term utility (0-1), long-term gets (1-short_term_weight)

    Returns:
        dict: Contains weighted scores and final vote decision
    """
    if short_term_weight < 0 or short_term_weight > 1:
        raise ValueError("short_term_weight must be between 0 and 1")

    long_term_weight = 1 - short_term_weight

    # Calculate weighted utility for YES vote
    yes_weighted = (parsed_response["yes_vote"]["short_util"] * short_term_weight +
                   parsed_response["yes_vote"]["long_util"] * long_term_weight)

    # Calculate weighted utility for NO vote
    no_weighted = (parsed_response["no_vote"]["short_util"] * short_term_weight +
                  parsed_response["no_vote"]["long_util"] * long_term_weight)

    # Determine vote
    if yes_weighted > no_weighted:
        vote = "Yes"
    elif no_weighted > yes_weighted:
        vote = "No"
    else:
        vote = "Neutral"

    return {
        "yes_weighted_utility": yes_weighted,
        "no_weighted_utility": no_weighted,
        "vote": vote,
        "short_term_weight": short_term_weight,
        "long_term_weight": long_term_weight
    }

# %%
# Batch Prediction Function
def predict_all_users_for_policy(policy_index, prompt_type="trustee_ls", model_name="claude-3-7-sonnet", prompt_num=0, n_users=None):
    """
    Get predictions for all users on a given policy and save to structured files.

    Args:
        policy_index (int): Index of policy to test (0-based)
        prompt_type (str): "delegate_ls" or "trustee_ls"
        model_name (str): Model name for directory structure
        prompt_num (int): Prompt number for directory structure
        n_users (int, optional): Number of users to process (defaults to all users)
    """
    import os

    # Get policy and prompt
    policy = policies.iloc[policy_index].statement
    prompt = prompts[str(prompt_num)][prompt_type]

    # Create output directory
    output_dir = f"../data/{prompt_type}/{model_name}/prompt-{prompt_num}"
    os.makedirs(output_dir, exist_ok=True)

    # Output file path
    prefix = "t_" if "trustee" in prompt_type else "d_"
    output_file = f"{output_dir}/{prefix}policy_{policy_index + 1}_votes.jsonl"

    # Determine how many users to process
    users_to_process = written_profiles.head(n_users) if n_users else written_profiles

    print(f"Processing policy {policy_index + 1}: {policy}")
    print(f"Saving to: {output_file}")
    print(f"Processing {len(users_to_process)} users...")

    # Open file for immediate writing
    results = []
    with open(output_file, 'w') as f:
        # Process each user
        for idx, row in users_to_process.iterrows():
            user_id = row['ID']
            profile = row['Profile']

            try:
                # Get LLM response
                response = run_claude(prompt, profile, policy)

                # Process response (similar to original script)
                clean_response = response.replace("\n", " ")
                try:
                    # Try to extract JSON
                    json_start = clean_response.find('{')
                    json_end = clean_response.rfind('}') + 1
                    json_str = clean_response[json_start:json_end]

                    # Parse and add ID
                    parsed = json.loads(json_str)
                    parsed['id'] = user_id

                    # Write immediately to file
                    f.write(json.dumps(parsed) + "\n")
                    f.flush()  # Ensure it's written to disk

                    results.append(parsed)

                except (json.JSONDecodeError, ValueError):
                    # If JSON parsing fails, create basic structure
                    result = {
                        'id': user_id,
                        'response': clean_response
                    }

                    # Write immediately to file
                    f.write(json.dumps(result) + "\n")
                    f.flush()  # Ensure it's written to disk

                    results.append(result)

                if len(results) % 10 == 0:
                    print(f"Processed {len(results)}/{len(users_to_process)} users")

            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue

    print(f"Completed! Saved {len(results)} responses to {output_file}")
    return results

# %%
# Delegate-Trustee Comparison Function
def create_delegate_trustee_comparison(model, policy_index, short_term_weight, prompt_num=0):
    """
    Create a comparison DataFrame of delegate vs trustee votes for a given policy.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        short_term_weight (float): Weight for short-term utility (0-1)
        prompt_num (int): Prompt number (default 0)

    Returns:
        pd.DataFrame: Columns: participant_id, policy_id, delegate_vote, trustee_vote
    """
    import os

    policy_id = policy_index + 1

    # File paths
    delegate_file = f"../data/delegate/{model}/prompt-{prompt_num}/d_policy_{policy_id}_votes.jsonl"
    trustee_file = f"../data/trustee_ls/{model}/prompt-{prompt_num}/t_policy_{policy_id}_votes.jsonl"

    # Check if files exist
    if not os.path.exists(delegate_file):
        raise FileNotFoundError(f"Delegate file not found: {delegate_file}")
    if not os.path.exists(trustee_file):
        raise FileNotFoundError(f"Trustee file not found: {trustee_file}")

    # Load delegate data
    delegate_data = []
    with open(delegate_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                delegate_data.append({
                    'participant_id': data['id'],
                    'delegate_vote': data['vote']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing delegate data: {e}")
                continue

    delegate_df = pd.DataFrame(delegate_data)

    # Load and process trustee data
    trustee_data = []
    with open(trustee_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())

                # Use calculate_weighted_vote to get the final vote
                vote_result = calculate_weighted_vote(data, short_term_weight)

                trustee_data.append({
                    'participant_id': data['id'],
                    'trustee_vote': vote_result['vote']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing trustee data: {e}")
                continue
            except Exception as e:
                print(f"Error calculating weighted vote for participant {data.get('id', 'unknown')}: {e}")
                continue

    trustee_df = pd.DataFrame(trustee_data)

    # Merge the dataframes
    comparison_df = pd.merge(delegate_df, trustee_df, on='participant_id', how='outer')

    # Add policy_id column
    comparison_df['policy_id'] = policy_id

    # Reorder columns
    comparison_df = comparison_df[['participant_id', 'policy_id', 'delegate_vote', 'trustee_vote']]

    # Sort by participant_id
    comparison_df = comparison_df.sort_values('participant_id').reset_index(drop=True)

    print(f"Loaded {len(delegate_df)} delegate votes and {len(trustee_df)} trustee votes")
    print(f"Final comparison DataFrame: {len(comparison_df)} participants")

    return comparison_df

# %%
#predict_all_users_for_policy(policy_index, prompt_type, "claude-3-sonnet-v2", 0, n_users=10)
# %%
# Disagreement Rate Analysis Function
def plot_disagreement_by_weight(model, policy_index, prompt_num=0, show_plot=True):
    """
    Analyze disagreement rates between delegate and trustee votes across different short-term weights.

    Args:
        model (str): Model name (e.g., "claude-3-sonnet-v2")
        policy_index (int): 0-based policy index
        prompt_num (int): Prompt number (default 0)
        show_plot (bool): Whether to display the plot (default True)

    Returns:
        pd.DataFrame: Contains weights and corresponding disagreement rates
    """
    # Generate weight range from 0.0 to 1.0 in steps of 0.1
    weights = np.arange(0.0, 1.1, 0.1)
    disagreement_rates = []

    print(f"Analyzing disagreement rates for {len(weights)} different short-term weights...")

    for weight in weights:
        try:
            # Get comparison data for this weight
            comparison_df = create_delegate_trustee_comparison(model, policy_index, weight, prompt_num)

            # Calculate disagreement rate
            disagreements = comparison_df['delegate_vote'] != comparison_df['trustee_vote']
            disagreement_rate = disagreements.sum() / len(comparison_df)
            disagreement_rates.append(disagreement_rate)

            print(f"Weight {weight:.1f}: {disagreement_rate:.3f} disagreement rate ({disagreements.sum()}/{len(comparison_df)} participants)")

        except Exception as e:
            print(f"Error at weight {weight:.1f}: {e}")
            disagreement_rates.append(np.nan)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'short_term_weight': weights,
        'disagreement_rate': disagreement_rates
    })

    # Create plot
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(weights, disagreement_rates, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Short-term Weight', fontsize=12)
        plt.ylabel('Disagreement Rate', fontsize=12)
        plt.title(f'Delegate vs Trustee Disagreement Rate by Short-term Weight\\n{model}, Policy {policy_index + 1}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Add annotations for key points
        min_idx = np.nanargmin(disagreement_rates)
        max_idx = np.nanargmax(disagreement_rates)

        plt.annotate(f'Min: {disagreement_rates[min_idx]:.1%}',
                    xy=(weights[min_idx], disagreement_rates[min_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.annotate(f'Max: {disagreement_rates[max_idx]:.1%}',
                    xy=(weights[max_idx], disagreement_rates[max_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.show()

    return results_df

# %%
comparison_df = create_delegate_trustee_comparison("claude-3-sonnet-v2", policy_index, 0, 0)
disagreements_df = comparison_df[comparison_df['delegate_vote'] != comparison_df['trustee_vote']]
len(disagreements_df)/len(comparison_df)
# %%
comparison_df
# %%
