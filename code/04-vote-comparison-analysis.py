#%%
import pandas as pd
import numpy as np
from itertools import product
#%%
# Load policies
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)

def compute_flip_percentage(trial1, trial2, role1, role2):
    """Compute the percentage of flipped votes between two trials.
    
    Args:
        trial1: First trial path (e.g., 'gpt-4o/prompt-3')
        trial2: Second trial path
        role1: First role ('trustee' or 'delegate')
        role2: Second role ('trustee' or 'delegate')
    """
    flip_votes_list = []
    
    for i in range(1, 21):
        # Load votes for both trials
        votes1 = pd.read_json(f"../data/{role1}/{trial1}/{role1[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        votes2 = pd.read_json(f"../data/{role2}/{trial2}/{role2[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        
        # Add source and index columns
        votes1['source'] = role1
        votes2['source'] = role2
        votes1['idx'] = range(len(votes1))
        votes2['idx'] = range(len(votes2))
        
        # Merge on index
        merged = pd.merge(votes1, votes2, how='inner', on='idx', suffixes=('_1', '_2'))
        
        # Find flipped votes
        flipped = merged[
            ((merged['vote_1'] == 'Yes') & (merged['vote_2'] == 'No')) |
            ((merged['vote_1'] == 'No') & (merged['vote_2'] == 'Yes'))
        ]
        
        # Add to list
        for _, row in flipped.iterrows():
            flip_votes_list.append({
                'participant_idx': row['idx'],
                'policy_id': i,
                'policy': policies.iloc[i-1].statement,
                'vote_1': row['vote_1'],
                'reason_1': row['reason_1'],
                'vote_2': row['vote_2'],
                'reason_2': row['reason_2']
            })
    
    # Calculate percentage of flipped votes
    total_votes = 20 * len(merged)  # 20 policies * number of participants
    flip_percentage = len(flip_votes_list) / total_votes if total_votes > 0 else 0
    
    return flip_percentage, flip_votes_list

# Define all combinations to analyze
trials = ['gpt-4o/prompt-3', 'gpt-4o/prompt-4']
results = []

# 1. Delegate-Trustee combinations (cross-role)
for t1, t2 in product(trials, trials):
    flip_percentage, _ = compute_flip_percentage(t1, t2, 'delegate', 'trustee')
    results.append({
        'type': 'cross-role',
        'trial1': t1,
        'trial2': t2,
        'role1': 'delegate',
        'role2': 'trustee',
        'flip_percentage': flip_percentage
    })

# 2. Delegate-Delegate combinations (same role)
for t1, t2 in product(trials, trials):
    if t1 != t2:  # Only compare different prompts
        flip_percentage, _ = compute_flip_percentage(t1, t2, 'delegate', 'delegate')
        results.append({
            'type': 'same-role',
            'trial1': t1,
            'trial2': t2,
            'role1': 'delegate',
            'role2': 'delegate',
            'flip_percentage': flip_percentage
        })

# 3. Trustee-Trustee combinations (same role)
for t1, t2 in product(trials, trials):
    if t1 != t2:  # Only compare different prompts
        flip_percentage, _ = compute_flip_percentage(t1, t2, 'trustee', 'trustee')
        results.append({
            'type': 'same-role',
            'trial1': t1,
            'trial2': t2,
            'role1': 'trustee',
            'role2': 'trustee',
            'flip_percentage': flip_percentage
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Calculate averages for different categories
cross_role_avg = results_df[results_df['type'] == 'cross-role']['flip_percentage'].mean()
same_role_avg = results_df[results_df['type'] == 'same-role']['flip_percentage'].mean()

# Print detailed results
print("\nAll Combinations:")
print(results_df)
print("\nCross-Role Combinations (Delegate-Trustee):")
print(results_df[results_df['type'] == 'cross-role'])
print("\nSame-Role Combinations (Delegate-Delegate and Trustee-Trustee):")
print(results_df[results_df['type'] == 'same-role'])

print("\nAverages:")
print(f"Cross-role average (Delegate-Trustee): {cross_role_avg:.4f}")
print(f"Same-role average (Delegate-Delegate and Trustee-Trustee): {same_role_avg:.4f}")
print(f"Difference (Cross-role - Same-role): {cross_role_avg - same_role_avg:.4f}")

# Save results to CSV
results_df.to_csv("../data/vote_comparison_results.csv", index=False) 
# %%
