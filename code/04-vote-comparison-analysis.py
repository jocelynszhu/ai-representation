#%%
import pandas as pd
import numpy as np
from itertools import product
#%%
#pd.read_json("data/delegate/claude-3-sonnet/prompt-3/d_policy_1_votes.jsonl", encoding='cp1252', lines=True)
#%%
# Load policies
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
#%%
def compute_flip_percentage(trial1, trial2, role1, role2, policies_to_ignore=None):
    """Compute the percentage of flipped votes between two trials.
    
    Args:
        trial1: First trial path (e.g., 'gpt-4o/prompt-3')
        trial2: Second trial path
        role1: First role ('trustee' or 'delegate')
        role2: Second role ('trustee' or 'delegate')
    """
    flip_votes_list = []
   # print(f"Processing {role1}-{role2} Trial1: {trial1} Trial2: {trial2}...")
    print(f"First path: {role1}/{trial1}/{role1[0]}")
    print(f"Second path: {role2}/{trial2}/{role2[0]}")
    all_data = []
    for i in range(1, 21):
        if policies_to_ignore is not None and i in policies_to_ignore:
            print(f"Skipping policy {i} because it is in the ignore list")
            continue
       # print(f"Processing policy {i}...")
        # Load votes for both trials
        try:
            votes1 = pd.read_json(f"../data/{role1}/{trial1}/{role1[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        except:
            print(f"File not found: {role1}/{trial1}/{role1[0]}_policy_{i}_votes.jsonl")
            raise Exception(f"File not found: {role1}/{trial1}/{role1[0]}_policy_{i}_votes.jsonl")
          #  continue
        try:
            votes2 = pd.read_json(f"../data/{role2}/{trial2}/{role2[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        except:
            print(f"File not found: {role2}/{trial2}/{role2[0]}_policy_{i}_votes.jsonl")
            raise Exception(f"File not found: {role2}/{trial2}/{role2[0]}_policy_{i}_votes.jsonl")
        
        # Add source and index columns
        votes1['source'] = role1
        votes2['source'] = role2
        votes1['idx'] = range(len(votes1))
        votes2['idx'] = range(len(votes2))
        
        # Merge on index
        merged = pd.merge(votes1, votes2, how='inner', on='idx', suffixes=('_1', '_2'))\
            .assign(policy_id=i)
        # Remove rows with NA votes from both sides
        merged = merged[
            (~merged['vote_1'].isin(['NA', 'na', 'N/A', 'n/a'])) & 
            (~merged['vote_2'].isin(['NA', 'na', 'N/A', 'n/a']))
        ]
        # Print number of rows removed due to NA votes
        num_removed = len(votes1) - len(merged)
       # print(f"Removed {num_removed} rows with NA votes")
        # Find flipped votes
        flipped = merged[
            ((merged['vote_1'] == 'Yes') & (merged['vote_2'] == 'No')) |
            ((merged['vote_1'] == 'No') & (merged['vote_2'] == 'Yes'))
        ]
        merged['flipped'] = merged['vote_1'] != merged['vote_2']
        all_data.append(merged)
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
    all_data_df = pd.concat(all_data)
    return flip_percentage, flip_votes_list, all_data_df

# Define all combinations to analyze
trials = ['gpt-4o/prompt-1', 'gpt-4o/prompt-2', 'gpt-4o/prompt-3', 'gpt-4o/prompt-4']
#trials = ['claude-3-sonnet/prompt-1', 'claude-3-sonnet/prompt-2', 'claude-3-sonnet/prompt-3', 'claude-3-sonnet/prompt-4']
#trials = ['llama-3.2/prompt-3', 'llama-3.2/prompt-4']
results = []
#policies_to_ignore = [17,19, 20, 7]
policies_to_ignore = None
#%%
# 1. Delegate-Trustee combinations (cross-role)
all_data_across_trials = []
for t1, t2 in product(trials, trials):
    flip_percentage, _, all_data = compute_flip_percentage(t1, t2, 'delegate', 'trustee', policies_to_ignore)
    results.append({
        'type': 'cross-role',
        'trial1': t1,
        'trial2': t2,
        'role1': 'delegate',
        'role2': 'trustee',
        'flip_percentage': flip_percentage
        })
    all_data_across_trials.append(all_data.assign(condition="cross-role"))
#%%
# 2. Delegate-Delegate combinations (same role)
for t1, t2 in product(trials, trials):
    if t1 != t2:  # Only compare different prompts
        flip_percentage, _, all_data = compute_flip_percentage(t1, t2, 'delegate', 'delegate', policies_to_ignore)
        results.append({
            'type': 'same-role',
            'trial1': t1,
            'trial2': t2,
            'role1': 'delegate',
            'role2': 'delegate',
            'flip_percentage': flip_percentage
        })
        all_data_across_trials.append(all_data.assign(condition="same-role"))
#%%
# 3. Trustee-Trustee combinations (same role)
for t1, t2 in product(trials, trials):
    if t1 != t2:  # Only compare different prompts
        flip_percentage, _, all_data = compute_flip_percentage(t1, t2, 'trustee', 'trustee', policies_to_ignore)
        results.append({
            'type': 'same-role',
            'trial1': t1,
            'trial2': t2,
            'role1': 'trustee',
            'role2': 'trustee',
            'flip_percentage': flip_percentage
            })
        all_data_across_trials.append(all_data.assign(condition="same-role"))
#%%
# Convert to DataFrame
results_df = pd.DataFrame(results)
all_data_across_trials_df = pd.concat(all_data_across_trials)
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
# Calculate separate averages for delegate-delegate and trustee-trustee
delegate_avg = results_df[(results_df['type'] == 'same-role') & 
                         (results_df['role1'] == 'delegate')]['flip_percentage'].mean()
trustee_avg = results_df[(results_df['type'] == 'same-role') & 
                        (results_df['role1'] == 'trustee')]['flip_percentage'].mean()

print("\nDetailed Same-Role Averages:")
print(f"Delegate-Delegate average: {delegate_avg:.4f}")
print(f"Trustee-Trustee average: {trustee_avg:.4f}")

# Save results to CSV
results_df.to_csv("../data/vote_comparison_results.csv", index=False)
#%%
import statsmodels.formula.api as smf
# Convert flipped to binary

all_data_across_trials_df['flipped'] = [1 if x else 0 for x in all_data_across_trials_df['flipped']]
all_data_across_trials_df['policy_id'] = all_data_across_trials_df['policy_id'].astype(str)
model = smf.logit("flipped ~ C(condition) + C(policy_id)", data=all_data_across_trials_df).fit(cov_type='cluster',
                                                                                 cov_kwds={'groups': all_data_across_trials_df['id_1']})
print(model.summary())
#%%
from statsmodels.tools.sm_exceptions import PerfectSeparationError
try:
    model = smf.logit("flipped ~ C(condition) + C(policy_id) + C(policy_id) * C(condition)", data=all_data_across_trials_df).fit(cov_type='cluster', method='bfgs', maxiter=1000, disp=True,
                                                                                 cov_kwds={'groups': all_data_across_trials_df['id_1']})
except PerfectSeparationError:
    print("Perfect separation detected")
print(model.summary())
#%%
model_basic = smf.logit("flipped ~ C(condition)", data=all_data_across_trials_df).fit(cov_type='cluster',
                                                                                 cov_kwds={'groups': all_data_across_trials_df['id_1']})
print(model_basic.summary())
#%%
all_data_across_trials_df.groupby("condition").flipped.mean()
#%%
def analyze_vote_variance(trial):
    """Analyze variance in votes between delegate and trustee conditions for a given trial."""
    variance_results = []
    
    for i in range(1, 21):
        # Load votes for both roles
        votes_delegate = pd.read_json(f"../data/delegate/{trial}/d_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        votes_trustee = pd.read_json(f"../data/trustee/{trial}/t_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        
        # Convert votes to numeric (Yes=1, No=0)
        votes_delegate['vote_numeric'] = (votes_delegate['vote'] == 'Yes').astype(int)
        votes_trustee['vote_numeric'] = (votes_trustee['vote'] == 'Yes').astype(int)
        
        # Calculate variance
        delegate_variance = votes_delegate['vote_numeric'].var()
        trustee_variance = votes_trustee['vote_numeric'].var()
        
        # Calculate mean
        delegate_mean = votes_delegate['vote_numeric'].mean()
        trustee_mean = votes_trustee['vote_numeric'].mean()
        
        # Calculate counts
        delegate_yes = (votes_delegate['vote'] == 'Yes').sum()
        delegate_no = (votes_delegate['vote'] == 'No').sum()
        trustee_yes = (votes_trustee['vote'] == 'Yes').sum()
        trustee_no = (votes_trustee['vote'] == 'No').sum()
        
        variance_results.append({
            'policy_id': i,
            'policy': policies.iloc[i-1].statement,
            'delegate_variance': delegate_variance,
            'trustee_variance': trustee_variance,
            'delegate_mean': delegate_mean,
            'trustee_mean': trustee_mean,
            'delegate_yes': delegate_yes,
            'delegate_no': delegate_no,
            'trustee_yes': trustee_yes,
            'trustee_no': trustee_no
        })
    
    # Convert to DataFrame
    variance_df = pd.DataFrame(variance_results)
    
    # Calculate overall statistics
    overall_stats = {
        'trial': trial,
        'mean_delegate_variance': variance_df['delegate_variance'].mean(),
        'mean_trustee_variance': variance_df['trustee_variance'].mean(),
        'mean_delegate_yes_percent': (variance_df['delegate_yes'] / (variance_df['delegate_yes'] + variance_df['delegate_no'])).mean() * 100,
        'mean_trustee_yes_percent': (variance_df['trustee_yes'] / (variance_df['trustee_yes'] + variance_df['trustee_no'])).mean() * 100
    }
    
    return variance_df, overall_stats

# Analyze both trials
trials = ['gpt-4o/prompt-3', 'gpt-4o/prompt-4']
all_variance_results = []

for trial in trials:
    variance_df, overall_stats = analyze_vote_variance(trial)
    all_variance_results.append({
        'trial': trial,
        'variance_df': variance_df,
        'overall_stats': overall_stats
    })

# Print results
print("\nVote Variance Analysis:")
for result in all_variance_results:
    print(f"\nTrial: {result['trial']}")
    print("\nOverall Statistics:")
    print(f"Mean Delegate Variance: {result['overall_stats']['mean_delegate_variance']:.3f}")
    print(f"Mean Trustee Variance: {result['overall_stats']['mean_trustee_variance']:.3f}")
    print(f"Mean Delegate Yes %: {result['overall_stats']['mean_delegate_yes_percent']:.1f}%")
    print(f"Mean Trustee Yes %: {result['overall_stats']['mean_trustee_yes_percent']:.1f}%")
    
    print("\nPolicy-level Statistics:")
    print(result['variance_df'][['policy_id', 'delegate_variance', 'trustee_variance', 
                                'delegate_yes', 'delegate_no', 'trustee_yes', 'trustee_no']])

# Save detailed results to CSV
for result in all_variance_results:
    result['variance_df'].to_csv(f"../data/vote_variance_{result['trial'].replace('/', '_')}.csv", index=False)

# %%