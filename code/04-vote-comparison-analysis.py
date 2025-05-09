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

# %% Add demographic analysis
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# Load biographies
biographies = pd.read_json("biographies.jsonl", lines=True)
biographies['participant_idx'] = range(len(biographies))

def analyze_demographic_factors(trial1, trial2, role1, role2):
    """Analyze demographic factors predicting vote flips between two trials."""
    # Get all votes for both trials
    all_votes_list = []
    
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
        
        # Add to list
        for _, row in merged.iterrows():
            all_votes_list.append({
                'participant_idx': row['idx'],
                'policy_id': i,
                'policy': policies.iloc[i-1].statement,
                'vote_1': row['vote_1'],
                'reason_1': row['reason_1'],
                'vote_2': row['vote_2'],
                'reason_2': row['reason_2']
            })
    
    # Convert to DataFrame
    all_votes_df = pd.DataFrame(all_votes_list)
    
    # Merge with biographies
    votes_biography = all_votes_df.merge(biographies, on='participant_idx', how='left')
    
    # Prepare demographic variables
    demographic_cols = [
    'Gender',
    'Race/Ethnicity',
    'Income Level',
    'Education Level',
    'Occupation',
    'Marital Status',
    'Household Size',
    'Geographic Location',
    'Religion',
    'Language',
    'Housing Status',
    'Health Status',
    'Political Affiliation'
    ]
    
    # Ensure all demographic columns exist
    for col in demographic_cols:
        if col not in votes_biography.columns:
            print(f"Warning: Column {col} not found in data")
            return None
    
    try:
        # First, ensure all categorical variables are strings
        for col in demographic_cols:
            votes_biography[col] = votes_biography[col].astype(str)
        
        # Drop any rows with missing values
        votes_biography = votes_biography.dropna()
        
        # Create dummy variables
        X = pd.get_dummies(votes_biography[demographic_cols], drop_first=True)
        
        # Convert all columns to float
        X = X.astype(float)
        
        # Add constant term
        X = sm.add_constant(X)
        
        # Create binary outcome (1 if vote flipped, 0 if not)
        y = (votes_biography['vote_1'] != votes_biography['vote_2']).astype(int)
        
        # Check for any NaN values
        if X.isna().any().any() or y.isna().any():
            print("Warning: NaN values found in data")
            return None
        
        # Print data types for debugging
        print(f"X dtypes:\n{X.dtypes}")
        print(f"y dtype: {y.dtype}")
        print(f"Number of flipped votes: {y.sum()} out of {len(y)} total votes")
        
        # Fit logistic regression
        model = sm.Logit(y, X)
        results = model.fit(disp=0)  # Suppress convergence messages
        
        # Extract coefficients and p-values
        coef_df = pd.DataFrame({
            'Variable': results.model.exog_names,
            'Coefficient': results.params,
            'P-value': results.pvalues
        })
        
        # Add pairing information
        coef_df['trial1'] = trial1
        coef_df['trial2'] = trial2
        coef_df['role1'] = role1
        coef_df['role2'] = role2
        
        return coef_df
        
    except Exception as e:
        print(f"Error in logistic regression for {role1}-{role2} {trial1}-{trial2}: {str(e)}")
        print("X shape:", X.shape if 'X' in locals() else "X not created")
        print("y shape:", y.shape if 'y' in locals() else "y not created")
        return None

# Collect results for all pairings
all_coefs = []

# 1. Delegate-Trustee combinations
for t1, t2 in product(trials, trials):
    coefs = analyze_demographic_factors(t1, t2, 'delegate', 'trustee')
    if coefs is not None:
        all_coefs.append(coefs)

# 2. Delegate-Delegate combinations
for t1, t2 in product(trials, trials):
    if t1 != t2:
        coefs = analyze_demographic_factors(t1, t2, 'delegate', 'delegate')
        if coefs is not None:
            all_coefs.append(coefs)

# 3. Trustee-Trustee combinations
for t1, t2 in product(trials, trials):
    if t1 != t2:
        coefs = analyze_demographic_factors(t1, t2, 'trustee', 'trustee')
        if coefs is not None:
            all_coefs.append(coefs)

# Combine all results
if all_coefs:
    final_results = pd.concat(all_coefs, ignore_index=True)
    
    # Format the results for better readability
    final_results['Significance'] = final_results['P-value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
    )
    
    # Round coefficients and p-values
    final_results['Coefficient'] = final_results['Coefficient'].round(3)
    final_results['P-value'] = final_results['P-value'].round(3)
    
    # Reorder columns
    final_results = final_results[[
        'trial1', 'trial2', 'role1', 'role2', 'Variable', 
        'Coefficient', 'P-value', 'Significance'
    ]]
    
    print("\nDemographic Analysis Results:")
    print(final_results)
    
    # Save to CSV
    final_results.to_csv("../data/demographic_analysis_results.csv", index=False)
else:
    print("No valid results found for any pairing")
# %%
all_coefs