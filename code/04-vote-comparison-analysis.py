#%%
import pandas as pd
import numpy as np
from itertools import product
#%%
#pd.read_json("data/delegate/llama-3.2/prompt-3/d_policy_1_votes.jsonl", encoding='cp1252', lines=True)
#%%
# Load policies
policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
#%%
def compute_flip_percentage(trial1, trial2, role1, role2):
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
    for i in range(1, 21):
        print(f"Processing policy {i}...")
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
#trials = ['gpt-4o/prompt-0', 'gpt-4o/prompt-1', 'gpt-4o/prompt-2', 'gpt-4o/prompt-3', 'gpt-4o/prompt-4']
trials = ['llama-3.2/prompt-3', 'llama-3.2/prompt-4']
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
#%%
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
#%%
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
#%%
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
    print(f"Analyzing {role1}-{role2} {trial1}-{trial2}...")
    
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
    # 'Occupation',
    # 'Marital Status',
    # 'Household Size',
    # 'Geographic Location',
    # 'Religion',
    # 'Language',
    # 'Housing Status',
    # 'Health Status',
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

# %% Visualize coefficients
import matplotlib.pyplot as plt
import seaborn as sns

def plot_demographic_coefficients(category, results_df=None, save_path="../data/"):
    """
    Plot coefficients for a specific demographic category.
    
    Args:
        category (str): The demographic category to plot (e.g., 'Gender', 'Political Affiliation')
        results_df (pd.DataFrame, optional): The results DataFrame. If None, will load from CSV.
        save_path (str): Path to save the plot
    """
    # Load results if not provided
    if results_df is None:
        results_df = pd.read_csv("../data/demographic_analysis_results.csv")
    
    # Filter results for the category
    category_results = results_df[results_df['Variable'].str.contains(category, case=False)]
    
    if len(category_results) == 0:
        print(f"No results found for category: {category}")
        return
    
    # Calculate figure size based on number of variables
    n_vars = len(category_results['Variable'].unique())
    fig_width = min(15, max(8, n_vars * 0.8))  # Width between 8 and 15 inches
    fig_height = 8  # Fixed height
    
    # Create a figure with appropriate size
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create a bar plot with improved styling
    ax = sns.barplot(data=category_results, 
                    x='Variable', 
                    y='Coefficient',
                    hue='role1',
                    palette='Set2')
    
    # Add significance stars with improved positioning
    for i, row in category_results.iterrows():
        if row['Significance']:
            # Position the star above the bar
            y_pos = row['Coefficient'] + (0.1 if row['Coefficient'] >= 0 else -0.1)
            plt.text(i, y_pos, row['Significance'], 
                    ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    plt.title(f'Coefficients for {category} Across Different Runs', fontsize=12, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel('Coefficient', fontsize=10)
    plt.xlabel('Variable', fontsize=10)
    
    # Improve legend
    plt.legend(title='Role', title_fontsize=10, fontsize=9)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Save the plot with reasonable DPI
    filename = f'coefficients_{category.lower().replace("/", "_")}.png'
    plt.savefig(save_path + filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics for this category
    print(f"\nSummary for {category}:")
    summary = category_results.groupby('Variable').agg({
        'Coefficient': ['mean', 'std', 'count'],
        'P-value': ['mean', 'min']
    }).round(3)
    print(summary)
    
    return category_results

# Example usage:
#%%
# plot_demographic_coefficients('Gender')
plot_demographic_coefficients('Political Affiliation')
# plot_demographic_coefficients('Race/Ethnicity')

# %%
category = 'Political Affiliation'
results_df = pd.read_csv("../data/demographic_analysis_results.csv")
category_results = results_df[results_df['Variable'].str.contains(category, case=False)]

# %%
category_results_filt = category_results[category_results['Variable'] == "Political Affiliation_Republican"]
# %%
category_results_filt
# %%

def analyze_political_flips(trial1, trial2, role1, role2):
    """Analyze vote flips by political affiliation between two trials."""
    flip_counts = {'Republican': 0, 'Democrat': 0}
    total_counts = {'Republican': 0, 'Democrat': 0}
    
    for i in range(1, 21):
        # Load votes for both trials
        votes1 = pd.read_json(f"../data/{role1}/{trial1}/{role1[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        votes2 = pd.read_json(f"../data/{role2}/{trial2}/{role2[0]}_policy_{i}_votes.jsonl", encoding='cp1252', lines=True)
        
        # Add index columns
        votes1['idx'] = range(len(votes1))
        votes2['idx'] = range(len(votes2))
        
        # Merge with biographies
        votes1 = votes1.merge(biographies[['participant_idx', 'Political Affiliation']], 
                            left_on='idx', right_on='participant_idx', how='left')
        votes2 = votes2.merge(biographies[['participant_idx', 'Political Affiliation']], 
                            left_on='idx', right_on='participant_idx', how='left')
        
        # Merge votes
        merged = pd.merge(votes1, votes2, on='idx', suffixes=('_1', '_2'))
        
        # Count flips by political affiliation
        for party in ['Republican', 'Democrat']:
            party_votes = merged[merged['Political Affiliation_1'] == party]
            total_counts[party] += len(party_votes)
            
            flipped = party_votes[
                ((party_votes['vote_1'] == 'Yes') & (party_votes['vote_2'] == 'No')) |
                ((party_votes['vote_1'] == 'No') & (party_votes['vote_2'] == 'Yes'))
            ]
            flip_counts[party] += len(flipped)
    
    # Calculate percentages
    results = {
        'trial1': trial1,
        'trial2': trial2,
        'role1': role1,
        'role2': role2,
        'republican_flips': flip_counts['Republican'],
        'republican_total': total_counts['Republican'],
        'republican_percent': (flip_counts['Republican'] / total_counts['Republican'] * 100) if total_counts['Republican'] > 0 else 0,
        'democrat_flips': flip_counts['Democrat'],
        'democrat_total': total_counts['Democrat'],
        'democrat_percent': (flip_counts['Democrat'] / total_counts['Democrat'] * 100) if total_counts['Democrat'] > 0 else 0
    }
    
    return results

# Collect results for all pairings
political_results = []

# 1. Delegate-Trustee combinations
for t1, t2 in product(trials, trials):
    results = analyze_political_flips(t1, t2, 'delegate', 'trustee')
    political_results.append(results)

# 2. Delegate-Delegate combinations
for t1, t2 in product(trials, trials):
    if t1 != t2:
        results = analyze_political_flips(t1, t2, 'delegate', 'delegate')
        political_results.append(results)

# 3. Trustee-Trustee combinations
for t1, t2 in product(trials, trials):
    if t1 != t2:
        results = analyze_political_flips(t1, t2, 'trustee', 'trustee')
        political_results.append(results)

# Convert to DataFrame
political_df = pd.DataFrame(political_results)

# Format results
political_df['republican_percent'] = political_df['republican_percent'].round(1)
political_df['democrat_percent'] = political_df['democrat_percent'].round(1)
# Print results
print("\nVote Flips by Political Affiliation:")
print(political_df[['trial1', 'trial2', 'role1', 'role2', 
                    'republican_flips', 'republican_total', 'republican_percent',
                    'democrat_flips', 'democrat_total', 'democrat_percent']])

# Save to CSV
political_df.to_csv("../data/political_flip_analysis.csv", index=False)

# %%
# Get only numeric columns
numeric_cols = political_df.select_dtypes(include=['int64', 'float64']).columns

political_df.groupby(["role1", "role2"])[numeric_cols].mean()
# %%
political_df.groupby(["role1", "role2"]).mean().to_csv("../data/political_flip_analysis_mean.csv")
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