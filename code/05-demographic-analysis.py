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