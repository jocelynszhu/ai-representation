#%%
import pandas as pd
import numpy as np
from itertools import product
from load_pairwise_data import load_pairwise_data
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import matplotlib.pyplot as plt
# # %%
# base_llm = "claude-3-sonnet"
# prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
# policies_to_ignore = None
# all_data = load_pairwise_data(base_llm, prompts, policies_to_ignore=policies_to_ignore)
# #%%

# biographies = pd.read_json("rep_biographies.jsonl", lines=True)
# biographies['id_1'] = biographies["ID"]
# # %%
# joined = all_data.merge(biographies, on='id_1', how='left')
# #%%
# simple = joined.rename({"Political Affiliation": "political_affiliation", 
#                         "Age Group": "age_group",
#                         "Marital Status": "marital_status"}, axis=1)
# simple = simple[simple.same_condition == False]
# demographics = simple.columns[17:]
# cols_to_use = ["policy_id", "flipped", "id_1"] + list(demographics)
# simple = simple[cols_to_use]
# simple["flipped"] = simple["flipped"].astype(int)
# #simple = simple.dropna()
# #%%
# # Filter out rows with any NA values in the columns we're using for regression
# simple[simple.isna().any(axis=1)]

# #%%
# formula = "flipped ~ C(Gender) +  C(age_group) + C(Income) + C(political_affiliation) + C(Race) + C(Education)  + C(policy_id)"
# # %%
# try:
#     model = smf.logit(formula, data=simple).fit(method='bfgs', maxiter=1000, disp=True)# cov_kwds={'groups': simple['id_1']}, cov_type='cluster', )
# except PerfectSeparationError:
#     print("Perfect separation detected")
# model_summary = model.summary()
# #%%
# simple
# # %%

# # Extract coefficients and confidence intervals
# coef_df = pd.DataFrame({
#     'coef': model.params,
#     'ci_lower': model.conf_int()[0],
#     'ci_upper': model.conf_int()[1],
#     'p_value': model.pvalues
# }).reset_index()
# coef_df = coef_df.rename(columns={'index': 'variable'})

# # Filter out reference categories and intercept
# coef_df = coef_df[~coef_df['variable'].isin(['Intercept', 'C(Gender)[T.Female]', 'C(age_group)[T.18-24]', 
#                                              'C(Income)[T.$0-$25,000]', 'C(political_affiliation)[T.Democrat]',
#                                              'C(Race)[T.White]', 'C(Education)[T.High School]',
#                                              'C(policy_id)[T.1]'])]

# # Sort by absolute coefficient value
# coef_df['abs_coef'] = abs(coef_df['coef'])
# coef_df = coef_df.sort_values('abs_coef', ascending=True)

# # Create coefficient plot
# plt.figure(figsize=(10, 12))
# y_pos = np.arange(len(coef_df))

# # Plot coefficients and CIs
# plt.hlines(y=y_pos, xmin=coef_df['ci_lower'], xmax=coef_df['ci_upper'], color='grey', alpha=0.5)

# # Plot points with different colors based on significance
# significant = coef_df['p_value'] < 0.05
# plt.scatter(coef_df.loc[significant, 'coef'], y_pos[significant], 
#             color='red', alpha=0.6, label='p < 0.05')
# plt.scatter(coef_df.loc[~significant, 'coef'], y_pos[~significant], 
#             color='blue', alpha=0.6, label='p ≥ 0.05')

# # Customize plot
# plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
# plt.yticks(y_pos, coef_df['variable'])
# plt.xlabel('Coefficient Value')
# plt.title('Regression Coefficients with 95% Confidence Intervals')

# # Set x-axis limits based on the data
# x_min = min(coef_df['ci_lower'].min(), -0.1)  # Add some padding
# x_max = max(coef_df['ci_upper'].max(), 0.1)   # Add some padding
# plt.xlim(x_min, x_max)

# plt.legend()
# plt.tight_layout()
# plt.xlim(-5, 5)

# plt.show()

# # Also print summary for reference
# print(model_summary)
# # %%
# model.conf_int()
# %%

def run_demographic_regression(data, prompt_1=None, prompt_2=None):
    """
    Run demographic regression analysis for a specific prompt pair.
    
    Args:
        data (pd.DataFrame): The full dataset
        prompt_1 (str): First prompt to filter by (e.g., "prompt-0")
        prompt_2 (str): Second prompt to filter by (e.g., "prompt-1")
    
    Returns:
        tuple: (model, model_summary, coef_df)
    """
    # Filter data for specific prompt pair if provided
    if prompt_1 is not None and prompt_2 is not None:
        filtered_data = data[
            ((data['combined_prompt_name_1'] == prompt_1) & (data['combined_prompt_name_2'] == prompt_2)) |
            ((data['combined_prompt_name_1'] == prompt_2) & (data['combined_prompt_name_2'] == prompt_1))
        ]
    else:
        filtered_data = data
    
    # Prepare data for regression
    simple = filtered_data.rename({
        "Political Affiliation": "political_affiliation", 
        "Age Group": "age_group",
        "Marital Status": "marital_status"
    }, axis=1)
    
    simple = simple[simple.same_condition == False]
    demographics = simple.columns[17:]
    cols_to_use = ["policy_id", "flipped", "id_1"] + list(demographics)
    simple = simple[cols_to_use]
    simple["flipped"] = simple["flipped"].astype(int)
    
    # Run regression
    formula = "flipped ~ C(Gender) + C(age_group) + C(Income) + C(political_affiliation) + C(Race) + C(Education) + C(policy_id)"
    
    try:
        model = smf.logit(formula, data=simple).fit(method='bfgs', maxiter=1000, disp=True)
    except PerfectSeparationError:
        print("Perfect separation detected")
        return None, None, None
    
    model_summary = model.summary()
    
    # Extract coefficients and confidence intervals
    coef_df = pd.DataFrame({
        'coef': model.params,
        'ci_lower': model.conf_int()[0],
        'ci_upper': model.conf_int()[1],
        'p_value': model.pvalues
    }).reset_index()
    coef_df = coef_df.rename(columns={'index': 'variable'})
    
    # Filter out reference categories and intercept
    coef_df = coef_df[~coef_df['variable'].isin([
        'Intercept', 
        'C(Gender)[T.Female]', 
        'C(age_group)[T.18-24]', 
        'C(Income)[T.$0-$25,000]', 
        'C(political_affiliation)[T.Democrat]',
        'C(Race)[T.White]', 
        'C(Education)[T.High School]',
        'C(policy_id)[T.1]'
    ])]
    
    return model, model_summary, coef_df

def plot_coefficients(coef_df, title="Regression Coefficients with 95% Confidence Intervals"):
    """
    Plot regression coefficients with confidence intervals.
    
    Args:
        coef_df (pd.DataFrame): DataFrame containing coefficients and confidence intervals
        title (str): Plot title
    """
    # Sort by absolute coefficient value
   # coef_df['abs_coef'] = abs(coef_df['coef'])
   # coef_df = coef_df.sort_values('abs_coef', ascending=True)
    
    # Create coefficient plot
    plt.figure(figsize=(10, 12))
    y_pos = np.arange(len(coef_df))
    
    # Plot coefficients and CIs
    plt.hlines(y=y_pos, xmin=coef_df['ci_lower'], xmax=coef_df['ci_upper'], color='grey', alpha=0.5)
    
    # Plot points with different colors based on significance
    significant = coef_df['p_value'] < 0.05
    plt.scatter(coef_df.loc[significant, 'coef'], y_pos[significant], 
                color='red', alpha=0.6, label='p < 0.05')
    plt.scatter(coef_df.loc[~significant, 'coef'], y_pos[~significant], 
                color='blue', alpha=0.6, label='p ≥ 0.05')
    
    # Customize plot
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.yticks(y_pos, coef_df['variable'])
    plt.xlabel('Coefficient Value')
    plt.title(title)
    
    # Set x-axis limits based on the data
    #x_min = min(coef_df['ci_lower'].min(), -0.1)
    #x_max = max(coef_df['ci_upper'].max(), 0.1)
    x_min = -5
    x_max = 5
    plt.xlim(x_min, x_max)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
#%%
    # Load data
base_llm = "claude-3-sonnet"
prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
policies_to_ignore = None
all_data = load_pairwise_data(base_llm, prompts, policies_to_ignore=policies_to_ignore)
different_conditions = all_data[all_data.same_condition == False]
#%%
all_pairs = all_data.filter(["combined_prompt_name_1", "combined_prompt_name_2"]).drop_duplicates().reset_index(drop=True)
#%%
# Load biographies
biographies = pd.read_json("rep_biographies.jsonl", lines=True)
biographies['id_1'] = biographies["ID"]

# Join data
joined = different_conditions.merge(biographies, on='id_1', how='left')
n = None
# Example: Run regression for prompt-0 vs prompt-1
if n is not None:   
    prompt_one = all_pairs.iloc[n]["combined_prompt_name_1"]
    prompt_two = all_pairs.iloc[n]["combined_prompt_name_2"]        
else:
    prompt_one = None
    prompt_two = None

model, model_summary, coef_df = run_demographic_regression(joined, prompt_one, prompt_two)
if model is not None:
    plot_coefficients(coef_df, title="Regression Coefficients (prompt-0 vs prompt-1)")
    print(model_summary)
#%%
prompt_two
# %%
