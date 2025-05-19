#%%
import pandas as pd
import os

# Define the path to the data directory
data_dir = os.path.join('..', 'data', 'demo')
coefs_dir = os.path.join(data_dir, 'coefs')

# Read the coefficient files
gpt4_coefs = pd.read_csv(os.path.join(coefs_dir, 'or_df_small_gpt-4o.csv'))
llama_coefs = pd.read_csv(os.path.join(coefs_dir, 'or_df_small_llama-3.2.csv'))
claude_coefs = pd.read_csv(os.path.join(coefs_dir, 'or_df_small_claude-3-sonnet-v2.csv'))

# Display coefficient information
print("\nGPT-4 Coefficients:")
print(gpt4_coefs)
print("\nLlama Coefficients:")
print(llama_coefs)
print("\nClaude Coefficients:")
print(claude_coefs)

# %%
claude_coefs
# %%
all_data = pd.concat([gpt4_coefs, llama_coefs, claude_coefs])
all_data.to_csv(os.path.join(coefs_dir, 'all_data.csv'), index=False)

#%%
import numpy as np
all_data["sig"] = all_data["sig"].replace(np.nan, "")
all_data ["odds.ratio"] = all_data ["odds.ratio"].round(2)
all_data["odds.ratio"] = all_data["odds.ratio"].astype(str) 
all_data["odds.ratio"] = all_data["odds.ratio"] + all_data["sig"]

# %%
all_data.pivot(index=["variable", "level"], columns="model", values="odds.ratio")
# %%
all_data
# %%
