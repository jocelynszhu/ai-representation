#%%
import pandas as pd
import numpy as np
from itertools import product
from load_pairwise_data import load_pairwise_data
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# %%
base_llm = "claude-3-sonnet"
prompts = ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"]
policies_to_ignore = None
all_data = load_pairwise_data(base_llm, prompts, policies_to_ignore=policies_to_ignore)
#%%

biographies = pd.read_json("rep_biographies.jsonl", lines=True)
biographies['id_1'] = range(len(biographies))
# %%
joined = all_data.merge(biographies, on='id_1', how='left')
#%%
simple = joined.rename({"Political Affiliation": "political_affiliation", 
                        "Age Group": "age_group",
                        "Marital Status": "marital_status"}, axis=1)
simple = simple[simple.same_condition == False]
demographics = simple.columns[17:]
cols_to_use = ["policy_id", "flipped", "id_1"] + list(demographics)
simple = simple[cols_to_use]
simple["flipped"] = simple["flipped"].astype(int)
simple = simple.dropna()
#%%
formula = "flipped ~ C(Gender) +  C(age_group) + C(Income) + C(political_affiliation) + C(Race) + C(Education)  + C(policy_id)"
# %%
try:
    model = smf.logit(formula, data=simple).fit(method='bfgs', maxiter=1000, disp=True)# cov_kwds={'groups': simple['id_1']}, cov_type='cluster', )
except PerfectSeparationError:
    print("Perfect separation detected")
model_summary = model.summary()
#%%
simple
# %%
print(model_summary)
# %%
