#%%

from load_pairwise_data import * 
# %%

# %%
def compute_bias(model):
    data = load_pairwise_data(model, ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"], num_policies=20)
    policies = pd.read_json("../self_selected_policies.jsonl", lines=True)
    policies["policy_id"] = policies["id"]

    default_votes = pd.read_json(f"../data/defaults/{model}.jsonl", lines=True)
    default_votes["policy_id"] = default_votes["id"]
    default_votes = default_votes.rename(columns={"vote": "default_vote"})

    data_merged = data.merge(default_votes, on="policy_id", how="left")
    flipped_votes = data_merged[~(data_merged["flipped"]) & ~(data_merged["same_condition"])]

    for index, row in flipped_votes.iterrows():
        if row["source_1"] == "delegate":
            flipped_votes.loc[index, "delegate_vote"] = row["vote_1"]
        else:
            flipped_votes.loc[index, "delegate_vote"] = row["vote_2"]
        if row["source_2"] == "delegate":
            flipped_votes.loc[index, "trustee_vote"] = row["vote_1"]
        else:
            flipped_votes.loc[index, "trustee_vote"] = row["vote_2"]

    flipped_votes["source_1"].value_counts()

    flipped_votes["trustee_agrees_default"] = (flipped_votes["trustee_vote"] == flipped_votes["default_vote"])

    return flipped_votes["trustee_agrees_default"].value_counts(normalize=True)


#%%
models = ["gpt-4o", "llama-3.2"]
for model in models:
    print(model)
    print(compute_bias(model))
    print()
# %%
