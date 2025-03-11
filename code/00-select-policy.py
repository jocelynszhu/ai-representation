# %%
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
import os
from openai import OpenAI
from pydantic import BaseModel


import re
import json
import importlib
import statistics
import utils

importlib.reload(utils)

# %%
policy_statements = pd.read_json(path_or_buf='../230118_chinchilla_questions.json')
affirmative_statements = policy_statements['affirming_statement'].tolist()
len(affirmative_statements)

# %% Remove UK specific references
exclude = ["UK", "United Kingdom", "Britain", "British", "Heathrow", "NHS"]
filtered_affirm = [item for item in affirmative_statements if not any(word in item for word in exclude)]
len(filtered_affirm)
# %%
SELECT_PROMPT = """ You are a U.S. policymaker. You are tasked with selecting a subset of policies that are most relevant to the United States and spans diverse topics. You will be given many affirmative policy statements, separated by a line break. Select the 20 most relevant policy statements."""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
messages = [
    {
        "role": "system",
        "content": SELECT_PROMPT,
    },
    {"role": "user", "content": re.sub(r'\n', ' ', '\n'.join(filtered_affirm))},
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)
        
# %%
gpt_content = response.choices[0].message.content
gpt_content
# %%
lines = gpt_content.split("\n")
selected_policies = [json.dumps({"id": int(line.split(".")[0]), "statement": line.split(". ", 1)[1]}) for line in lines]

# Write to file
with open("../selected_policies.jsonl", "w") as f:
    f.write("\n".join(selected_policies))
# %%
