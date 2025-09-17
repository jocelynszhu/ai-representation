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
import random
import utils

importlib.reload(utils)

# %% load demographics dictionary
demographics = json.load(open("../data/demographics.json", "r"))

# %%

BIO_PROMPT="""
You are given a dictionary with possible values for the following demographics: Age, Gender, Race/Ethnicity, Income level, Education level, Occupation, Marital Status, Household Size, Geographic Location, Religion, Language, Housing Status, Health Status, and Political affiliation. 

Select demographics to create 100 diverse, plausible biographies that represent the demographic makeup of the United States.

Return the bios in JSON format with the keys matching the keys of the demographics dictionary, and the values selected from the demographics dictionary. Label each biography with a UUID.
"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
messages = [
    {
        "role": "system",
        "content": BIO_PROMPT
    },
    {"role": "user", "content": str(demographics)},
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={ "type": "json_object" }
)
# %%
gpt_response = response.choices[0].message.content

bios = json.loads(gpt_response)
bios
# %%
with open("biographies.jsonl", "w") as file:
    for key, value in bios.items():
        print(key, value, "\n")
        json_record = {"UUID": key, **value} 
        file.write(json.dumps(json_record) + "\n")

# %% Create written profiles
biographies = pd.read_json("rep_biographies.jsonl", lines=True)vc
# %%

PROFILE_PROMPT = """
You are a biographer. Generate a detailed, plausible biography for an individual with the given demographics.

Return the written profile in JSONL form with two key-value pairs, the ID given in the demographics dictionary and the Profile in natural language form.
"""
def run_llm(PROFILE_PROMPT, bio):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        messages = [
            {
                "role": "system",
                "content": PROFILE_PROMPT,
            },
            {"role": "user", "content": bio},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        gpt_content = response.choices[0].message.content
        
        return gpt_content


# %%
for index, row in biographies.iterrows():
    bio_dict = row.to_dict()
    response = run_llm(PROFILE_PROMPT, str(bio_dict))
    print(f"Generated profile for ID {bio_dict['ID']}")
    
    with open("gpt-o_written_profiles.jsonl", "a") as file:
        file.write(response + "\n")
# %%
