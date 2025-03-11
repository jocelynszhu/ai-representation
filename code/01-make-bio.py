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

# %%
