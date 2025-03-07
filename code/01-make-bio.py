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