from openai import OpenAI
import ollama
import anthropic
import os
import json
def get_llm_response(api_key, task, prompt, model="gpt-4o", **kwargs):
    if model[0:3] == "gpt":
        try:
            client = OpenAI(api_key=api_key)
            messages = [
                {
                    "role": "system",
                    "content": task.format(kwargs),
                },
                {"role": "user", "content": prompt},
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                )
            gpt_content = response.choices[0].message.content
            return gpt_content
        except Exception as e:
            print(e)
            return None
    elif model[:5] =="llama":
        try:
            client = ollama.Client(host="localhost")
            response = ollama.chat(
                model=model,                          
                options={"temperature": 0},
                messages = [
                    {
                        "role": "system",
                        "content": task.format(kwargs),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return response['message']['content']
        except Exception as e:
            print(e)
            return None

    elif model == "claude-3-sonnet":
        try:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            
            messages = [
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ]
   
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                        max_tokens=1024,
                        messages=messages,
                        system = task,
                        temperature=0.0
                    )
           # text = json.loads(response.content[0].text)
            return process_claude_response(response.content[0].text)
        except Exception as e:
            print(e)
            return None

def process_claude_response(response):
    clean = response[ response.find('{') : response.rfind('}')+1 ]
    clean = clean.replace("\n", " ")
    return clean

def process_gpt_response(response, profile_id):
    clean = response[ response.find('{') : response.rfind('}')+1 ]
    clean = clean.replace("\n", " ")
    clean = clean[0] + f' "id": {profile_id}, ' + clean[1:]
    return clean