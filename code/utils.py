from openai import OpenAI
import ollama

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

def process_gpt_response(response, profile_id):
    clean = response[ response.find('{') : response.rfind('}')+1 ]
    clean = clean.replace("\n", " ")
    clean = clean[0] + f' "id": {profile_id}, ' + clean[1:]
    return clean