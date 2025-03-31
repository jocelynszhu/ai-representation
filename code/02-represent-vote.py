
def run_llm(policy, community, content):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        messages = [
            {
                "role": "system",
                "content": BIO_PROMPT.format(
                    policy=policy,
                    community=community,
                ),
            },
            {"role": "user", "content": re.sub(r'\n', ' ', content)},
        ]

        

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={ "type": "json_object" }
        )
        gpt_content = response.choices[0].message.content
        
        return gpt_content
