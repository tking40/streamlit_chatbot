from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

MODEL = "gpt-3.5-turbo"


def fetch_response(messages, model):
    collected_response = ""
    for chunk in client.chat.completions.create(
        model=model, messages=messages, temperature=0.5, stream=True
    ):
        content = chunk.choices[0].delta.content
        if content is not None:
            print(content, end="", flush=True)
            collected_response += content
    print("")
    return collected_response


messages = []
prompt = input("User: ")
while prompt:
    messages.append({"role": "user", "content": prompt})
    if prompt == "redo":
        messages = messages[:-2]
    response = fetch_response(messages, MODEL)
    messages.append({"role": "system", "content": response})
    prompt = input("User: ")
