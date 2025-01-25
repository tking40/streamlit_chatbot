import openai
import groq
import anthropic
import google.generativeai as genai
from typing import List, Any
import json
import tiktoken

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview"]
GROQ_MODELS = ["llama-3.3-70b-versatile", "gemma2-9b-it"]
ANTHROPIC_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
]
GOOGLE_MODELS = [
    "gemini-1.5-pro-latest",
    "models/gemini-1.5-flash-latest",
    "gemini-1.5-pro-exp-0827",
]
google_safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 4
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class ClientInterface:
    def __init__(
        self,
        client_api: Any,
        models: List[str],
        messages=None,
        assistant_role="assistant",
        user_role="user",
    ):
        if not models:
            raise ValueError("models list cannot be empty")
        self.client_api = client_api
        self.models = models
        self.model_name = None
        self.set_model(models[0])
        # Messages are the only private member so far, as the format will be unique to each client.
        # Setter/getter methods will convert as necessary to the shared format
        self._messages = []
        if messages is not None:
            self.set_messages(messages)
        self.assistant_role = assistant_role
        self.user_role = user_role

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_messages(self):
        return self._messages

    def set_messages(self, messages):
        self._messages = messages

    def set_model(self, model_name):
        if self.model_name == model_name:
            return
        self.model_name = model_name

    def get_response(self):
        raise NotImplementedError("get_response not implemented!")

    def stream_generator(self):
        raise NotImplementedError("stream_response not implemented!")

    def prompt(self, prompt: str) -> None:
        self.add_message(role=self.user_role, content=prompt)

    def reset(self):
        self._messages = []

    def save_to_file(self, filepath):
        messages = self.get_messages()
        assert (
            messages[-1]["role"] != self.user_role
        ), "Last message was from user, this shouldn't happen!"
        with open(filepath, "w") as file:
            json.dump(messages, file)

    def load_from_file(self, filepath):
        with open(filepath, "r") as file:
            self.set_messages(json.load(file))

    def count_tokens(self):
        return num_tokens_from_messages(self.get_messages())


class AnthropicClient(ClientInterface):
    def __init__(self, **kwargs):
        super().__init__(
            client_api=anthropic.Anthropic(), models=ANTHROPIC_MODELS, **kwargs
        )

    def __str__(self):
        return "Anthropic"

    def get_response(self) -> str:
        response = self.client_api.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=self._messages,
        )
        message = response.content[0].text
        self.add_message(role="assistant", content=message)
        return message

    def stream_generator(self):
        stream_manager = self.client_api.messages.stream(
            model=self.model_name,
            max_tokens=1024,
            messages=self._messages,
        )

        # Enter the context to get the MessageStream
        stream = stream_manager.__enter__()

        try:
            # Yield text from the stream
            for text in stream.text_stream:
                yield text
        finally:
            # Exit the context and clean up the stream
            stream_manager.__exit__(None, None, None)


class OpenAIClient(ClientInterface):
    def __init__(self, client_api=openai.OpenAI(), models=OPENAI_MODELS, **kwargs):
        super().__init__(client_api=client_api, models=models, **kwargs)

    def __str__(self):
        return "OpenAI"

    def get_response(self) -> str:
        response = self.client_api.chat.completions.create(
            model=self.model_name,
            messages=self._messages,
            stream=False,
        )
        content = response.choices[0].message.content
        self.add_message(role="assistant", content=content)
        return content

    def stream_generator(self):
        stream = self.client_api.chat.completions.create(
            model=self.model_name,
            messages=self._messages,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text


class GroqClient(OpenAIClient):
    def __init__(self, client_api=groq.Groq(), models=GROQ_MODELS, **kwargs):
        super().__init__(client_api=client_api, models=models, **kwargs)

    def __str__(self):
        return "Groq"


def chat_history_to_messages(history):
    return [
        {
            "role": "user" if m["role"] == "user" else "assistant",
            "content": m["parts"][0],
        }
        for m in history
    ]


def messages_to_chat_history(messages):
    history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in messages
    ]
    return history


class GoogleClient(ClientInterface):
    def __init__(self, **kwargs):
        genai.configure()
        # client api is configured in set_model
        super().__init__(
            client_api=None, models=GOOGLE_MODELS, assistant_role="model", **kwargs
        )

    def __str__(self):
        return "Google"

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "parts": [content]})

    def get_messages(self):
        return chat_history_to_messages(self._messages)

    def set_messages(self, messages):
        self._messages = messages_to_chat_history(messages)

    def set_model(self, model_name):
        if self.model_name == model_name:
            return
        self.model_name = model_name
        self.client_api = genai.GenerativeModel(self.model_name)

    def get_response(self) -> str:
        response = self.client_api.generate_content(self._messages)
        self.add_message(role=self.assistant_role, content=response.text)
        return response.text

    def stream_generator(self):
        stream = self.client_api.generate_content(
            self._messages, stream=True, safety_settings=google_safety_settings
        )
        for chunk in stream:
            yield chunk.text

    def count_tokens(self):
        if not self._messages:
            return 0
        return self.client_api.count_tokens(self._messages).total_tokens


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    print("###### Client Hello #######")
    print("--- Anthropic ---")
    client = AnthropicClient()
    client.prompt("Hello")
    for chunk in client.stream_generator():
        print(chunk, end="", flush=True)
    print("")

    print("--- OpenAI ---")
    client = OpenAIClient()
    client.prompt("Hello")
    for chunk in client.stream_generator():
        print(chunk, end="", flush=True)
    print("")

    print("--- Google ---")
    client = GoogleClient()
    client.prompt("Hello")
    for chunk in client.stream_generator():
        print(chunk, end="", flush=True)
    print("")

    print("")
    print("####### Rotate Clients #######")
    client_api_1 = OpenAIClient()
    client_api_1.prompt("What is 2 + 2?")
    print("OpenAI:", client_api_1.get_response())
    print("------------------------")
    client_api_2 = GoogleClient(messages=client_api_1.get_messages())
    client_api_2.prompt("And double that?")
    print("Google:", client_api_2.get_response())
    print("------------------------")
    client_api_3 = AnthropicClient(messages=client_api_2.get_messages())
    client_api_3.prompt("And double that?")
    print("Anthropic:", client_api_3.get_response())
