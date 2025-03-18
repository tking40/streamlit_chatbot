import litellm

from typing import List, Any, Dict
import json
import tiktoken


OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "o1-mini", "o1-preview", "gpt-4.5-preview-2025-02-27"]
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
]
ANTHROPIC_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
]
GEMINI_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

PROVIDER_MODELS = {
    "gemini": GEMINI_MODELS,
    "openai": OPENAI_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "groq": GROQ_MODELS,
}
DEFAULT_MODEL_NAME = "gemini/gemini-1.5-flash-latest"

google_safety_settings = [
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
        provider_models: Dict[str, List[str]],
        messages=None,
        assistant_role="assistant",
        user_role="user",
        system_prompt="",
    ):
        if not provider_models:
            raise ValueError("provider_models dict cannot be empty")
        self.client_api = client_api
        self.provider_models = provider_models
        # Messages are the only private member so far, as the format will be unique to each client.
        # Setter/getter methods will convert as necessary to the shared format
        self._messages = []
        if system_prompt:
            self.add_message(role="system", content=system_prompt)
        if messages is not None:
            self._messages.extend(messages)
        self.assistant_role = assistant_role
        self.user_role = user_role

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_messages(self):
        return self._messages

    def set_messages(self, messages):
        self._messages = messages

    def set_model(self, model_name, provider_name=None):
        if provider_name is not None:
            self.model_name = f"{provider_name}/{model_name}"
        else:
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


class LiteLLMClient(ClientInterface):
    def __init__(self, client_api=litellm, provider_models=PROVIDER_MODELS, **kwargs):
        super().__init__(
            client_api=client_api, provider_models=provider_models, **kwargs
        )
        self.set_model(DEFAULT_MODEL_NAME)

    def __str__(self):
        return "LiteLLM"

    def model_kwargs(self):
        kwargs = {}
        if self.model_name in GEMINI_MODELS:
            kwargs["safety_settings"] = google_safety_settings
        return kwargs

    def get_response(self) -> Any:
        response = self.client_api.completion(
            model=self.model_name,
            messages=self._messages,
            **self.model_kwargs(),
        )
        content = response.choices[0].message.content
        return content

    def stream_generator(self):
        response = self.client_api.completion(
            model=self.model_name,
            messages=self._messages,
            stream=True,
            **self.model_kwargs(),
        )
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                yield text


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    client = LiteLLMClient()
    print("")
    print("####### Rotate Clients #######")
    client.prompt("What is 2 + 2?")
    print("Client:", client.get_response())
    print("------------------------")
    client.prompt("And double that?")
    print("Client:", end=" ")
    for chunk in client.stream_generator():
        print(chunk, end="", flush=True)
    print("")
