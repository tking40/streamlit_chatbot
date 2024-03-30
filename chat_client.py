import openai
import anthropic
from dotenv import load_dotenv
from typing import List, Any

load_dotenv()


OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]
ANTHROPIC_MODELS = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]


class ClientInterface():
    def __init__(self, client: Any, models: List[str]):
        self.client = client
        self.models = models
        self.model = models[0]
        self.messages = []

    def add_message(self, role: str, message: str):
        self.messages.append({"role": role, "content": message})

    def get_response(self):
        raise NotImplementedError("get_response not implemented!")

    def stream_response(self):
        raise NotImplementedError("stream_response not implemented!")
    
    def reset(self):
        self.messages = []


class AnthropicClient(ClientInterface):
    def __init__(self):
        super().__init__(client=anthropic.Anthropic(), models=ANTHROPIC_MODELS)

    def get_response(self) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=self.messages,
        )
        message = response.content[0].text
        self.add_message(role="assistant", message=message)
        return message
    
    def prompt(self, prompt: str) -> None:
        self.add_message(role="user", message=prompt)

class OpenAIClient(ClientInterface):
    def __init__(self):
        super().__init__(client=openai.OpenAI(), models=OPENAI_MODELS)

    def get_response(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False,
        )
        message = response.choices[0].message.content
        self.add_message(role="assistant", message=message)
        return message
    
    def prompt(self, prompt: str) -> None:
        self.add_message(role="user", message=prompt)
    
# # Example:
# client_api = AnthropicClient()
# client_api.prompt("Hello")
# print(client_api.get_response())
# client_api.prompt("What is 2 + 2?")
# print(client_api.get_response())