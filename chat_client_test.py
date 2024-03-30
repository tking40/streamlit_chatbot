import unittest
from chat_client import ClientInterface


class TestClientInterface_Init(unittest.TestCase):
    def test_normal_init(self):
        client_api = None
        models = ["A", "B", "C"]
        client = ClientInterface(client_api, models)

        self.assertEqual(client.models, models)
        self.assertEqual(client.model, models[0])
        self.assertEqual(client.client_api, None)


class TestClientInterface_Methods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Base class doesn't invoke any client methods, so None is fine for this test
        client_api = None
        models = ["A", "B", "C"]
        cls.client = ClientInterface(client_api, models)

    def test_prompt(self):
        # Act
        prompt = "Hello, how are you?"
        self.client.prompt(prompt)

        # Assert
        assert len(self.client.messages) == 1
        assert self.client.messages[-1]["role"] == "user"
        assert self.client.messages[-1]["content"] == prompt


unittest.main()
