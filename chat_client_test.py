import unittest
from chat_client import ClientInterface
import tempfile
import json
import os


class TestClientInterface_Init(unittest.TestCase):
    def test_normal_init(self):
        client_api = None
        models = ["A", "B", "C"]
        client = ClientInterface(client_api, models)

        self.assertEqual(client.models, models)
        self.assertEqual(client.model, models[0])
        self.assertEqual(client.client_api, None)

    def test_init_with_empty_models(self):
        client_api = None
        models = []
        with self.assertRaises(ValueError):
            ClientInterface(client_api, models)


class TestClientInterface_ChatMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Base class doesn't invoke client API, so None is fine for this test
        client_api = None
        models = ["A", "B", "C"]
        cls.client = ClientInterface(client_api, models)

    def tearDown(self):
        self.client.reset()

    def test_add_message(self):
        # Arrange - input roles and messages
        expected_role_1 = "user"
        expected_message_1 = "hello"
        expected_role_2 = "assistant"
        expected_message_2 = "hi there"

        # Act - add messages
        self.client.add_message(role=expected_role_1, message=expected_message_1)
        self.client.add_message(role=expected_role_2, message=expected_message_2)

        # Assert - messages added correctly
        self.assertEqual(len(self.client.messages), 2)
        self.assertEqual(self.client.messages[0]["role"], expected_role_1)
        self.assertEqual(self.client.messages[0]["content"], expected_message_1)
        self.assertEqual(self.client.messages[1]["role"], expected_role_2)
        self.assertEqual(self.client.messages[1]["content"], expected_message_2)

    def test_get_response_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.client.get_response()

    def test_stream_generator_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.client.stream_generator()

    def test_prompt(self):
        # Arrange - input prompt and expected output
        prompt = "Hello, how are you?"
        expected_messages = [{"role": "user", "content": prompt}]

        # Act - call prompt
        self.client.prompt(prompt)

        # Assert - after prompt, stored messages contain prompt
        self.assertEqual(self.client.messages, expected_messages)

    def test_reset(self):
        # Arrange - add messages to object
        self.client.messages = [{"role": "user", "content": "test"}]

        # Act - reset class
        self.client.reset()

        # Assert - messages are empty
        self.assertEqual(len(self.client.messages), 0)


class TestClientInterface_FileMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Base class doesn't invoke client API, so None is fine for this test
        client_api = None
        models = ["A", "B", "C"]
        cls.client = ClientInterface(client_api, models)
        cls.workspace = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def tearDown(self):
        self.client.reset()

    def test_save_to_file_success(self):
        # Arrange - Set up messages and filepath
        expected_messages = [{"role": "assistant", "content": "test"}]
        self.client.messages = expected_messages
        filepath = os.path.join(self.workspace.name, "test_save.json")

        # Act - Save messages to file
        self.client.save_to_file(filepath)

        # Assert - loaded file messages should match expected
        with open(filepath, "r") as file:
            new_messages = json.load(file)
            self.assertEqual(new_messages, expected_messages)

    def test_save_to_file_failure(self):
        # Arrange - Set up expected messages and filepath
        self.client.messages = [{"role": "user", "content": "test"}]
        filepath = os.path.join(self.workspace.name, "test_save.json")

        # Assert - raise AssertionError when trying to save messages with last message from user
        with self.assertRaises(AssertionError):
            self.client.save_to_file(filepath)

    def test_load_from_file(self):
        # Arrange - Write expected messages to test file
        expected_messages = [{"role": "user", "content": "test"}]
        filepath = os.path.join(self.workspace.name, "test_load.json")
        with open(filepath, "w") as file:
            json.dump(expected_messages, file)

        # Act - load from file
        self.client.load_from_file(filepath)

        # Assert - loaded file messages should match expected
        self.assertEqual(self.client.messages, expected_messages)


if __name__ == "__main__":
    unittest.main()
