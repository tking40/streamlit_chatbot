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
        # Base class doesn't invoke client API, so None is fine for this test
        client_api = None
        models = ["A", "B", "C"]
        cls.client = ClientInterface(client_api, models)

    def tearDown(self):
        self.client.reset()

    def test_add_message(self):
        # Arrange
        expected_role = "user"
        expected_message = "testing123"

        # Act
        self.client.add_message(role=expected_role, message=expected_message)

        # Assert
        self.assertEqual(self.client.messages[-1]["role"], expected_role)
        self.assertEqual(self.client.messages[-1]["content"], expected_message)

    def test_get_response_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.client.get_response()

    def test_stream_response_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.client.stream_response()

    def test_prompt(self):
        # Arrange
        prompt = "Hello, how are you?"

        # Act
        self.client.prompt(prompt)

        # Assert
        self.assertEqual(len(self.client.messages), 1)
        self.assertEqual(self.client.messages[-1]["role"], "user")
        self.assertEqual(self.client.messages[-1]["content"], prompt)

    def test_reset(self):
        # Arrange
        self.client.messages = [{"role": "user", "content": "test"}]

        # Act
        self.client.reset()

        # Assert
        self.assertEqual(len(self.client.messages), 0)


if __name__ == "__main__":
    unittest.main()
