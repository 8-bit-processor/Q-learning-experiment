import unittest
from unittest.mock import patch, MagicMock
from src.ollama_client import OllamaClient
from ollama import ResponseError

class TestOllamaClient(unittest.TestCase):

    def setUp(self):
        # Patch ollama.Client directly to avoid actual network calls
        self.patcher_ollama_client = patch('src.ollama_client.ollama.Client')
        self.mock_ollama_client_class = self.patcher_ollama_client.start()
        self.mock_ollama_client_instance = MagicMock()
        self.mock_ollama_client_class.return_value = self.mock_ollama_client_instance

        self.default_host = "http://localhost:11434"
        self.default_model = "test_model"
        self.client = OllamaClient(host=self.default_host, model=self.default_model)

    def tearDown(self):
        self.patcher_ollama_client.stop()

    def test_initialization(self):
        self.assertEqual(self.client.host, self.default_host)
        self.assertEqual(self.client.model, self.default_model)
        self.mock_ollama_client_class.assert_called_once_with(host=self.default_host)

    def test_is_ollama_running_success(self):
        self.mock_ollama_client_instance.list.return_value = {"models": []} # Simulate success

        result = self.client.is_ollama_running()
        self.assertTrue(result)
        self.mock_ollama_client_instance.list.assert_called_once()

    def test_is_ollama_running_response_error(self):
        self.mock_ollama_client_instance.list.side_effect = ResponseError("Connection refused")

        result = self.client.is_ollama_running()
        self.assertFalse(result)
        self.mock_ollama_client_instance.list.assert_called_once()

    def test_is_ollama_running_other_error(self):
        self.mock_ollama_client_instance.list.side_effect = Exception("Unexpected error")

        result = self.client.is_ollama_running()
        self.assertFalse(result)
        self.mock_ollama_client_instance.list.assert_called_once()

    @patch('src.ollama_client.OllamaClient.is_ollama_running', return_value=False)
    def test_generate_response_ollama_not_running(self, mock_is_running):
        prompt = "Hello"
        response = self.client.generate_response(prompt)
        self.assertIsNone(response)
        mock_is_running.assert_called_once()
        self.mock_ollama_client_instance.generate.assert_not_called()

    @patch('src.ollama_client.OllamaClient.is_ollama_running', return_value=True)
    def test_generate_response_success(self, mock_is_running):
        prompt = "What is your name?"
        expected_response_content = "I am an AI."
        # Mock streaming response
        self.mock_ollama_client_instance.generate.return_value = [
            {"response": "I am "},
            {"response": "an AI."}
        ]

        response = self.client.generate_response(prompt)
        self.assertEqual(response, expected_response_content)
        mock_is_running.assert_called_once()
        self.mock_ollama_client_instance.generate.assert_called_once_with(model=self.default_model, prompt=prompt, stream=True)

    @patch('src.ollama_client.OllamaClient.is_ollama_running', return_value=True)
    def test_generate_response_with_specified_model(self, mock_is_running):
        prompt = "What is your name?"
        specific_model = "gemma:7b"
        expected_response_content = "I am Gemma."
        # Mock streaming response
        self.mock_ollama_client_instance.generate.return_value = [
            {"response": "I am "},
            {"response": "Gemma."}
        ]

        response = self.client.generate_response(prompt, model=specific_model)
        self.assertEqual(response, expected_response_content)
        mock_is_running.assert_called_once()
        self.mock_ollama_client_instance.generate.assert_called_once_with(model=specific_model, prompt=prompt, stream=True)

    @patch('src.ollama_client.OllamaClient.is_ollama_running', return_value=True)
    def test_generate_response_ollama_response_error(self, mock_is_running):
        prompt = "Hello"
        self.mock_ollama_client_instance.generate.side_effect = ResponseError("Model not found")

        response = self.client.generate_response(prompt)
        self.assertIsNone(response)
        mock_is_running.assert_called_once()
        self.mock_ollama_client_instance.generate.assert_called_once()

    @patch('src.ollama_client.OllamaClient.is_ollama_running', return_value=True)
    def test_generate_response_other_error(self, mock_is_running):
        prompt = "Hello"
        self.mock_ollama_client_instance.generate.side_effect = Exception("Network issue")

        response = self.client.generate_response(prompt)
        self.assertIsNone(response)
        mock_is_running.assert_called_once()
        self.mock_ollama_client_instance.generate.assert_called_once()

if __name__ == '__main__':
    unittest.main()
