import ollama
import logging
import sys
# Import configuration settings for Ollama host and model from the src.config module.
from src.config import OLLAMA_HOST, OLLAMA_MODEL

# Configure logging for the module.
# Log messages will be displayed in the console (sys.stdout) with a timestamp,
# log level, and the message itself. INFO level messages and above will be shown.
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaClient:
    """
    A client class to interact with a local Ollama server.
    This class handles checking server availability and generating responses from LLMs.
    """
    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL):
        """
        Initializes the OllamaClient.

        Args:
            host (str): The URL of the Ollama server. Defaults to OLLAMA_HOST from config.
            model (str): The name of the LLM model to use. Defaults to OLLAMA_MODEL from config.
        """
        self.host = host
        self.model = model
        # Initialize the Ollama client instance using the provided host.
        self.client = ollama.Client(host=self.host)
        logging.info(f"OllamaClient initialized with host: {self.host}, model: {self.model}")

    def is_ollama_running(self) -> bool:
        """
        Checks if the Ollama server is running and accessible at the configured host.

        Returns:
            bool: True if the server is running, False otherwise.
        """
        try:
            # Attempt to list local models as a way to verify connectivity to the server.
            self.client.list()
            logging.info("Ollama server is running and accessible.")
            return True
        except ollama.ResponseError as e:
            # Catch specific Ollama API response errors (e.g., connection refused, server down).
            logging.error(f"Ollama server not accessible at {self.host}. Error: {e}")
            return False
        except Exception as e:
            # Catch any other unexpected errors during the connectivity check.
            logging.error(f"An unexpected error occurred while checking Ollama server: {e}")
            return False

    def generate_response(self, prompt: str, model: str = None, timeout: int = 120) -> str | None:
        """
        Sends a text prompt to the specified Ollama LLM and retrieves its generated response.
        Uses streaming to allow for non-blocking operations and better user feedback.

        Args:
            prompt (str): The text prompt to send to the LLM.
            model (str, optional): The specific LLM model to use for this request.
                                   If None, the client's default model (self.model) is used.
            timeout (int): Maximum seconds to wait for response (default 120s).

        Returns:
            str | None: The generated text response from the LLM, or None if an error occurred.
        """
        # First, ensure the Ollama server is running before attempting to generate a response.
        if not self.is_ollama_running():
            logging.error("Cannot generate response: Ollama server is not running.")
            return None

        # Determine which model to use: the specified one for this call, or the default client model.
        target_model = model if model else self.model
        try:
            logging.info(f"Generating response from model '{target_model}' with prompt: '{prompt[:50]}...'")
            # Use streaming mode to get real-time feedback
            response_text = ""
            response = self.client.generate(model=target_model, prompt=prompt, stream=True)
            
            # Handle both real streaming and mock/dict responses
            if isinstance(response, dict):
                # Handle dict response (for non-streaming or mocked responses)
                return response.get('response')
            
            # Collect streamed response chunks
            for chunk in response:
                if isinstance(chunk, dict) and 'response' in chunk:
                    response_text += chunk['response']
            
            return response_text if response_text else None
        except ollama.ResponseError as e:
            # Handle errors specifically from the Ollama API during response generation.
            logging.error(f"Error generating response from model '{target_model}': {e}")
            return None
        except Exception as e:
            # Handle any other unexpected errors during response generation.
            logging.error(f"An unexpected error occurred during response generation: {e}")
            return None

# Example usage block:
# This code runs only when the script is executed directly, not when imported as a module.
if __name__ == "__main__":
    # Create an instance of the OllamaClient.
    client = OllamaClient()
    
    # Check if the Ollama server is running before attempting interactions.
    if client.is_ollama_running():
        print(f"Using model: {client.model}")
        
        # Define a test prompt.
        test_prompt = "What is the capital of France?"
        # Generate a response using the client.
        response = client.generate_response(test_prompt)
        
        # Print the response or an error message if generation failed.
        if response:
            print(f"Response: {response}")
        else:
            print("Failed to get response from Ollama.")
    else:
        # Inform the user if the Ollama server is not accessible.
        print("Ollama server is not running. Please start it and ensure the model is available.")