# src/config.py

# This file stores global configuration settings for the entire project.
# It includes parameters for external services like Ollama and other
# potentially configurable aspects of the simulation.

# --- Ollama Configuration ---
# OLLAMA_HOST: Specifies the network address where the local Ollama server is running.
#              Typically, Ollama runs on localhost port 11434 by default.
OLLAMA_HOST: str = "http://localhost:11434"

# OLLAMA_MODEL: Defines the default Large Language Model (LLM) that the OllamaClient
#               will attempt to use. Ensure this model is pulled and available
#               on your local Ollama server (e.g., by running 'ollama pull llama2').
#               This can be overridden when initializing agents or making specific requests.
OLLAMA_MODEL: str = "llama2"  # Example: "llama2", "gemma:7b", "mistral"

# --- Other Potential Configuration Settings ---
# This section can be expanded as the project grows to include:
# - Learning rates, discount factors, exploration rates for Q-learning.
# - Simulation parameters (e.g., default number of rounds, evolution interval).
# - Paths for input data or output logs.