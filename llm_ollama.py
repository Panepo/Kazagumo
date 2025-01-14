from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("MODEL")

from llama_index.llms.ollama import Ollama

if model == "llama3.2":
  llm = Ollama(model="llama3.2", request_timeout=360.0)
elif model == "phi3":
  raise ValueError("phi3 is not supported")
  llm = Ollama(model="phi3", request_timeout=360.0)
else:
  raise ValueError(f"Unknown model: {model}")
