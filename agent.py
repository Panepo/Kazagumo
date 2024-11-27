from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from models.modelsOV import llm, embedding
elif backend == "ollama":
  from models.modelsOllama import llm, embedding
else:
  raise ValueError(f"Unknown backend: {backend}")

from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from tools.math import multiply_tool, add_tool

Settings.embed_model = embedding
Settings.llm = llm

tools = [multiply_tool, add_tool]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
