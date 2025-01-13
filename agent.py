from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "ollama":
  from modelsOllama import llm
  from embeddingsOllama import embedding
else:
  raise ValueError(f"Unknown backend: {backend}")

from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from tools.math import multiply_tool, add_tool, divide_tool
from rag import rag_tool

Settings.llm = llm
Settings.embed_model = embedding

tools = [multiply_tool, add_tool, divide_tool, rag_tool]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
