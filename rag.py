from dotenv import load_dotenv
import os

load_dotenv()
backend = os.getenv("BACKEND")

if backend == "openvino":
  from models.modelsOV import llm, embedding
elif backend == "cuda" or backend == "cpu":
  from models.modelsOllama import llm, embedding
else:
  raise ValueError(f"Unknown backend: {backend}")

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata

text_example_en_path = "2.pdf"

Settings.embed_model = embedding
Settings.llm = llm

reader = SimpleDirectoryReader(input_files=[text_example_en_path])
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents)
vector_tool = QueryEngineTool(
  index.as_query_engine(streaming=True),
  metadata=ToolMetadata(
    name="vector_search",
    description="Useful for searching for basic facts about 'Intel Xeon 6 processors'",
  ),
)
