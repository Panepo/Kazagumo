from llama_index.core.agent import ReActAgent
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from models.ovmodels import llm, embedding
from tools.math import multiply_tool, add_tool

from pathlib import Path

text_example_en_path = Path("text_example_en.pdf")

Settings.embed_model = embedding
Settings.llm = llm
loader = PyMuPDFReader()
documents = loader.load(file_path=text_example_en_path)
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)

rag_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="Xeon6",
    description="A RAG engine with some basic facts about Intel Xeon 6 processors with E-cores",
)

agent = ReActAgent.from_tools([multiply_tool, add_tool, rag_tool], llm=llm, verbose=True)
