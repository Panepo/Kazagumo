from llama_index.embeddings.ollama import OllamaEmbedding

embedding = OllamaEmbedding(
    model_name="bge-m3",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
