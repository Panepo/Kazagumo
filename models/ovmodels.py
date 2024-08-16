import openvino as ov
from llama_index.llms.openvino import OpenVINOLLM
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding

core = ov.Core()
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
model_path = "models/ovmodels/Mistral-7B-Instruct-v0.3-ov-int4"
embedding_model_path = "models/ovmodels/bge-small-en-v1.5-ov"

def completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


llm = OpenVINOLLM(
    model_id_or_path=str(model_path),
    context_window=3900,
    max_new_tokens=1000,
    model_kwargs={"ov_config": ov_config},
    device_map="GPU",
    completion_to_prompt=completion_to_prompt,
)

embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device="GPU")
