from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("MODEL")

if model == "llama":
  model_path = "models/ovmodels/Meta-Llama-3-8B-Instruct-ov/"

  def completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
elif model == "phi3":
  model_path = "models/ovmodels/phi-3-mini-instruct/"

  def completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"
else:
  raise ValueError(f"Unknown model: {model}")


from llama_index.llms.openvino import OpenVINOLLM
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams

ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

embedding_model_path = "models/ovmodels/bge-small-en-v1.5-ov/"

llm = OpenVINOLLM(
  model_id_or_path=str(model_path),
  context_window=3900,
  max_new_tokens=1000,
  model_kwargs={"ov_config": ov_config, "trust_remote_code": True},
  generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True},
  device_map="GPU",
  completion_to_prompt=completion_to_prompt,
)

embedding = OpenVINOEmbedding(model_id_or_path=embedding_model_path, device="GPU")
