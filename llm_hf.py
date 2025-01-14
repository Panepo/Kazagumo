from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("MODEL")

if model == "llama3.2":
  raise NotImplementedError("LLAMA model is not supported in this version of the codebase. Please use the OpenVINO version.")
  model_path = "models/hfmodels/Meta-Llama-3-8B-Instruct-ov/"

  def completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
elif model == "phi3":
  model_path = "models/hfmodels/Phi-3-mini-4k-instruct/"

  def completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"
else:
  raise ValueError(f"Unknown model: {model}")

from transformers import AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
import torch


embedding_model_path = "models/hfmodels/BAAI--bge-small-en-v1.5/"

def completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|><|eot_id|><|start_header_id|>user<|end_header_id|>{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

quantization_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  device_map="cuda",
  load_in_4bit=True,
  trust_remote_code=True,
  torch_dtype="auto",
  _attn_implementation='eager',
)

llm = HuggingFaceLLM(
  model=model,
  context_window=3900,
  max_new_tokens=1000,
  device_map="cuda",
  completion_to_prompt=completion_to_prompt,
  generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True},
  model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},
)

embedding = HuggingFaceEmbedding(model_name=embedding_model_path, device="cuda")
