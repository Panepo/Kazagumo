import sys

def pip_install(*args):
    import subprocess  # nosec - disable B404:import-subprocess check

    cli_args = []
    for arg in args:
        cli_args.extend(str(arg).split(" "))
    subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

pip_install(
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu",
    "llama-index",
    "faiss-cpu",
    "pymupdf",
    "langchain",
    "llama-index-readers-file",
    "llama-index-vector-stores-faiss",
    "llama-index-llms-langchain",
    "llama-index-llms-huggingface>=0.3.0,<0.3.4",
    "llama-index-embeddings-huggingface>=0.3.0",
    "python-dotenv",
)
pip_install("git+https://github.com/huggingface/optimum-intel.git", "git+https://github.com/openvinotoolkit/nncf.git", "datasets", "accelerate", "gradio")
pip_install("--pre", "-U", "openvino>=2024.2", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
pip_install("--pre", "-U", "openvino-tokenizers[transformers]>=2024.2", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly")
pip_install("--no-deps", "llama-index-llms-openvino>=0.3.1", "llama-index-embeddings-openvino>=0.2.1", "llama-index-postprocessor-openvino-rerank>=0.2.0")
