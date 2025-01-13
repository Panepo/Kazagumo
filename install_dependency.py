import sys

def pip_install(*args):
    import subprocess  # nosec - disable B404:import-subprocess check

    cli_args = []
    for arg in args:
        cli_args.extend(str(arg).split(" "))
    subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

pip_install("-Uq", "pip")
pip_install("gradio>=4.19", "python-dotenv", "transformers", "bitsandbytes", 'accelerate>=0.26.0')
pip_install("--index-url", "https://download.pytorch.org/whl/cu124", "torch", "torchvision", "torchaudio")
pip_install("llama_index", "llama-index-llms-ollama", "llama-index-embeddings-ollama")
