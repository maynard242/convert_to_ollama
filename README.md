# convert_to_ollama

Small utility repo to turn a Hugging Face safetensors checkpoint into a quantized GGUF model and register it with Ollama.

The script uses:

- `huggingface_hub` to download only safetensors/model metadata from HF
- `llama.cpp` to convert HF -> GGUF and quantize
- `ollama create` with a generated Modelfile to make the model runnable through Ollama

## Requirements

System tools:

- Python 3.10+
- Git
- CMake + a C/C++ compiler
- Ollama CLI/app if you want the script to create and run the model

Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need gated/private Hugging Face models, set a token:

```bash
export HF_TOKEN=hf_...
```

## Quick start

Use a small safetensors model first to validate the pipeline:

```bash
python convert_to_ollama.py Qwen/Qwen2.5-0.5B-Instruct \
  --quant Q4_K_M \
  --ollama-name qwen2.5-0.5b-q4 \
  --smoke-prompt "Say hello in one short sentence."
```

Then run it:

```bash
ollama run qwen2.5-0.5b-q4
```

## What the script does

1. Clones `ggerganov/llama.cpp` into `./llama.cpp` if it is not already present.
2. Builds the `llama-quantize` binary with CMake.
3. Downloads the requested HF model snapshot into `./downloads/<model>/`, restricted to `.safetensors` and tokenizer/config files.
4. Runs `llama.cpp/convert_hf_to_gguf.py` to produce an intermediate GGUF file in `./models/`.
5. Runs `llama-quantize` to create a quantized GGUF file.
6. Writes `./models/Modelfile.<ollama-name>`.
7. Runs `ollama create <ollama-name> -f <modelfile>` unless `--no-ollama-create` is set.

## Common examples

Create a Q5 quantized model but skip Ollama registration:

```bash
python convert_to_ollama.py Qwen/Qwen2.5-1.5B-Instruct \
  --quant Q5_K_M \
  --no-ollama-create
```

Use an existing llama.cpp checkout:

```bash
python convert_to_ollama.py meta-llama/Llama-3.2-1B-Instruct \
  --llama-cpp-dir ~/src/llama.cpp \
  --ollama-name llama3.2-1b-q4
```

Pass extra converter flags through to llama.cpp:

```bash
python convert_to_ollama.py some-org/some-model \
  --convert-arg=--verbose \
  --convert-arg=--model-name \
  --convert-arg=some-model
```

## Output layout

```text
.
├── convert_to_ollama.py
├── requirements.txt
├── downloads/              # ignored by git
├── llama.cpp/              # ignored by git
└── models/                 # ignored by git
    ├── <model>.f16.gguf
    ├── <model>.q4_k_m.gguf
    └── Modelfile.<name>
```

## Notes

- This script deliberately refuses to proceed if the HF snapshot does not contain `.safetensors` files.
- GGUF conversion support depends on upstream `llama.cpp`; if a model architecture is not supported there, conversion will fail with the upstream error.
- Larger models require substantial disk/RAM during conversion. Start with a small model to verify the toolchain.
- On macOS, the script enables llama.cpp Metal support during CMake configuration.
