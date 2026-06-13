# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A single-script utility (`convert_to_ollama.py`) that automates the full HF → Ollama pipeline:

1. Clone and build `llama.cpp` (if not present)
2. Download a safetensors snapshot from Hugging Face
3. Convert to an intermediate GGUF via `convert_hf_to_gguf.py`
4. Quantize with `llama-quantize`
5. Write a Modelfile and run `ollama create`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

System requirements: Python 3.10+, Git, CMake + C/C++ compiler, Ollama (if registering models).

For gated HF models: `export HF_TOKEN=hf_...`

## Running

```bash
# Single model — validate the pipeline first with a small model
python convert_to_ollama.py Qwen/Qwen2.5-0.5B-Instruct --quant Q4_K_M --smoke-prompt "Say hello."

# Batch (multiple model IDs) — --outfile/--quantized-outfile/--ollama-name are ignored in batch mode
python convert_to_ollama.py aisingapore/Qwen-SEA-LION-v4.5-27B-IT aisingapore/Gemma-SEA-LION-v4.5-E2B --quant Q4_K_M

# Skip Ollama registration (just produce the GGUF files)
python convert_to_ollama.py some-org/some-model --quant Q5_K_M --no-ollama-create

# Use an existing llama.cpp checkout
python convert_to_ollama.py some-org/some-model --llama-cpp-dir ~/src/llama.cpp --skip-llama-build

# Pass extra flags to convert_hf_to_gguf.py
python convert_to_ollama.py some-org/some-model --convert-arg=--verbose
```

The orchestrator script `run_conversion.sh` runs a smoke-test on Qwen 0.5B then converts the two SEA-LION v4.5 models.

## Key Design Decisions

- **Shells out to llama.cpp intentionally.** `llama.cpp` is the upstream source of truth for GGUF compatibility; the script never reimplements conversion logic.
- **`--outtype f16` intermediate by default.** Produces a full-precision GGUF first, then quantizes. Change with `--outtype`.
- **HF fallback.** If a model ID fails to download and doesn't already end in `-it`, the script automatically retries with `-IT` appended.
- **macOS Metal.** `cmake` is invoked with `-DGGML_METAL=ON` automatically on Darwin.
- **`convert_hf_to_gguf.py` detection.** Looks for the new name first (`convert_hf_to_gguf.py`), falls back to the old `convert.py` — handles both llama.cpp eras.
- **`llama-quantize` binary search.** Looks in `build/bin/`, `build/`, and the root of the llama.cpp dir in that order.

## Output Layout

```
downloads/<model-slug>/     # HF snapshot (gitignored)
llama.cpp/                  # cloned + built (gitignored)
models/
    <model>.f16.gguf        # intermediate (gitignored)
    <model>.q4_k_m.gguf     # quantized (gitignored)
    Modelfile.<ollama-name> # written by the script (gitignored)
```
