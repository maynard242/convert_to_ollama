#!/usr/bin/env python3
"""Download a safetensors Hugging Face model, convert it to GGUF, quantize it, and register it with Ollama.

This script intentionally shells out to llama.cpp for conversion/quantization because
llama.cpp is the upstream source of truth for GGUF compatibility.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


DEFAULT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "*.tiktoken",
    "*.model",
    "README.md",
]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    if cwd:
        print(f"$ cd {cwd} && {printable}")
    else:
        print(f"$ {printable}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def require_executable(name: str, hint: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found: {name}. {hint}")


def slugify_model_id(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_id).strip("-._")
    return slug.lower() or "model"


def find_convert_script(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "convert_hf_to_gguf.py",
        llama_cpp_dir / "convert.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find convert_hf_to_gguf.py in {llama_cpp_dir}. "
        "Check --llama-cpp-dir or update the llama.cpp checkout."
    )


def find_quantize_binary(llama_cpp_dir: Path) -> Path:
    names = ["llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"]
    roots = [
        llama_cpp_dir / "build" / "bin",
        llama_cpp_dir / "build",
        llama_cpp_dir,
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and os.access(candidate, os.X_OK):
                return candidate
    raise FileNotFoundError(
        f"Could not find llama.cpp quantize binary under {llama_cpp_dir}. "
        "Run without --skip-llama-build or build llama.cpp manually."
    )


def ensure_llama_cpp(llama_cpp_dir: Path, repo: str, skip_build: bool) -> None:
    if not llama_cpp_dir.exists():
        require_executable("git", "Install Git or pass --llama-cpp-dir pointing at an existing checkout.")
        run(["git", "clone", "--depth", "1", repo, str(llama_cpp_dir)])
    else:
        print(f"Using existing llama.cpp checkout: {llama_cpp_dir}")

    if skip_build:
        print("Skipping llama.cpp build (--skip-llama-build).")
        return

    require_executable("cmake", "Install CMake or rerun with --skip-llama-build after building llama.cpp yourself.")
    build_dir = llama_cpp_dir / "build"
    cmake_cmd = ["cmake", "-B", str(build_dir), "-DLLAMA_BUILD_TESTS=OFF"]
    if sys.platform == "darwin":
        cmake_cmd.append("-DGGML_METAL=ON")
    run(cmake_cmd, cwd=llama_cpp_dir)

    try:
        run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "-j",
                str(os.cpu_count() or 4),
                "--target",
                "llama-quantize",
            ],
            cwd=llama_cpp_dir,
        )
    except subprocess.CalledProcessError:
        print("Targeted llama-quantize build failed; building default target instead.")
        run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "-j",
                str(os.cpu_count() or 4),
            ],
            cwd=llama_cpp_dir,
        )


def download_model(model_id: str, revision: str | None, downloads_dir: Path, token: str | None) -> Path:
    local_dir = downloads_dir / slugify_model_id(model_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading safetensors snapshot for {model_id} -> {local_dir}")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
        token=token,
    )
    safetensors = sorted(local_dir.rglob("*.safetensors"))
    if not safetensors:
        raise RuntimeError(
            f"No .safetensors files were downloaded for {model_id}. "
            "This script only converts safetensors checkpoints; choose a safetensors model or revision."
        )
    print(f"Downloaded {len(safetensors)} safetensors file(s).")
    return local_dir


def convert_to_gguf(
    model_dir: Path,
    llama_cpp_dir: Path,
    outfile: Path,
    outtype: str,
    extra_args: Iterable[str],
) -> None:
    convert_script = find_convert_script(llama_cpp_dir)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    gguf_py = llama_cpp_dir / "gguf-py"
    env["PYTHONPATH"] = f"{gguf_py}{os.pathsep}{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
        *extra_args,
    ]
    run(cmd, cwd=llama_cpp_dir, env=env)
    if not outfile.exists():
        raise RuntimeError(f"Expected GGUF file was not created: {outfile}")


def quantize_gguf(llama_cpp_dir: Path, source_gguf: Path, quantized_gguf: Path, quant: str) -> None:
    quantized_gguf.parent.mkdir(parents=True, exist_ok=True)
    quantize = find_quantize_binary(llama_cpp_dir)
    run([str(quantize), str(source_gguf), str(quantized_gguf), quant], cwd=llama_cpp_dir)
    if not quantized_gguf.exists():
        raise RuntimeError(f"Expected quantized GGUF file was not created: {quantized_gguf}")


def write_modelfile(path: Path, gguf_path: Path, num_ctx: int | None) -> None:
    lines = [f"FROM {gguf_path.resolve()}\n"]
    if num_ctx:
        lines.append(f"PARAMETER num_ctx {num_ctx}\n")
    path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote Ollama Modelfile: {path}")


def create_ollama_model(name: str, modelfile: Path, smoke_prompt: str | None) -> None:
    if shutil.which("ollama") is None:
        print("Ollama CLI not found. Install Ollama, then run:")
        print(f"  ollama create {name} -f {modelfile}")
        print(f"  ollama run {name}")
        return

    try:
        run(["ollama", "create", name, "-f", str(modelfile)])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "ollama create failed. Make sure the Ollama app/server is running "
            "(`ollama serve` or the desktop app), then retry."
        ) from exc

    print(f"Created Ollama model: {name}")
    print(f"Run it with: ollama run {name}")
    if smoke_prompt:
        run(["ollama", "run", name, smoke_prompt])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HF safetensors model, convert to GGUF, quantize, and create an Ollama model."
    )
    parser.add_argument("model_id", help="Hugging Face model repo ID, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", help="HF branch/tag/commit to download")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory for GGUF outputs")
    parser.add_argument("--downloads-dir", type=Path, default=Path("downloads"), help="Directory for HF snapshot downloads")
    parser.add_argument("--llama-cpp-dir", type=Path, default=Path("llama.cpp"), help="llama.cpp checkout path")
    parser.add_argument("--llama-cpp-repo", default="https://github.com/ggerganov/llama.cpp.git", help="llama.cpp git repo URL")
    parser.add_argument("--skip-llama-build", action="store_true", help="Do not configure/build llama.cpp")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"], help="Intermediate GGUF output type")
    parser.add_argument("--quant", default="Q4_K_M", help="llama.cpp quantization type, e.g. Q4_K_M, Q5_K_M, Q8_0")
    parser.add_argument("--outfile", type=Path, help="Intermediate GGUF path; defaults to models/<model>.f16.gguf")
    parser.add_argument("--quantized-outfile", type=Path, help="Quantized GGUF path; defaults to models/<model>.<quant>.gguf")
    parser.add_argument("--ollama-name", help="Name for `ollama create`; defaults to <model>-<quant>")
    parser.add_argument("--num-ctx", type=int, help="Optional Ollama PARAMETER num_ctx value")
    parser.add_argument("--no-ollama-create", action="store_true", help="Write Modelfile but do not run `ollama create`")
    parser.add_argument("--smoke-prompt", help="Optional prompt to run after `ollama create`")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token; defaults to HF_TOKEN env var")
    parser.add_argument(
        "--convert-arg",
        action="append",
        default=[],
        help="Extra argument to pass to llama.cpp convert_hf_to_gguf.py. Repeat for multiple args.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slug = slugify_model_id(args.model_id)
    quant_slug = args.quant.lower()
    output_dir = args.output_dir.resolve()
    downloads_dir = args.downloads_dir.resolve()
    llama_cpp_dir = args.llama_cpp_dir.resolve()
    intermediate = (args.outfile or output_dir / f"{slug}.{args.outtype}.gguf").resolve()
    quantized = (args.quantized_outfile or output_dir / f"{slug}.{quant_slug}.gguf").resolve()
    ollama_name = args.ollama_name or f"{slug}-{quant_slug}"
    modelfile = output_dir / f"Modelfile.{ollama_name}"

    ensure_llama_cpp(llama_cpp_dir, args.llama_cpp_repo, args.skip_llama_build)
    model_dir = download_model(args.model_id, args.revision, downloads_dir, args.hf_token)
    convert_to_gguf(model_dir, llama_cpp_dir, intermediate, args.outtype, args.convert_arg)
    quantize_gguf(llama_cpp_dir, intermediate, quantized, args.quant)
    write_modelfile(modelfile, quantized, args.num_ctx)

    if args.no_ollama_create:
        print("Skipped `ollama create` (--no-ollama-create). Run manually:")
        print(f"  ollama create {ollama_name} -f {modelfile}")
        print(f"  ollama run {ollama_name}")
    else:
        create_ollama_model(ollama_name, modelfile, args.smoke_prompt)

    print("\nDone.")
    print(f"Safetensors download: {model_dir}")
    print(f"Intermediate GGUF:    {intermediate}")
    print(f"Quantized GGUF:       {quantized}")
    print(f"Ollama model:         {ollama_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
