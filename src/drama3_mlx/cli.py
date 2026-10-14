from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf

from drama3_mlx.convert import convert_drama3_model
from drama3_mlx.model import Drama3Model


def cmd_tts(args: argparse.Namespace) -> None:
    print(f"Loading Drama 3 MLX model from {args.model}...")
    model = Drama3Model.from_pretrained(args.model)

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")

    print(f"Synthesizing text: {text[:60]}...")
    started = time.perf_counter()
    audio = model.tts(
        text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    duration = len(audio) / model.config.codec_sample_rate
    elapsed = time.perf_counter() - started
    rtf = elapsed / max(duration, 0.001)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, model.config.codec_sample_rate)
    print(f"Saved audio to {out_path} ({duration:.2f}s, elapsed: {elapsed:.2f}s, RTF: {rtf:.3f})")


def cmd_convert(args: argparse.Namespace) -> None:
    convert_drama3_model(args.checkpoint, args.output, dtype=args.dtype)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from drama3_mlx.server import app, set_server_model

    print(f"Loading Drama 3 model for server from {args.model}...")
    model = Drama3Model.from_pretrained(args.model)
    set_server_model(model)

    print(f"Starting Drama 3 MLX server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="drama3-mlx", description="Fish Audio Drama 3 pure MLX runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. tts / synthesize
    tts_parser = subparsers.add_parser("tts", help="Synthesize speech from text")
    tts_parser.add_argument("--model", "-m", default="models/drama-3-mlx", help="Path to MLX model directory")
    tts_parser.add_argument("--text", "-t", help="Text to synthesize")
    tts_parser.add_argument("--file", "-f", help="Text file to synthesize")
    tts_parser.add_argument("--ref-audio", "-r", help="Reference audio for voice cloning")
    tts_parser.add_argument("--ref-text", help="Exact transcript of the reference audio")
    tts_parser.add_argument("--output", "-o", default="outputs/output.wav", help="Output WAV path")
    tts_parser.add_argument("--temperature", type=float, default=1.0)
    tts_parser.add_argument("--top-k", type=int, default=30)
    tts_parser.add_argument("--top-p", type=float, default=0.95)
    tts_parser.set_defaults(func=cmd_tts)

    # 2. convert
    conv_parser = subparsers.add_parser("convert", help="Convert HuggingFace Drama 3 weights to MLX")
    conv_parser.add_argument("--checkpoint", "-c", required=True, help="Input directory containing safetensors & codec.pth")
    conv_parser.add_argument("--output", "-o", required=True, help="Output directory for MLX model")
    conv_parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    conv_parser.set_defaults(func=cmd_convert)

    # 3. serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI/Fish Audio compatible HTTP server")
    serve_parser.add_argument("--model", "-m", default="models/drama-3-mlx", help="Path to MLX model directory")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
