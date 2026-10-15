# Drama 3 MLX

Standalone, pure [Apple Silicon MLX](https://github.com/ml-explore/mlx) implementation of [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3).

Drama 3 is Fish Audio's next-generation multi-speaker conversational text-to-speech model featuring a **35B parameter (3B active)** Mixture-of-Experts (MoE) text backbone with hybrid linear/full attention and a **10-codebook Codebook-MTP audio decoder**.

This repository provides an end-to-end, zero-PyTorch-at-runtime MLX port supporting all official features:
- **Zero-Shot Voice Cloning**: Clone any voice using reference audio and transcript.
- **Multi-Speaker Characters**: Supports tags `<|speaker:0|>` through `<|speaker:15|>` allowing up to 16 distinct characters in a single dialogue.
- **Point Tags & Emotion**: In-text paralinguistic and emotional cues like `[sighs]`, `[laughs]`, `[whispers]`.
- **Audio Editing & Continuation**: Provide `previous_takes` for context-aware take editing.
- **Automatic Long-Text Chunking**: Integrated `split_drama3_text` preserving speaker consistency across turns.
- **Streaming Audio Synthesis**: Real-time token-by-token and chunked audio streaming.
- **Modded DAC Neural Codec**: 44.1kHz 10-codebook pure MLX audio decoder.
- **HTTP Server**: Dual OpenAI (`/v1/audio/speech`) and Fish Audio (`/v1/tts`) compatible endpoints.

---

## 🎧 Live Demo & Benchmark Showcase

Experience the generated audio samples across 4-bit, 8-bit, and BF16 with zero-shot voice cloning, emotional point tags, and multi-speaker dialogues:
👉 **[Online Interactive Audio Demo & Benchmark Report](https://vanch007.github.io/drama-3-mlx/)**

---

## Architecture Overview

1. **Text Backbone (`FishOmniBackbone`)**:
   - 40 layers combining 30 GatedDeltaNet (Linear Attention) layers and 10 Full Softmax Attention layers.
   - 256 routed MoE experts (top-8 active per token) plus a shared expert with sigmoid gating.
   - ZeroCenteredRMSNorm (`1.0 + weight`) and partial RoPE.
2. **Audio Decoder Tower (`FishOmniAudioDecoder`)**:
   - 10 codebooks causally decoded over time.
   - 4 shared enrichment layers + 9 residual codebook layers.
   - Repetition-Aware Sampling (RAS) window with automatic retry.
3. **Audio Codec (`DAC`)**:
   - 44.1kHz sample rate, 10 codebooks (1 semantic + 9 acoustic).
   - WindowLimitedTransformer + Residual Vector Quantization (RVQ).

---

## Installation

```bash
cd /Users/vanch/drama-3-mlx
pip install -e .
```

To enable FastAPI server support:
```bash
pip install -e ".[server]"
```

---

## Converting Official Weights

To convert the official HuggingFace checkpoint (`fishaudio/drama-3`) to pure MLX format:

```bash
drama3-mlx convert \
  --checkpoint /path/to/fishaudio/drama-3 \
  --output models/drama-3-mlx \
  --dtype bfloat16
```

The conversion script automatically:
1. Dequantizes Block-FP8 MoE weights to target precision (bfloat16 or float16).
2. Stacks expert weights into fast unified expert tensors.
3. Transposes Conv1d weights for MLX layout.
4. Converts `codec.pth` into pure MLX `codec.safetensors`.

---

## CLI Usage

### 1. Plain Text-to-Speech
```bash
drama3-mlx tts \
  --model models/drama-3-mlx \
  --text "你好，欢迎使用 Fish Audio Drama 3 纯 MLX 版本！" \
  --output outputs/hello.wav
```

### 2. Zero-Shot Voice Cloning
```bash
drama3-mlx tts \
  --model models/drama-3-mlx \
  --ref-audio ref.wav \
  --ref-text "We asked over twenty different people, and they all said it was his." \
  --text "Get the trust fund to the bank early, before the storm arrives tonight." \
  --output outputs/clone.wav
```

### 3. Multi-Speaker Dialogue
```bash
drama3-mlx tts \
  --model models/drama-3-mlx \
  --text "<|speaker:0|> 晚上想吃什么？ <|speaker:1|> 随便，听你的就好。" \
  --output outputs/dialogue.wav
```

### 4. Paralinguistic Point Tags
```bash
drama3-mlx tts \
  --model models/drama-3-mlx \
  --text "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair." \
  --output outputs/tags.wav
```

---

## Python API

```python
from drama3_mlx.model import Drama3Model
import soundfile as sf

# Load model
model = Drama3Model.from_pretrained("models/drama-3-mlx")

# 1. Direct synthesis
audio = model.tts("这是一段纯 MLX 本地推理的测试语音。")
sf.write("output.wav", audio, model.config.codec_sample_rate)

# 2. Voice cloning
audio = model.tts(
    "Hello from Drama 3 on Apple Silicon!",
    ref_audio="ref.wav",
    ref_text="Reference speech transcript."
)

# 3. Real-time streaming
for chunk in model.tts_stream("实时音频流式生成示例。"):
    # chunk is a float32 numpy audio array
    pass
```

---

## HTTP Serving

Start the high-performance local server:

```bash
drama3-mlx serve --model models/drama-3-mlx --port 8000
```

### OpenAI-Compatible API
```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world from Drama 3 MLX!",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Official Fish Audio API
```bash
curl -X POST http://127.0.0.1:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Get the trust fund to the bank early.",
    "references": [
      {
        "audio_path": "ref.wav",
        "text": "Exact transcript of reference audio."
      }
    ]
  }' \
  --output speech.wav
```

---

## Tests and Verification

Run the test suite:
```bash
python -m unittest discover -s tests
```
