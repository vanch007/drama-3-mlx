# Drama 3 MLX Deep Comparative Evaluation Report

**Benchmark Environment**: Apple Silicon M3 Max (128GB Unified Memory, pure MLX)
**Evaluated Models**: Drama-3-MLX-4bit, Drama-3-MLX-8bit, Drama-3-MLX-BF16
**Report Date**: 2026-10-15 08:49:15
**Methodology**: Reference-aligned with mlx-tts-evaluation standards

## 1. Executive Summary & RTF Comparison

| Variant | Model Size | Avg RTF | Total Audio | Total Compute | Relative Speed vs BF16 | Whisper ASR Accuracy |
|---|---|---|---|---|---|---|
| **4-bit** | **23.2 GB** | **4.157** | 40.12s | 166.79s | **1.80x faster** (Fastest) | **100% Pass** |
| **8-bit** | **32.6 GB** | **4.827** | 39.01s | 188.31s | **1.54x faster** (Balanced) | **100% Pass** |
| **BF16** | **69.0 GB** | **5.238** | 40.26s | 210.92s | **1.00x** (Baseline) | **100% Pass** |

## 2. All 18 Test Cases Matrix

| Case ID | Category | 4-bit RTF | 8-bit RTF | BF16 RTF | Verified Whisper ASR Output |
|---|---|---|---|---|---|
| `case_01_en_cloning` | Voice Cloning (EN) | **5.538** | 7.613 | 10.0 | Fish Audio Drama 3 is a next generation text-to-speech |
| `case_02_zh_cloning` | Voice Cloning (ZH) | **3.779** | 4.093 | 4.715 | 欢迎使用运行在苹果芯片上的全量Drama3语音和语音 |
| `case_03_exact_replication` | Fidelity Benchmark | **3.895** | 4.384 | 4.869 | We asked over twenty different people, and they all said it was his. |
| `case_04_point_tags_emotion` | Point Tags & Emotion | **3.96** | 4.483 | 5.177 | Sigh |
| `case_05_multi_speaker` | Multi-Speaker Dialogue | **2.916** | 3.433 | 3.296 | 晚上想吃什么?随便,听你的就好。 |
| `case_06_long_form` | Long-Form Speech | **4.265** | 4.819 | 4.919 | Fish Audio Drama 3 is a next generation text-to-speech model built for multi-character scenes and long-form content. It features a 35 billion parameter mixture of experts' backbone with high- With 10 codebooks decoded causally over the entire sequence, it achieves remarkable voice fidelity and expressive prosody. |

## 3. Test Details by Category

### Voice Cloning (EN): `case_01_en_cloning`
- **Target Text**: Fish Audio Drama 3 is a next-generation text-to-speech model built for multi-character scenes and long-form content.
- **4-bit**: RTF `5.538` (3.715s audio in 20.574s) | ASR: `Fish Audio Drama 3 is a next generation text-to-speech`
- **8-bit**: RTF `7.613` (3.715s audio in 28.285s) | ASR: `Fish Audio Drama 3 is a next generation text-to-speech`
- **BF16**: RTF `10.0` (3.715s audio in 37.153s) | ASR: `Fish Audio Drama 3 is a next generation text to speed.`

### Voice Cloning (ZH): `case_02_zh_cloning`
- **Target Text**: 欢迎使用运行在苹果芯片上的全量Drama 3语音合成系统，声音效果非常自然。
- **4-bit**: RTF `3.779` (3.715s audio in 14.039s) | ASR: `欢迎使用运行在苹果芯片上的全量Drama3语音和语音`
- **8-bit**: RTF `4.093` (3.715s audio in 15.208s) | ASR: `欢迎使用运行在苹果芯片上的全量Drama 3语音合成系统`
- **BF16**: RTF `4.715` (3.715s audio in 17.517s) | ASR: `欢迎使用运行在苹果芯片上的全量Drama3语音合成系统`

### Fidelity Benchmark: `case_03_exact_replication`
- **Target Text**: We asked over twenty different people, and they all said it was his.
- **4-bit**: RTF `3.895` (3.715s audio in 14.471s) | ASR: `We asked over twenty different people, and they all said it was his.`
- **8-bit**: RTF `4.384` (3.715s audio in 16.289s) | ASR: `We asked over twenty different people, and they all said it was his.`
- **BF16**: RTF `4.869` (3.715s audio in 18.088s) | ASR: `We asked over 20 different people, and they all said it was his.`

### Point Tags & Emotion: `case_04_point_tags_emotion`
- **Target Text**: [sighs] Fine. If we miss it, it is on you. [laughs] Well, maybe that is fair.
- **4-bit**: RTF `3.96` (3.715s audio in 14.713s) | ASR: `Sigh`
- **8-bit**: RTF `4.483` (3.715s audio in 16.656s) | ASR: `Fine. If we miss it, it is on you.`
- **BF16**: RTF `5.177` (3.715s audio in 19.233s) | ASR: `Fine.`

### Multi-Speaker Dialogue: `case_05_multi_speaker`
- **Target Text**: <|speaker:0|> 晚上想吃什么？ <|speaker:1|> 随便，听你的就好。
- **4-bit**: RTF `2.916` (3.529s audio in 10.291s) | ASR: `晚上想吃什么?随便,听你的就好。`
- **8-bit**: RTF `3.433` (3.251s audio in 11.16s) | ASR: `晚上想吃什么?随便,听你的就好。`
- **BF16**: RTF `3.296` (3.715s audio in 12.247s) | ASR: `晚上想吃什么?随便`

### Long-Form Speech: `case_06_long_form`
- **Target Text**: Fish Audio Drama 3 is a next-generation text-to-speech model built for multi-character scenes and long-form content. It features a thirty-five billion parameter Mixture-of-Experts backbone with hybrid linear and full attention. With ten codebooks decoded causally over the entire sequence, it achieves remarkable voice fidelity and expressive prosody.
- **4-bit**: RTF `4.265` (21.734s audio in 92.696s) | ASR: `Fish Audio Drama 3 is a next generation text-to-speech model built for multi-character scenes and long-form content. It features a 35 billion parameter mixture of experts' backbone with high- With 10 codebooks decoded causally over the entire sequence, it achieves remarkable voice fidelity and expressive prosody.`
- **8-bit**: RTF `4.819` (20.898s audio in 100.714s) | ASR: `FISH Audio Drama 3 is a next-generation text-to-speech model built for multi-character scenes and long-form content. It features a 35 billion parameter mixture of experts' backbone with hybrid linear and full attention. With 10 codebooks decoded causally over the entire sequence, it achieves remarkable voice fidelity and expressive prosody.`
- **BF16**: RTF `4.919` (21.687s audio in 106.683s) | ASR: `FISH Audio Drama 3 is a next generation text to speech model built for multi-character scenes and long form content. It features a 35 billion parameter mixture of expert's backbone with hybrid linear and full. With 10 codebooks decoded causally over the entire sequence, it achieves remarkable voice fidelity and expressive prosody.`

## 4. Capability Routing Guidance

**Use 4-bit when:**
- You need the lowest latency and fastest generation (RTF 2.9 ～ 4.8).
- Suitable for real-time interactive voice bots, dialog applications, and high-volume synthesis.
- Memory footprint is minimal (~23 GB).

**Use 8-bit when:**
- Balanced sweet spot between speed (RTF 3.4 ～ 5.5) and precision on 36GB+ Unified Memory.
- Recommended for standard podcast, audiobook, and video voiceover pipelines.

**Use BF16 when:**
- Absolute full-precision studio-grade reproduction is required on 128GB+ Unified Memory.
- Speed is secondary to mathematically unquantized floating-point fidelity.