from __future__ import annotations

import time
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from drama3_mlx.chunking import Drama3History, split_drama3_text
from drama3_mlx.prompt import build_drama3_prompt

if TYPE_CHECKING:
    from drama3_mlx.model import Drama3Model

DRAMA3_PREFILL_CHUNK_SIZE = 128
DEFAULT_MAX_NEW_TOKENS = 2048


def generate_drama3(
    model: Drama3Model,
    prompt: mx.array,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.95,
    prefill_chunk_size: int = DRAMA3_PREFILL_CHUNK_SIZE,
) -> Tuple[mx.array, str]:
    """Sample audio code frames after prompt rows.
    Returns (generated_codes [num_codebooks, num_frames], finish_reason).
    """
    backbone = model.backbone
    tower = model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens

    # Allocate caches
    backbone.allocate_caches(batch_size=1, max_seq_len=capacity)
    tower.setup_caches(max_seq_len=capacity)

    frames = mx.zeros((capacity, prompt.shape[1]), dtype=mx.int32)
    frames[:prompt_rows] = prompt
    positions = mx.arange(capacity)

    # 1. Prefill
    hidden = None
    for start in range(0, prompt_rows, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prompt_rows)
        chunk = frames[start:end]
        text_emb = backbone.embed(chunk[:, 0])
        embeddings = tower.embed_backbone(chunk, text_emb)

        hidden = backbone(
            inputs_embeds=embeddings[None],
            input_pos=positions[start:end],
            use_cache=True,
            continue_state=start > 0,
        )[0]

        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(
                hidden[: paired_end - start],
                positions[start:paired_end],
                frames[start:paired_end],
                frames[start + 1 : paired_end + 1],
            )

    hidden = hidden[-1:]  # [1, hidden_dim]
    produced = 0
    finish_reason = "length"

    # 2. Generation Loop
    for step in range(max_new_tokens):
        pos = prompt_rows - 1 + step
        logits = backbone.compute_logits(hidden)

        sampled = tower.decode_step(
            logits,
            hidden,
            positions[pos : pos + 1],
            frames[pos : pos + 1],
            frames[: pos + 1, 0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        main_token = int(sampled[0, 0].item())
        if main_token == model.config.eos_token_id:
            finish_reason = "eos"
            break

        frames[pos + 1] = sampled[0]
        produced += 1

        if step + 1 < max_new_tokens:
            text_emb = backbone.embed(sampled[:, 0])
            embeddings = tower.embed_backbone(sampled, text_emb)
            hidden = backbone(
                inputs_embeds=embeddings[None],
                input_pos=positions[pos + 1 : pos + 2],
                use_cache=True,
                continue_state=True,
            )[0]

    # Return codes: [num_codebooks, num_frames]
    gen_frames = frames[prompt_rows : prompt_rows + produced]
    codes = gen_frames[:, 1:].T  # [10, produced]
    return codes, finish_reason


def generate_drama3_stream(
    model: Drama3Model,
    prompt: mx.array,
    *,
    chunk_frames: int = 10,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.95,
) -> Generator[np.ndarray, None, None]:
    """Stream audio chunks as code frames are generated."""
    backbone = model.backbone
    tower = model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens

    backbone.allocate_caches(batch_size=1, max_seq_len=capacity)
    tower.setup_caches(max_seq_len=capacity)

    frames = mx.zeros((capacity, prompt.shape[1]), dtype=mx.int32)
    frames[:prompt_rows] = prompt
    positions = mx.arange(capacity)

    # Prefill
    hidden = None
    for start in range(0, prompt_rows, DRAMA3_PREFILL_CHUNK_SIZE):
        end = min(start + DRAMA3_PREFILL_CHUNK_SIZE, prompt_rows)
        chunk = frames[start:end]
        text_emb = backbone.embed(chunk[:, 0])
        embeddings = tower.embed_backbone(chunk, text_emb)
        hidden = backbone(
            inputs_embeds=embeddings[None],
            input_pos=positions[start:end],
            use_cache=True,
            continue_state=start > 0,
        )[0]
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(
                hidden[: paired_end - start],
                positions[start:paired_end],
                frames[start:paired_end],
                frames[start + 1 : paired_end + 1],
            )

    hidden = hidden[-1:]
    buffer_frames = []

    for step in range(max_new_tokens):
        pos = prompt_rows - 1 + step
        logits = backbone.compute_logits(hidden)
        sampled = tower.decode_step(
            logits,
            hidden,
            positions[pos : pos + 1],
            frames[pos : pos + 1],
            frames[: pos + 1, 0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        main_token = int(sampled[0, 0].item())
        if main_token == model.config.eos_token_id:
            break

        frames[pos + 1] = sampled[0]
        buffer_frames.append(sampled[0, 1:])

        if len(buffer_frames) >= chunk_frames:
            chunk_codes = mx.stack(buffer_frames, axis=0).T  # [10, chunk_frames]
            audio_chunk = model.decode_codes_to_audio(chunk_codes)
            yield audio_chunk
            buffer_frames = []

        if step + 1 < max_new_tokens:
            text_emb = backbone.embed(sampled[:, 0])
            embeddings = tower.embed_backbone(sampled, text_emb)
            hidden = backbone(
                inputs_embeds=embeddings[None],
                input_pos=positions[pos + 1 : pos + 2],
                use_cache=True,
                continue_state=True,
            )[0]

    # Flush remainder
    if buffer_frames:
        chunk_codes = mx.stack(buffer_frames, axis=0).T
        audio_chunk = model.decode_codes_to_audio(chunk_codes)
        yield audio_chunk


def generate_long_speech(
    model: Drama3Model,
    text: str,
    *,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[mx.array] = (),
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.95,
) -> np.ndarray:
    """Split long text into sentence/speaker chunks and synthesize sequentially."""
    chunks = split_drama3_text(text)
    history = Drama3History()
    audio_pieces = []

    for chunk in chunks:
        # Candidate histories (pinned first chunk + recent chunks)
        chosen_history = []
        for cand in history.candidates():
            chosen_history = cand
            break

        prompt = build_drama3_prompt(
            text=chunk,
            tokenizer=model.tokenizer,
            config=model.config,
            prompt_texts=prompt_texts,
            prompt_tokens=prompt_tokens,
            history=chosen_history,
        )

        codes, _ = generate_drama3(
            model,
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        history.push(chunk, codes)
        audio = model.decode_codes_to_audio(codes)
        audio_pieces.append(audio)

    if not audio_pieces:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(audio_pieces, axis=-1)
