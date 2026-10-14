from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
from tokenizers import Tokenizer

from drama3_mlx.config import Drama3Config

DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"

DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)

SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER_ID = 15

UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)

DRAMA3_MAX_PROMPT_ROWS = 8192


class InvalidInputError(ValueError):
    pass


class Drama3PromptTooLongError(ValueError):
    pass


def validate_drama3_text(text: str, location: str) -> None:
    if not text.strip(UNICODE_WHITE_SPACE):
        raise InvalidInputError(f"{location}: text is empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise InvalidInputError(f"{location}: contains structural token {token}")
    for match in SPEAKER_TAG_PATTERN.finditer(text):
        speaker_id = int(match.group(1))
        if speaker_id > MAX_SPEAKER_ID:
            raise InvalidInputError(
                f"{location}: speaker id {speaker_id} exceeds maximum {MAX_SPEAKER_ID}"
            )


def with_speaker_tag(text: str) -> str:
    if SPEAKER_TAG_PATTERN.search(text) is None:
        leading = len(text) - len(text.lstrip(UNICODE_WHITE_SPACE))
        return text[:leading] + "<|speaker:0|> " + text[leading:]
    return text


def build_drama3_prompt(
    text: str,
    tokenizer: Any,
    config: Drama3Config,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[mx.array] = (),
    history: Sequence[Tuple[str, mx.array]] = (),
) -> mx.array:
    """Build prompt rows of shape [rows, 1 + num_codebooks].
    Text rows are [token_id, 0, ..., 0].
    Audio rows are [c0 + semantic_start_token_id, c0, c1, ..., c9].
    """
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size

    if len(prompt_texts) != len(prompt_tokens):
        raise InvalidInputError("every reference needs both a transcript and codes")

    validate_drama3_text(text, "text")

    def check_codes(codes: mx.array, location: str) -> None:
        if (
            codes.ndim != 2
            or codes.shape[0] != num_codebooks
            or codes.shape[1] == 0
        ):
            raise InvalidInputError(
                f"{location}: codes must be [{num_codebooks}, frames >= 1]; got {tuple(codes.shape)}"
            )

    segments: List[Union[str, mx.array]] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if len(prompt_tokens) > 0 else DRAMA3_TTS_INSTRUCTION,
    ]

    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        location = f"reference {index}"
        validate_drama3_text(transcript, location)
        check_codes(codes, location)

        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]

    segments.append(DRAMA3_MESSAGE_END)

    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_drama3_text(turn_text, location)
        check_codes(codes, location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            codes,
            DRAMA3_MESSAGE_END,
        ]

    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            encoded = tokenizer.encode(segment)
            ids = encoded.ids if hasattr(encoded, "ids") else encoded
            ids_arr = mx.array(ids, dtype=mx.int32)
            text_rows = mx.zeros((len(ids_arr), 1 + num_codebooks), dtype=mx.int32)
            text_rows[:, 0] = ids_arr
            rows.append(text_rows)
        else:
            # codes shape: [num_codebooks, frames]
            c0 = segment[0:1]
            main = c0 + config.semantic_start_token_id
            stacked = mx.concatenate([main, segment], axis=0)  # [1 + num_codebooks, frames]
            audio_rows = stacked.T.astype(mx.int32)  # [frames, 1 + num_codebooks]
            rows.append(audio_rows)

    all_rows = mx.concatenate(rows, axis=0)
    if len(all_rows) > DRAMA3_MAX_PROMPT_ROWS:
        raise Drama3PromptTooLongError(
            f"prompt has {len(all_rows)} rows; maximum is {DRAMA3_MAX_PROMPT_ROWS}"
        )
    return all_rows
