from __future__ import annotations

import io
from typing import Any, List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from drama3_mlx.model import Drama3Model

app = FastAPI(title="Drama 3 MLX Speech Server", version="0.1.0")
_model: Optional[Drama3Model] = None


def set_server_model(model: Drama3Model) -> None:
    global _model
    _model = model


def get_server_model() -> Drama3Model:
    if _model is None:
        raise HTTPException(status_code=503, detail="Drama 3 model not initialized")
    return _model


class ReferenceItem(BaseModel):
    audio_path: str
    text: str


class FishTTSRequest(BaseModel):
    text: str
    references: Optional[List[ReferenceItem]] = None
    previous_takes: Optional[List[str]] = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 30
    max_new_tokens: int = 2048
    format: str = "wav"


class OpenAISpeechRequest(BaseModel):
    input: str
    model: str = "drama-3-mlx"
    voice: Optional[str] = None
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "model": "drama-3-mlx"}


@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest):
    model = get_server_model()

    if req.stream:
        async def stream_generator():
            for chunk in model.tts_stream(req.input):
                # Convert float32 array to 16-bit PCM bytes
                pcm16 = (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                yield pcm16

        return StreamingResponse(stream_generator(), media_type="audio/pcm")

    # Non-streaming
    audio = model.tts(req.input)
    buf = io.BytesIO()
    sf.write(buf, audio, model.config.codec_sample_rate, format=req.response_format.upper())
    buf.seek(0)
    return Response(content=buf.read(), media_type=f"audio/{req.response_format}")


@app.post("/v1/tts")
async def fish_tts(req: FishTTSRequest):
    model = get_server_model()

    ref_audio = None
    ref_text = None
    if req.references and len(req.references) > 0:
        ref_audio = req.references[0].audio_path
        ref_text = req.references[0].text

    audio = model.tts(
        req.text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_new_tokens=req.max_new_tokens,
    )

    buf = io.BytesIO()
    fmt = req.format.upper()
    if fmt == "MP3":
        fmt = "WAV"  # fallback to WAV if mp3 writer not available
    sf.write(buf, audio, model.config.codec_sample_rate, format=fmt)
    buf.seek(0)
    return Response(content=buf.read(), media_type=f"audio/{req.format}")
