import json
import os
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

from app.translation import translate_text, translate_batch
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.getenv("VERCEL"))
UPLOADS = Path("/tmp/uploads") if IS_VERCEL else BASE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
# Groq free-tier audio upload limit is documented as 25 MB; leave a little headroom.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "24"))
ALLOWED_EXT = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_CLIENT: Groq | None = None

app = FastAPI(title="Speech Timeline API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

class TranslationRequest(BaseModel):
    text: str
    source: str = "auto"
    target: str = "hi"

class BatchTranslationRequest(BaseModel):
    texts: list[str]
    source: str = "auto"
    target: str = "hi"


def get_client() -> Groq:
    global _CLIENT
    if not GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY is not configured on the backend.")
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:
                _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT

def validate_filename(filename: str | None) -> str:
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported audio/video format: {suffix or '(none)'}")
    return suffix


async def read_upload(upload: UploadFile) -> bytes:
    data = bytearray()
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB. Compress or split it and retry.")
    finally:
        await upload.close()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    return bytes(data)


def transcribe_bytes(audio: bytes, filename: str, language: str | None = None) -> dict[str, Any]:
    suffix = validate_filename(filename)
    kwargs: dict[str, Any] = {
        "file": (f"audio{suffix}", audio),
        "model": GROQ_STT_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
        "temperature": 0.0,
    }
    if language and language.lower() not in {"auto", "unknown"}:
        kwargs["language"] = language.lower()
    try:
        response = get_client().audio.transcriptions.create(**kwargs)
    except Exception as exc:
        traceback.print_exc()
        status = getattr(exc, "status_code", None)
        if status == 401:
            raise HTTPException(502, "Groq rejected the key; check GROQ_API_KEY.") from exc
        if status == 429:
            raise HTTPException(429, "Groq rate limit/quota reached. Check your Groq usage.") from exc
        if status == 413 or "too large" in str(exc).lower():
            raise HTTPException(413, "Audio is too large for the speech API; split or compress it.") from exc
        raise HTTPException(502, f"Speech provider failed: {exc}") from exc

    out_segments = []
    for seg in (getattr(response, "segments", None) or []):
        if isinstance(seg, dict):
            start, end, text = seg.get("start", 0), seg.get("end", 0), seg.get("text", "")
        else:
            start, end, text = getattr(seg, "start", 0), getattr(seg, "end", 0), getattr(seg, "text", "")
        text = (text or "").strip()
        if text:
            out_segments.append({"id": len(out_segments), "start": round(float(start or 0), 2),
                                 "end": round(float(end or 0), 2), "text": text})
    text = (getattr(response, "text", "") or "").strip()
    if not text and out_segments:
        text = " ".join(s["text"] for s in out_segments)
    duration = getattr(response, "duration", None)
    if duration is None and out_segments:
        duration = out_segments[-1]["end"]
    return {"text": text, "language": getattr(response, "language", None) or language or "unknown",
            "duration": round(float(duration or 0), 2), "language_probability": None,
            "segments": out_segments}


def fmt(seconds: float, srt: bool = False) -> str:
    ms_total = max(0, int(round(float(seconds) * 1000)))
    total, ms = divmod(ms_total, 1000)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{ms:03d}"

def execute_job(job_id: str, filename: str, audio: bytes) -> dict[str, Any]:
    JOBS[job_id].update(status="processing", progress=10, message="Transcribing with Groq...")
    result = transcribe_bytes(audio, filename)
    JOBS[job_id].update(status="completed", progress=100, message="Transcription completed.", **result)
    try:
        (UPLOADS / f"{job_id}.json").write_text(json.dumps(JOBS[job_id], ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return JOBS[job_id]

@app.get("/")
def root():
    return {"name": "Speech Timeline API", "status": "ok", "provider": "groq",
            "model": GROQ_STT_MODEL, "api_key_configured": bool(GROQ_API_KEY)}

@app.get("/api/health")
def health():
    return {"status": "ok", "provider": "groq", "model": GROQ_STT_MODEL,
            "api_key_configured": bool(GROQ_API_KEY), "max_upload_mb": MAX_UPLOAD_MB}

@app.post("/api/translate")
async def translate(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    try:
        translated = translate_text(text=request.text, source=request.source, target=request.target)
        return {"success": True, "source": request.source, "target": request.target,
                "original": request.text, "translated": translated}
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(502, f"Translation failed: {exc}") from exc

@app.post("/api/translate/batch")
async def translate_batch_api(request: BatchTranslationRequest):
    texts = [t.strip() for t in request.texts if t and t.strip()]
    if not texts:
        raise HTTPException(400, "Texts cannot be empty.")
    try:
        translations = translate_batch(texts=texts, source=request.source, target=request.target)
        return {"success": True, "source": request.source, "target": request.target,
                "count": len(texts), "translations": translations}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(502, f"Batch translation failed: {exc}") from exc

@app.post("/api/live/transcribe")
async def live_transcribe(file: UploadFile = File(...), target: str = Form("hi"), source: str = Form("auto")):
    validate_filename(file.filename)
    audio = await read_upload(file)
    result = transcribe_bytes(audio, file.filename or "chunk.wav", source)
    translated = ""
    translation_error = None
    if result["text"]:
        try:
            translated = translate_text(text=result["text"], source=result["language"], target=target)
        except Exception as exc:
            translation_error = str(exc)
    return {"success": True, "original": result["text"], "translated": translated,
            "source": result["language"], "target": target, "segments": result["segments"],
            "duration": result["duration"], "translation_error": translation_error}

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    validate_filename(file.filename)
    filename = file.filename or "audio.wav"
    audio = await read_upload(file)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"id": job_id, "filename": filename, "status": "queued", "progress": 0,
                    "message": "Queued for transcription.", "segments": []}
    if IS_VERCEL:
        # Avoid background threads on serverless. The result is returned immediately.
        try:
            return execute_job(job_id, filename, audio)
        except HTTPException as exc:
            JOBS[job_id].update(status="failed", error=exc.detail, message="Transcription failed.")
            raise
    def worker():
        try:
            execute_job(job_id, filename, audio)
        except Exception as exc:
            JOBS[job_id].update(status="failed", error=str(exc), message="Transcription failed.")
            traceback.print_exc()
    threading.Thread(target=worker, daemon=True).start()
    return JOBS[job_id]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id in JOBS:
        return JOBS[job_id]
    saved = UPLOADS / f"{job_id}.json"
    if saved.exists():
        try:
            return json.loads(saved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    raise HTTPException(404, "Job not found; serverless job state may not persist between requests.")

@app.get("/api/jobs/{job_id}/export/{fmt_name}")
def export_job(job_id: str, fmt_name: str):
    job = JOBS.get(job_id)
    if job is None:
        saved = UPLOADS / f"{job_id}.json"
        if saved.exists():
            try:
                job = json.loads(saved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                job = None
    if not job or job.get("status") != "completed":
        raise HTTPException(400, "Transcription is not completed or job is unavailable.")
    segments = job.get("segments", [])
    if fmt_name == "txt":
        content = "\n\n".join(f"[{fmt(s['start']).split('.')[0]}] {s['text']}" for s in segments) + "\n"
        media_type = "text/plain"
    elif fmt_name == "srt":
        content = "\n\n".join(f"{i+1}\n{fmt(s['start'], True)} --> {fmt(s['end'], True)}\n{s['text']}" for i, s in enumerate(segments)) + "\n"
        media_type = "application/x-subrip"
    elif fmt_name == "vtt":
        content = "WEBVTT\n\n" + "\n\n".join(f"{fmt(s['start'])} --> {fmt(s['end'])}\n{s['text']}" for s in segments) + "\n"
        media_type = "text/vtt"
    else:
        raise HTTPException(400, "Format must be txt, srt or vtt.")
    output = UPLOADS / f"{job_id}.{fmt_name}"
    try:
        output.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, "Cannot create export in this environment.") from exc
    return FileResponse(output, media_type=media_type, filename=f"transcript.{fmt_name}")
