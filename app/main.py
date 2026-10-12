import json, os, threading, traceback, uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
#from faster_whisper import WhisperModel
from pydantic import BaseModel
from app.translation import translate_text, translate_batch

class TranslationRequest(BaseModel):
    text: str
    source: str = "auto"
    target: str = "hi"

BASE = Path(__file__).resolve().parent.parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))

JOBS: dict[str, dict] = {}
MODEL = None
MODEL_LOCK = threading.Lock()

app = FastAPI(title="Speech Timeline API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".mov", ".aac", ".flac", ".mkv"}


def get_model():
    raise RuntimeError(
        "Transcription is disabled during Vercel API testing."
    )


def fmt(sec: float, srt=False):
    ms = int(round((sec - int(sec)) * 1000))
    total = int(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(job_id: str, path: Path):
    try:
        JOBS[job_id].update(status="processing", progress=5, message="Loading Whisper model...")
        model = get_model()
        JOBS[job_id].update(progress=10, message="Transcribing audio...")
        segments, info = model.transcribe(str(path), beam_size=5, vad_filter=True, word_timestamps=False)
        result = []
        count = 0
        for s in segments:
            text = s.text.strip()
            if not text:
                continue
            result.append({"id": count, "start": round(float(s.start), 2), "end": round(float(s.end), 2), "text": text})
            count += 1
            # Whisper's iterator doesn't expose deterministic percent; keep UI moving.
            JOBS[job_id].update(progress=min(95, 10 + count), message=f"Transcribed {count} segments...")
        payload = {"language": info.language, "language_probability": round(float(info.language_probability), 4),
                   "duration": round(float(info.duration or 0), 2), "segments": result}
        (UPLOADS / f"{job_id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        JOBS[job_id].update(status="completed", progress=100, message="Transcription completed.", **payload)
    except Exception as e:
        traceback.print_exc()
        JOBS[job_id].update(status="failed", progress=0, message="Transcription failed.", error=str(e))


@app.get("/")
def root():
    return {"name": "Speech Timeline API", "status": "ok", "model": MODEL_SIZE}


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE, "compute": COMPUTE}

@app.post("/api/translate")
async def translate(request: TranslationRequest):
    try:
        if not request.text.strip():
            raise HTTPException(400, "Text cannot be empty")

        translated = translate_text(
            text=request.text,
            source=request.source,
            target=request.target
        )

        return {
            "success": True,
            "source": request.source,
            "target": request.target,
            "original": request.text,
            "translated": translated
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

class BatchTranslationRequest(BaseModel):
    texts: list[str]
    source: str = "auto"
    target: str = "hi"


@app.post("/api/translate/batch")
async def translate_batch_api(request: BatchTranslationRequest):
    try:
        if not request.texts:
            raise HTTPException(
                status_code=400,
                detail="Texts cannot be empty"
            )

        # Remove empty strings
        texts = [
            text.strip()
            for text in request.texts
            if text and text.strip()
        ]

        if not texts:
            raise HTTPException(
                status_code=400,
                detail="No valid text provided"
            )

        translations = translate_batch(
            texts=texts,
            source=request.source,
            target=request.target
        )

        return {
            "success": True,
            "source": request.source,
            "target": request.target,
            "count": len(texts),
            "translations": translations
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Batch translation failed: {str(e)}"
        )

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}")
    job_id = uuid.uuid4().hex
    path = UPLOADS / f"{job_id}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB")
                out.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    JOBS[job_id] = {"id": job_id, "filename": file.filename, "status": "queued", "progress": 0,
                    "message": "Waiting for transcription worker...", "segments": []}
    threading.Thread(target=transcribe, args=(job_id, path), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id in JOBS:
        return JOBS[job_id]
    saved = UPLOADS / f"{job_id}.json"
    if saved.exists():
        return {"id": job_id, "status": "completed", "progress": 100, **json.loads(saved.read_text(encoding="utf-8"))}
    raise HTTPException(404, "Job not found")


@app.get("/api/jobs/{job_id}/export/{fmt}")
def export_job(job_id: str, fmt: str):
    if job_id not in JOBS or JOBS[job_id].get("status") != "completed":
        raise HTTPException(400, "Transcription is not completed")
    segs = JOBS[job_id]["segments"]
    if fmt == "txt":
        content = "\n\n".join(f"[{fmt_time(s['start'])}] {s['text']}" for s in segs) + "\n"
    elif fmt == "srt":
        content = "\n\n".join(f"{i+1}\n{fmt_time(s['start'], True)} --> {fmt_time(s['end'], True)}\n{s['text']}" for i, s in enumerate(segs)) + "\n"
    elif fmt == "vtt":
        content = "WEBVTT\n\n" + "\n\n".join(f"{fmt_vtt(s['start'])} --> {fmt_vtt(s['end'])}\n{s['text']}" for s in segs) + "\n"
    else:
        raise HTTPException(400, "Format must be txt, srt or vtt")
    out = UPLOADS / f"{job_id}.{fmt}"
    out.write_text(content, encoding="utf-8")
    return FileResponse(out, media_type="text/plain", filename=f"transcript.{fmt}")


def fmt_time(sec, subtitle=False):
    return fmt(float(sec), subtitle)


def fmt_vtt(sec):
    return fmt(float(sec), False) + ".000"
