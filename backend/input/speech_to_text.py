import whisper
import uuid
from pathlib import Path

# Load model once
model = whisper.load_model("base", download_root=None)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)

def speech_to_text(audio_bytes: bytes) -> str:
    if not audio_bytes:
        raise ValueError("Empty audio data received")

    temp_path = TEMP_DIR / f"audio_{uuid.uuid4().hex}.wav"

    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        result = model.transcribe(
            str(temp_path),
            fp16=False
        )
        return result["text"].strip()

    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_path}: {e}")
