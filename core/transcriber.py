import whisper
import os
from pydub import AudioSegment

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """English audio -> English text."""
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def transcribe_chunk_whisper_translate(chunk_path: str) -> str:
    """
    Hindi / Hinglish / mixed audio -> English text.
    Whisper's task="translate" always outputs English regardless of the
    spoken language, so this replaces the Sarvam API call entirely.
    """
    model = load_model()
    result = model.transcribe(chunk_path, task="translate", language="hi")
    return result["text"]


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_whisper_translate(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    engine = "Whisper (translate)" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "

    print("Transcription complete.")
    return full_transcript.strip()