from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://youtu.be/LPZh9BOjkQs"
chunks = process_input(source)
print(transcribe_all(chunks))