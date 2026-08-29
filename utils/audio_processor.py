import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Optional: export cookies from a logged-in YouTube session in your browser
# (e.g. the "Get cookies.txt LOCALLY" extension) and set this env var to the
# path of that file. Authenticated cookie requests are far less likely to be
# blocked with a 403 than anonymous requests from a datacenter IP.
YOUTUBE_COOKIES_FILE = os.getenv("YOUTUBE_COOKIES_FILE")


class YouTubeDownloadBlockedError(RuntimeError):
    """Raised when YouTube refuses the download (e.g. 403), as opposed to
    some other yt-dlp failure. Lets the UI show a clear, actionable message
    instead of a raw traceback."""


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }
        ],
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["default", "-android_sdkless"],
            }
        },
    }

    if YOUTUBE_COOKIES_FILE and os.path.exists(YOUTUBE_COOKIES_FILE):
        ydl_opts["cookiefile"] = YOUTUBE_COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
        return filename
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            raise YouTubeDownloadBlockedError(
                "YouTube blocked this download (HTTP 403). This usually happens "
                "when requests come from a cloud/datacenter IP without a valid "
                "PO token or browser cookies. Try uploading the file directly "
                "instead, or configure YOUTUBE_COOKIES_FILE with a cookies.txt "
                "exported from a logged-in browser session."
            ) from e
        raise

def convert_to_wav(input_path:str)->str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path
    

def chunk_audio(wav_path:str, chunk_minutes:int = 10)-> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i,start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start:start+chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks
        

def process_input(source: str)->str:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected Youtube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
    
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio Ready - {len(chunks)} chunks created.")
    return chunks