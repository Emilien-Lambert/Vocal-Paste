import os
import time
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import pyperclip
from pynput import keyboard

from src import config
from src.utils import log, paste_text_to_system
from src.audio import AudioRecorder
from src.inference import InferenceService

AUDIOS_DIR = os.path.join(config.BASE_DIR, "audios")
TRANSCRIPT_FILE = os.path.join(config.BASE_DIR, "transcript.txt")
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".wma"}
CHUNK_SIZE = 4096


def find_audio_file():
    """Find the first audio file in the audios/ directory."""
    if not os.path.isdir(AUDIOS_DIR):
        return None
    for name in os.listdir(AUDIOS_DIR):
        if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
            return os.path.join(AUDIOS_DIR, name)
    return None


def transcribe_file(service, filepath):
    """Transcribe an audio file and save the result to transcript.txt."""
    log(f"Transcribing: {os.path.basename(filepath)}")
    log("Please wait — do not start a voice recording until transcription is complete.")

    # Convert to 16kHz mono wav via ffmpeg for broad format support
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ar", str(config.SAMPLE_RATE),
             "-ac", "1", "-f", "wav", tmp_path],
            check=True, capture_output=True,
        )
        data, _ = sf.read(tmp_path, dtype="float32", always_2d=True)
        audio = data[:, 0]
    finally:
        os.unlink(tmp_path)

    service.start_streaming()
    for i in range(0, len(audio), CHUNK_SIZE):
        service.send_chunk(audio[i:i + CHUNK_SIZE])
    text = service.stop_streaming()

    if text and not text.startswith("ERROR:"):
        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        log(f"Transcription saved to transcript.txt")
        print(f"\n> {text}")
    else:
        log(f"Transcription failed: {text}")

    return text


def main():
    if not os.path.exists(config.MODELS_DIR) or not os.listdir(config.MODELS_DIR):
        log("Downloading model...")
        from src.weights import download_model
        download_model(config.MODEL_ID, local_dir=config.MODELS_DIR)

    recorder = AudioRecorder()
    service = InferenceService()
    service.initialize()

    # Check for audio files to transcribe at startup
    audio_file = find_audio_file()
    if audio_file:
        transcribe_file(service, audio_file)

    def start_recording():
        service.start_streaming()
        recorder.start(on_chunk=service.send_chunk)
        start_recording.time = time.time()

    def stop_recording():
        duration = time.time() - start_recording.time
        recorder.stop()
        text = service.stop_streaming()
        if text and not text.startswith("ERROR:"):
            pyperclip.copy(text)
            print(f"\n> {text}")
            log(f"Recording duration: {duration:.2f}s", verbose_only=True)
            paste_text_to_system()
        else:
            print(text)

    def on_press(key):
        if key == keyboard.Key.cmd_r:
            if config.HOLD_TO_TALK:
                start_recording()
            else:
                if recorder.is_recording:
                    stop_recording()
                else:
                    start_recording()

    def on_release(key):
        if key == keyboard.Key.cmd_r and config.HOLD_TO_TALK:
            if recorder.is_recording:
                stop_recording()

    if config.VERBOSE:
        print("Ready")
        print(f"HOLD_TO_TALK={str(config.HOLD_TO_TALK).lower()}")
        print(f"AUTO_PASTE={str(config.AUTO_PASTE).lower()}")
    else:
        print("ready")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()
        log("Goodbye.", verbose_only=True)

if __name__ == "__main__":
    main()
