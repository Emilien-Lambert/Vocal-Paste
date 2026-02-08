import os
import pyperclip
from pynput import keyboard
from huggingface_hub import snapshot_download

import src.config as config
from src.utils import log, paste_text_to_system
from src.audio import AudioRecorder
from src.inference import InferenceService

def main():
    # 1. Setup Resources
    if not os.path.exists(config.MODELS_DIR) or not os.listdir(config.MODELS_DIR):
        log("Downloading model...")
        snapshot_download(repo_id=config.MODEL_ID, local_dir=config.MODELS_DIR, token=True)

    recorder = AudioRecorder()
    service = InferenceService()
    service.initialize()

    # 2. Define Interaction Logic
    def on_transcription_complete(wav_path):
        if not wav_path:
            return

        text = service.transcribe(wav_path)
        
        if text and not text.startswith("ERROR:"):
            pyperclip.copy(text)
            print(f"> {text}")
            paste_text_to_system()
        else:
            print(text)

    def on_press(key):
        if key == keyboard.Key.cmd_r:
            if config.HOLD_TO_TALK:
                recorder.start()
            else:
                # Toggle logic
                if recorder.is_recording:
                    wav_file = recorder.stop()
                    on_transcription_complete(wav_file)
                else:
                    recorder.start()

    def on_release(key):
        if key == keyboard.Key.cmd_r and config.HOLD_TO_TALK:
            if recorder.is_recording:
                wav_file = recorder.stop()
                on_transcription_complete(wav_file)

    # 3. Start UI
    if config.VERBOSE:
        print("Ready")
        print(f"MODEL_TIMEOUT={config.MODEL_TIMEOUT}")
        print(f"HOLD_TO_TALK={str(config.HOLD_TO_TALK).lower()}")
        print(f"AUTO_PASTE={str(config.AUTO_PASTE).lower()}")
    else:
        print("ready")

    # 4. Event Loop
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