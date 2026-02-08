import os
import time
import pyperclip
from pynput import keyboard

import src.config as config
from src.utils import log, paste_text_to_system
from src.audio import AudioRecorder
from src.inference import InferenceService


def main():
    if not os.path.exists(config.MODELS_DIR) or not os.listdir(config.MODELS_DIR):
        log("Downloading model...")
        from src.weights import download_model
        download_model(config.MODEL_ID)

    recorder = AudioRecorder()
    service = InferenceService()
    service.initialize()

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
