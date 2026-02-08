import os
import time
import multiprocessing
import threading
import queue
import pyperclip
import argparse
import subprocess
from pynput import keyboard
from huggingface_hub import snapshot_download
import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

# Parse arguments
parser = argparse.ArgumentParser(description="Vocal-Paste: Voice to Clipboard/Paste utility")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
args = parser.parse_args()

# Load configuration
load_dotenv()
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", 300))
HOLD_TO_TALK = os.getenv("HOLD_TO_TALK", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "false").lower() == "true"

# Disable progress bars for silent startup
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
LOCAL_DIR = os.path.join(os.getcwd(), "models")
WAVE_FILENAME = "temp_voice.wav"
SAMPLE_RATE = 48000

def log(message, verbose_only=False):
    """Helper for conditional logging"""
    if not verbose_only or args.verbose:
        print(message)

def paste_text():
    """Simulate Cmd+V to paste the clipboard content on macOS"""
    if not AUTO_PASTE:
        return
    try:
        script = 'tell application "System Events" to keystroke "v" using {command down}'
        subprocess.run(['osascript', '-e', script])
    except Exception as e:
        log(f"ERROR: Failed to auto-paste: {e}")

# --- Worker Process (Handles model and RAM) ---
def model_worker(input_queue, output_queue, model_path):
    """
    This code runs in a separate process to isolate RAM usage (4GB+).
    """
    try:
        # Import is done here to avoid polluting the main process memory
        from voxmlx import transcribe
        
        while True:
            try:
                # Blocking wait for a task
                task_file = input_queue.get()
                
                if task_file == "STOP":
                    break
                
                # Transcription
                text = transcribe(task_file, model_path=model_path).strip()
                output_queue.put(text)
                
            except Exception as e:
                output_queue.put(f"ERROR: {str(e)}")
                
    except ImportError:
        output_queue.put("ERROR: Could not import voxmlx")
    except Exception as main_e:
        output_queue.put(f"ERROR: Worker crash: {str(main_e)}")

# --- Main Controller ---
class AudioState:
    is_recording = False
    recording_data = []
    stream = None

class ServiceState:
    process = None
    input_queue = None
    output_queue = None
    timeout_timer = None

audio_state = AudioState()
service_state = ServiceState()

def stop_service_timeout():
    """Called by the timer when the timeout is reached"""
    if service_state.process and service_state.process.is_alive():
        log(f"\n[Info] Inactivity detected ({MODEL_TIMEOUT}s). Unloading model from RAM.", verbose_only=True)
        service_state.process.terminate()
        service_state.process.join()
        service_state.process = None
        service_state.input_queue = None
        service_state.output_queue = None

def reset_timer():
    """Restarts the idle timeout timer"""
    if service_state.timeout_timer:
        service_state.timeout_timer.cancel()
    
    service_state.timeout_timer = threading.Timer(MODEL_TIMEOUT, stop_service_timeout)
    service_state.timeout_timer.start()

def get_transcription(wav_file):
    """Handles the lifecycle of the transcription process"""
    
    # 1. Start worker if necessary
    if service_state.process is None or not service_state.process.is_alive():
        log("[Info] Loading model into memory... (This may take a few seconds)", verbose_only=True)
        service_state.input_queue = multiprocessing.Queue()
        service_state.output_queue = multiprocessing.Queue()
        
        service_state.process = multiprocessing.Process(
            target=model_worker,
            args=(service_state.input_queue, service_state.output_queue, LOCAL_DIR)
        )
        service_state.process.start()
    
    # 2. Send task
    service_state.input_queue.put(wav_file)
    
    # 3. Wait for result
    try:
        # Wait for result with a generous timeout
        result = service_state.output_queue.get(timeout=60)
        return result
    except queue.Empty:
        return "ERROR: Transcription timeout"

def callback(indata, frames, time, status):
    if audio_state.is_recording:
        audio_state.recording_data.append(indata.copy())

def toggle_recording(start):
    if start and not audio_state.is_recording:
        # Cancel timeout timer while recording
        if service_state.timeout_timer:
            service_state.timeout_timer.cancel()
            
        log("Recording started...", verbose_only=True)
        audio_state.recording_data = []
        audio_state.is_recording = True
        audio_state.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
        audio_state.stream.start()
        
    elif not start and audio_state.is_recording:
        log("Recording stopped. Processing...", verbose_only=True)
        audio_state.is_recording = False
        audio_state.stream.stop()
        audio_state.stream.close()
        
        # Save audio
        if audio_state.recording_data:
            audio_data = np.concatenate(audio_state.recording_data, axis=0)
            sf.write(WAVE_FILENAME, audio_data, SAMPLE_RATE)
            
            # Call transcription service
            result = get_transcription(WAVE_FILENAME)
            
            if result and not result.startswith("ERROR:"):
                pyperclip.copy(result)
                print(f"> {result}")
                paste_text()
            else:
                print(f"{result}")
            
            # Start idle countdown to free RAM
            reset_timer()
            
        else:
            log("No audio data recorded.", verbose_only=True)

def on_press(key):
    if key == keyboard.Key.cmd_r:
        if HOLD_TO_TALK:
            if not audio_state.is_recording:
                toggle_recording(True)
        else:
            if audio_state.is_recording:
                toggle_recording(False)
            else:
                toggle_recording(True)

def on_release(key):
    if key == keyboard.Key.cmd_r and HOLD_TO_TALK:
        if audio_state.is_recording:
            toggle_recording(False)

# --- Required entry point for multiprocessing on macOS ---
if __name__ == "__main__":
    # Initial download if missing
    if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
        log("Initial model download...")
        snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, token=True)

    mode_str = "HOLD TO TALK" if HOLD_TO_TALK else "TOGGLE (Press/Press)"
    auto_paste_str = "ON" if AUTO_PASTE else "OFF"
    
    # Startup messages
    if args.verbose:
        print(f"Ready. Mode: {mode_str} (Key: Cmd+R)")
        print(f"Auto-Paste: {auto_paste_str}")
        print(f"RAM Timeout: {MODEL_TIMEOUT}s")
    else:
        print("ready.")
    
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        log("\nStopping program...", verbose_only=True)
    finally:
        # Proper cleanup
        if service_state.timeout_timer:
            service_state.timeout_timer.cancel()
        
        if service_state.process and service_state.process.is_alive():
            service_state.process.terminate()
            service_state.process.join()
        
        log("Cleanup finished. Goodbye!", verbose_only=True)
