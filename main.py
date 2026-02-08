import os
import subprocess
import signal
import pyperclip
from pynput import keyboard
from huggingface_hub import snapshot_download

# On désactive les barres pour le silence demandé
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Import différé pour éviter les logs au démarrage si possible
from voxmlx import transcribe

MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
LOCAL_DIR = os.path.join(os.getcwd(), "models")
WAVE_FILENAME = "temp_voice.wav"

# --- TELECHARGEMENT LOCAL FORCE ---
if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
    print("chargement du model")
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, token=True)

class State:
    recording_process = None

state = State()

def toggle_recording(start):
    if start and not state.recording_process:
        state.recording_process = subprocess.Popen(
            ["rec", "-q", "-r", "48000", "-c", "1", "-b", "16", WAVE_FILENAME],
            preexec_fn=os.setsid
        )
    elif not start and state.recording_process:
        os.killpg(os.getpgid(state.recording_process.pid), signal.SIGTERM)
        state.recording_process.wait()
        state.recording_process = None
        
        try:
            # On utilise LOCAL_DIR au lieu du MODEL_ID pour forcer l'usage local
            text = transcribe(WAVE_FILENAME, model_path=LOCAL_DIR).strip()
            if text:
                pyperclip.copy(text)
                print(text)
        except:
            pass

def on_press(key):
    if key == keyboard.Key.cmd_r:
        if state.recording_process:
            toggle_recording(False)
        else:
            toggle_recording(True)

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()