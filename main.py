import os
import pyperclip
from pynput import keyboard
from huggingface_hub import snapshot_download
import sounddevice as sd
import soundfile as sf
import numpy as np

# On désactive les barres pour le silence demandé
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Import différé pour éviter les logs au démarrage si possible
from voxmlx import transcribe

MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"
LOCAL_DIR = os.path.join(os.getcwd(), "models")
WAVE_FILENAME = "temp_voice.wav"
SAMPLE_RATE = 48000

# --- TELECHARGEMENT LOCAL FORCE ---
if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
    print("chargement du model")
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR, token=True)

class State:
    is_recording = False
    recording_data = []
    stream = None

state = State()

def callback(indata, frames, time, status):
    if state.is_recording:
        state.recording_data.append(indata.copy())

def toggle_recording(start):
    if start and not state.is_recording:
        print("Début de l'enregistrement...", end="\r", flush=True)
        state.recording_data = []
        state.is_recording = True
        state.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
        state.stream.start()
        
    elif not start and state.is_recording:
        print("Fin de l'enregistrement.    ", end="\r", flush=True)
        state.is_recording = False
        state.stream.stop()
        state.stream.close()
        
        # Sauvegarde du fichier WAV
        if state.recording_data:
            audio_data = np.concatenate(state.recording_data, axis=0)
            sf.write(WAVE_FILENAME, audio_data, SAMPLE_RATE)
            
            try:
                # On utilise LOCAL_DIR au lieu du MODEL_ID pour forcer l'usage local
                text = transcribe(WAVE_FILENAME, model_path=LOCAL_DIR).strip()
                if text:
                    pyperclip.copy(text)
                    print(text)
            except Exception as e:
                print(f"Erreur de transcription : {e}")
        else:
            print("Aucune donnée enregistrée.")

def on_press(key):
    if key == keyboard.Key.cmd_r:
        if state.is_recording:
            toggle_recording(False)
        else:
            toggle_recording(True)

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()