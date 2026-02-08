import os
from dotenv import load_dotenv
import argparse

# Determine paths relative to this file (src/config.py)
# This ensures we find the project root (Vocal-Paste/) correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_WAVE_FILE = os.path.join(BASE_DIR, "temp_voice.wav")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load environment variables from the root .env
load_dotenv(dotenv_path=ENV_PATH)

# CLI Arguments parsing
parser = argparse.ArgumentParser(description="Vocal-Paste: Voice to Clipboard/Paste utility")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
args = parser.parse_args()

# Constants
VERBOSE = args.verbose
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", 300))
HOLD_TO_TALK = os.getenv("HOLD_TO_TALK", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "false").lower() == "true"
SAMPLE_RATE = 48000

# Model Settings
MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"

# Environment Setup
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"