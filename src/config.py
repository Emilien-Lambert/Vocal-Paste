import os
from dotenv import load_dotenv
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

parser = argparse.ArgumentParser(description="Vocal-Paste: Voice to Clipboard/Paste utility")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
args = parser.parse_args()

VERBOSE = args.verbose
HOLD_TO_TALK = os.getenv("HOLD_TO_TALK", "false").lower() == "true"
AUTO_PASTE = os.getenv("AUTO_PASTE", "false").lower() == "true"
SAMPLE_RATE = 16000

MODEL_ID = "mlx-community/Voxtral-Mini-4B-Realtime-6bit"

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
