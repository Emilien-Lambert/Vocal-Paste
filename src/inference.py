import multiprocessing
import queue
import time
import threading
from pathlib import Path
import src.config as config
from src.utils import log

# --- Isolated Worker Process ---
def _model_worker_process(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, model_path: str):
    """
    Runs in a separate process. Loads model once, then handles requests.
    """
    try:
        from voxmlx import load_model, generate, SpecialTokenPolicy
        from voxmlx import _build_prompt_tokens
        
        # Load Model & Tokenizer (Once)
        model, sp, config_model = load_model(model_path)
        prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
        
        while True:
            try:
                task_file = input_queue.get()
                if task_file == "STOP":
                    break
                
                # Inference
                output_tokens = generate(
                    model,
                    task_file,
                    prompt_tokens,
                    n_delay_tokens=n_delay_tokens,
                    temperature=0.0,
                    eos_token_id=sp.eos_id,
                )
                
                text = sp.decode(output_tokens, special_token_policy=SpecialTokenPolicy.IGNORE).strip()
                output_queue.put(text)
                
            except Exception as e:
                output_queue.put(f"ERROR: {e}")
                
    except ImportError as e:
        output_queue.put(f"ERROR: Import failed: {e}")
    except Exception as e:
        output_queue.put(f"ERROR: Worker crash: {e}")

# --- Main Thread Service Manager ---
class InferenceService:
    def __init__(self):
        self.process = None
        self.input_queue = None
        self.output_queue = None
        self.timeout_timer = None

    def _start_worker(self):
        """Starts the background worker process."""
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=_model_worker_process,
            args=(self.input_queue, self.output_queue, config.MODELS_DIR)
        )
        self.process.start()

    def _stop_worker(self):
        """Stops the worker process to free RAM."""
        if self.process and self.process.is_alive():
            log(f"Inactivity detected ({config.MODEL_TIMEOUT}s). Unloading model.", verbose_only=True)
            self.process.terminate()
            self.process.join()
        self.process = None
        self.input_queue = None
        self.output_queue = None

    def _reset_timer(self):
        """Resets the inactivity timer."""
        if self.timeout_timer:
            self.timeout_timer.cancel()
        self.timeout_timer = threading.Timer(config.MODEL_TIMEOUT, self._stop_worker)
        self.timeout_timer.start()

    def initialize(self):
        """Pre-loads the model and starts the inactivity timer."""
        if self.process is None or not self.process.is_alive():
            log("Pre-loading model...", verbose_only=True)
            self._start_worker()
            self._reset_timer()

    def transcribe(self, wav_path: str) -> str:
        """Public API to get transcription."""
        start_time = time.time()

        # Ensure worker is running
        if self.process is None or not self.process.is_alive():
            self._start_worker()

        # Send Task
        self.input_queue.put(wav_path)

        # Wait for Result
        try:
            result = self.output_queue.get(timeout=60)
            elapsed = time.time() - start_time
            log(f"Inference time: {elapsed:.2f}s", verbose_only=True)
            
            self._reset_timer()
            return result
        except queue.Empty:
            return "ERROR: Timeout"

    def shutdown(self):
        """Cleanup on exit."""
        if self.timeout_timer:
            self.timeout_timer.cancel()
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
