import sounddevice as sd
import numpy as np
import src.config as config
from src.utils import log


class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.stream = None
        self._on_chunk = None

    def start(self, on_chunk=None):
        """Starts audio recording at 16kHz. Chunks are sent via on_chunk callback."""
        if not self.is_recording:
            log("Recording started...", verbose_only=True)
            self._on_chunk = on_chunk
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()

    def _callback(self, indata, frames, time, status):
        if self.is_recording and self._on_chunk is not None:
            self._on_chunk(indata[:, 0].copy())

    def stop(self):
        """Stops recording."""
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self._on_chunk = None
            log("Recording stopped.", verbose_only=True)
