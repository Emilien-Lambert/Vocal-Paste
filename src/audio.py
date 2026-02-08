import sounddevice as sd
import soundfile as sf
import numpy as np
import src.config as config
from src.utils import log

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.recording_data = []
        self.stream = None

    def _callback(self, indata, frames, time, status):
        """Internal callback for sounddevice."""
        if self.is_recording:
            self.recording_data.append(indata.copy())

    def start(self):
        """Starts the audio recording stream."""
        if not self.is_recording:
            log("Recording started...", verbose_only=True)
            self.recording_data = []
            self.is_recording = True
            self.stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE, 
                channels=1, 
                callback=self._callback
            )
            self.stream.start()

    def stop(self) -> str | None:
        """Stops recording, saves file, and returns the filename."""
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            
            log("Recording stopped.", verbose_only=True)
            log("Processing...", verbose_only=True)

            if not self.recording_data:
                log("No audio data recorded.", verbose_only=True)
                return None

            # Save to file
            audio_data = np.concatenate(self.recording_data, axis=0)
            sf.write(config.TEMP_WAVE_FILE, audio_data, config.SAMPLE_RATE)
            return config.TEMP_WAVE_FILE
        return None
