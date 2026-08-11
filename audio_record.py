import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
MAX_SECONDS = 90

_chunks = []
_stream = None
_safety_timer = None


def start_recording():
    global _stream, _chunks, _safety_timer
    _chunks = []

    def callback(indata, frames, time_info, status):
        _chunks.append(indata.copy())

    _stream = sd.InputStream(samplerate=SAMPLE_RATE,
                             channels=1, callback=callback)
    _stream.start()

    _safety_timer = threading.Timer(MAX_SECONDS, stop_recording)
    _safety_timer.start()


def stop_recording(output_path: str = "answer.wav") -> str:
    global _stream, _safety_timer

    if _safety_timer:
        _safety_timer.cancel()
    _stream.stop()
    _stream.close()

    audio_data = np.concatenate(_chunks, axis=0)
    write(output_path, SAMPLE_RATE, audio_data)
    print(f"Saved recording to {output_path}")
    return output_path


def record_until_enter(output_path: str = "answer.wav") -> str:
    print("Recording... press ENTER when you're done answering.")
    start_recording()
    input()
    return stop_recording(output_path)
