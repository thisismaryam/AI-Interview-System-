"""
Isolated speech I/O. interview.py never imports elevenlabs directly —
only this file does. Swap TTS/STT providers here without touching the graph.

Requires ffmpeg installed and on PATH (used to decode streamed mp3 -> PCM,
since raw pcm output from ElevenLabs requires a Pro-tier subscription).
"""
import os
import subprocess
import threading
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_TTS_API_KEY"))
stt_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_STT_API_KEY"))
VOICE_ID = "0SuGVMgHnvTR1BjC6j4u"  # free-tier accessible default voice
# VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default, works on all accounts)


def speak(text: str):
    """Stream TTS audio, decode mp3 -> PCM on the fly via ffmpeg, play as it arrives."""
    audio = tts_client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_turbo_v2",  # Faster than flash
        # model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128")

    # ffmpeg decodes the incoming mp3 stream into raw PCM as chunks arrive
    decoder = subprocess.Popen(
        ["ffmpeg", "-i", "pipe:0", "-f", "s16le",
            "-ar", "44100", "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit terminal so ffmpeg errors are visible, not swallowed
    )

    def feed_mp3_chunks():
        for chunk in audio:
            decoder.stdin.write(chunk)
        decoder.stdin.close()

    feeder = threading.Thread(target=feed_mp3_chunks, daemon=True)
    feeder.start()

    stream = sd.OutputStream(samplerate=44100, channels=1, dtype=np.int16)
    stream.start()

    while True:
        pcm_chunk = decoder.stdout.read(4096)
        if not pcm_chunk:
            break
        chunk_array = np.frombuffer(pcm_chunk, dtype=np.int16)
        stream.write(chunk_array)

    stream.stop()
    stream.close()
    feeder.join(timeout=1)
    decoder.wait()


def transcribe(audio_path: str) -> str:
    """Transcribe a local audio file to text."""
    with open(audio_path, "rb") as f:
        transcript = stt_client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            tag_audio_events=False,
            language_code="eng",
            diarize=False,
        )
    return transcript.text
