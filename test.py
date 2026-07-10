import wave
import math
import struct

sample_rate = 44100
duration = 1.0
frequency = 440
amplitude = 16000

with wave.open("test.wav", "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)

    for i in range(int(sample_rate * duration)):
        sample = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        wav_file.writeframes(struct.pack("<h", sample))

print("Created test.wav")

import base64

import base64

with open("test.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

with open("audio_base64.txt", "w", encoding="utf-8") as f:
    f.write(audio_base64)