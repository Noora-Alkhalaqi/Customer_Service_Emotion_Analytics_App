import os
import sys

import librosa
import torch
from pyannote.audio import Pipeline

audio_path = sys.argv[1]
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN is not set. Set it first (see instructions).")

samples, sr = librosa.load(audio_path, sr=16000, mono=True)
waveform = torch.from_numpy(samples).unsqueeze(0)
print("Audio duration (seconds):", round(len(samples) / 16000, 1))

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
output = pipeline({"waveform": waveform, "sample_rate": 16000}, num_speakers=2)

annotation = output.speaker_diarization if hasattr(output, "speaker_diarization") else output

totals = {}
for turn, _, speaker in annotation.itertracks(yield_label=True):
    print(f"{turn.start:7.2f} - {turn.end:7.2f}  {speaker}")
    totals[speaker] = totals.get(speaker, 0) + (turn.end - turn.start)

print("Speaking time per speaker (seconds):")
for speaker, seconds in totals.items():
    print(" ", speaker, round(seconds, 1))