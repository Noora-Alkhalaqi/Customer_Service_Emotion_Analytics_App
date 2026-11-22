"""Speed test for live analysis. Run it once from the project folder:   python bench_live.py
It measures how long YOUR computer needs to analyze 5 seconds of audio with the emotion model (and optionally with
Whisper). If the audio-only time is clearly below 5 s, live analysis every 5 seconds is realistic on this computer.
Nothing is uploaded; the audio is synthetic noise (speed does not depend on the content)."""
import os
import statistics
import sys
import time

import numpy as np

import notifier  # noqa: F401  (loads the .env file, so HF_TOKEN is available)

SR = 16000
CHUNK_SECONDS = 5
RUNS = 8
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def report(name, times):
    med, worst = statistics.median(times), max(times)
    verdict = "fast enough" if worst < CHUNK_SECONDS * 0.8 else ("borderline" if med < CHUNK_SECONDS else "TOO SLOW for every-5-seconds")
    print(f"{name:34s} median {med:5.2f} s | slowest {worst:5.2f} s | for {CHUNK_SECONDS} s of audio -> {verdict}")


def main():
    import joblib
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel

    print(f"CPU threads used by torch: {torch.get_num_threads()}  |  GPU available: {torch.cuda.is_available()}")
    t0 = time.perf_counter()
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()
    clf = joblib.load(os.path.join(MODEL_DIR, "emotion_classifier.joblib"))
    print(f"models loaded in {time.perf_counter() - t0:.1f} s (one time, at start)\n")

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SR * CHUNK_SECONDS) * 0.05).astype(np.float32)

    def emotion_once():
        inputs = extractor(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            hidden = wavlm(**inputs).last_hidden_state
        clf.predict_proba(hidden.mean(dim=1).numpy())

    emotion_once()                                   # warm-up (first call is always slower)
    times = []
    for _ in range(RUNS):
        t = time.perf_counter(); emotion_once(); times.append(time.perf_counter() - t)
    report("emotion model (audio only)", times)

    if "--whisper" in sys.argv:
        from faster_whisper import WhisperModel
        asr = WhisperModel("base.en", device="cpu", compute_type="int8")
        def asr_once():
            segments, _ = asr.transcribe(audio, language="en"); list(segments)
        asr_once()
        times = []
        for _ in range(RUNS):
            t = time.perf_counter(); asr_once(); times.append(time.perf_counter() - t)
        report("Whisper base.en (speech to text)", times)
    else:
        print("\n(Run 'python bench_live.py --whisper' to also time the speech-to-text step.)")


if __name__ == "__main__":
    main()
