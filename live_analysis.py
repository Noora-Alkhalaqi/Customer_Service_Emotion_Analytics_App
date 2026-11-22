"""Emotion model for live calls (phase B). Same models and label rule as the main app:
WavLM-base embedding (mean of the last hidden state) -> the saved logistic-regression classifier -> 'uncertain' below the threshold."""
import os
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODEL_DIR = Path(__file__).resolve().parent / "model"


class EmotionAnalyzer:
    """Loads the models once and turns a chunk of 16 kHz mono audio into class probabilities."""

    def __init__(self, model_dir=MODEL_DIR):
        import joblib
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel

        self._torch = torch
        self.extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").eval()
        self.clf = joblib.load(os.path.join(model_dir, "emotion_classifier.joblib"))
        self.classes = [str(c) for c in self.clf.classes_]

    def predict(self, audio):
        """audio: float32 array in [-1, 1] at 16 kHz. Returns one probability per class."""
        if len(audio) < 1600:
            audio = np.pad(audio, (0, 1600 - len(audio)))
        inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with self._torch.no_grad():
            hidden = self.wavlm(**inputs).last_hidden_state
        return self.clf.predict_proba(hidden.mean(dim=1).numpy())[0]


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
