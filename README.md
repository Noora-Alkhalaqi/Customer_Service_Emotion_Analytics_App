# Customer Service Speech Emotion Analytics

A Streamlit app that analyzes English customer-service calls, either uploaded after the call or live. It separates the two speakers, predicts the emotion of each speech segment, and writes a short AI call report.

> Emotion labels are **model predictions, not confirmed emotions**. The app is not meant for disciplinary or HR decisions about people.

## What it does
- **Upload & Analysis:** upload a WAV/MP3 call and press *Start analysis*. One click runs:
  1. speaker separation (pyannote `speaker-diarization-3.1`, 2 speakers),
  2. emotion analysis for the Customer and the Agent (WavLM-base audio model + saved classifier; optionally combined with a text model on a Whisper transcript),
  3. the AI Call Report (Qwen2.5-1.5B-Instruct, run locally).
- **Speaker roles:** the speaker who talks first is the Agent; the other is the Customer.
- **Emotion analytics (per speaker):** timeline with real timestamps, duration bar chart, percentage donut, KPI cards (analyzed segments, analyzed speech duration, classification coverage, uncertain %, most frequent label, audio-text agreement) and a segment-level table.
- **AI Call Report:** call overview, reason for contact, key discussion points, customer requests, agent actions, outcome and important details. Items not supported by the transcript are removed and sensitive data is redacted. It needs speech-to-text to be on.
- **Live Call:** create a call, send the customer link through Telegram, and talk in the browser from two devices (each side hears the other). The emotion of each speaker is shown every few seconds. *End call and analyze* gives the same sections and AI report as an uploaded call.
- **Export:** a self-contained HTML report (works offline, prints to A4), plus an optional Telegram message with a short summary and the report file.

### Sidebar settings
| Setting | Default |
|---|---|
| Confidence threshold (below it the label is *uncertain*) | 0.40 |
| Maximum analysis window | 4 s |
| Live call: analyze every | 5 s |
| Speech-to-text cross-check (Whisper + text model) | on if `text_classifier.joblib` exists |
| Speech-to-text model size | `base.en` (or `small.en`) |
| Include transcript in the downloaded report | off |

## Files
| File | Purpose |
|---|---|
| `app.py` | the Streamlit app |
| `analysis.py` | analysis, charts, AI report and the HTML export (also defines the emotion colors) |
| `ui_theme.py` | app colors, CSS and sidebar navigation |
| `live_server.py` | live-call server: call page, audio relay, recording, per-speaker emotion analysis |
| `live_analysis.py` | emotion model used by the live server |
| `notifier.py` | Telegram sender and the `.env` reader |
| `bench_live.py` | speed test: how long this computer needs to analyze 5 s of audio |
| `test_diarization.py` | quick check that speaker separation works on a file |
| `.streamlit/config.toml` | Streamlit theme |
| `requirements.txt` | Python packages |
| `.gitignore` | files that must not be published (`.env`, `live_calls/`, caches) |
| `model/` | saved classifiers and their test results (see below) |
| `live_calls/` | recordings of live calls and the local admin key (created automatically; do not publish) |
| `Call1.wav` … `Call4.wav` | sample calls for testing |

## Setup (Windows, PowerShell)
Python 3.12 was used for development.

1. Create and activate a virtual environment, then install the packages:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Install a CPU build of torch if you have no GPU. The classifiers were saved with scikit-learn 1.6.1; if you see a version warning, run `pip install scikit-learn==1.6.1`.
2. Accept the terms of `pyannote/speaker-diarization-3.1` on your Hugging Face account and create an access token.
3. Create a file named `.env` next to `app.py`:
   ```
   HF_TOKEN=...               # required: Hugging Face token
   TELEGRAM_BOT_TOKEN=...     # optional: from @BotFather
   TELEGRAM_CHAT_ID=...       # optional: the chat that receives messages
   LIVE_PUBLIC_URL=...        # optional: https address of your tunnel, for customers on other devices
   ```
   Advanced, optional: `LIVE_PORT` (default 8765), `LIVE_SERVER_URL` (default `http://localhost:8765`), `LIVE_CALLS_DIR` (default `live_calls`), `LIVE_ADMIN_KEY` (by default a random key is created in `live_calls/.admin_key`).
   Variables already set in the terminal win over `.env`. **Never share or publish `.env`.**
4. Make sure the `model` folder is next to `app.py`.

The first run downloads the Hugging Face models (WavLM, Whisper, the text encoder, Qwen), which takes a while.

## Run
- **Upload mode only:** `streamlit run app.py`
- **Live calls**, in separate terminals:
  1. `python live_server.py`
  2. `streamlit run app.py`
  3. only if the customer is on another device: `cloudflared tunnel --url http://localhost:8765`, put the address it prints in `.env` as `LIVE_PUBLIC_URL`, and restart Streamlit (the address changes every time).

Use headphones in live calls to avoid echo.

### Helper scripts
- `python bench_live.py` – measures emotion-model speed; add `--whisper` to also time speech-to-text. If the time is clearly below 5 s, live analysis every 5 seconds is realistic on this computer.
- `python test_diarization.py Call1.wav` – prints the speaker turns and speaking time per speaker.

## Model quality
Tested on the MELD test split (2,610 utterances, 7 labels: anger, disgust, fear, joy, neutral, sadness, surprise):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Audio only (WavLM) | 0.459 | 0.236 |
| Text only (all-mpnet-base-v2) | 0.543 | 0.338 |
| Audio + text (audio weight 0.4) | 0.566 | 0.341 |
| Always "neutral" baseline | 0.481 | – |

The audio-only model is below the always-neutral baseline in accuracy, so treat single labels with care. Full numbers are in `model/metrics.json` and `model/fusion_results.json`.

## How the numbers are defined
- Segment duration = end − start. Percentages are seconds of an emotion divided by the speaker's analyzed seconds (*uncertain* included).
- Classification coverage = time of segments that got one of the 7 emotion labels / time of all analyzed segments. It is **not** model accuracy.
- A segment is *uncertain* when its top probability is below the confidence threshold (a user setting, default 0.40, not validated on real calls).
- Uncertain %, most frequent label and audio-text agreement are counted by segments.

## Privacy
Recordings in `live_calls/`, the `.env` file and the sample calls may contain personal data. Keep them off public repositories (see `.gitignore`). The AI report runs locally; only Telegram messages you choose to send leave the computer.
