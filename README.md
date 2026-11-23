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

## Installation and local setup (Windows / PowerShell)

These instructions describe how to run the project **on your own computer**. Python **3.12** was used during development. A CPU-only computer can run the application, although model loading and analysis may be slow. Internet access is needed for the initial model downloads.

### 1. Download and organize the project

Clone the repository (replace the URL with this project's GitHub repository URL), or download its ZIP and extract it. Open PowerShell **inside the project folder**. The files must have their original names (for example, `app.py`, not `app(3).py`). The expected layout is:

```text
project-folder/
├── app.py
├── analysis.py
├── ui_theme.py
├── notifier.py
├── live_server.py
├── live_analysis.py
├── bench_live.py
├── test_diarization.py
├── requirements.txt
├── README.md
├── .env                         # create locally; never commit
├── model/
│   ├── emotion_classifier.joblib
│   ├── text_classifier.joblib
│   ├── metrics.json
│   └── fusion_results.json
└── .streamlit/
    └── config.toml             # optional theme configuration
```

**Important:** The Python code expects the saved classifiers and JSON files **inside `model/`**, not beside `app.py`. The audio classifier is required; the text classifier enables optional audio-text fusion. These saved classifier files must be obtained from the project distribution; `pip install` does not create them. The `.streamlit/config.toml` file is optional, and its absence does not prevent the app from starting.

### 2. Create a Python environment

Verify Python 3.12 is installed:

```powershell
py -3.12 --version
```

Then, from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the following command **for the current terminal session** and try activating again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The supplied `requirements.txt` uses PyTorch's CPU wheel index as an extra index. The saved classifiers were created with **scikit-learn 1.6.1**; for compatibility, install that version if necessary:

```powershell
python -m pip install scikit-learn==1.6.1
```

### 3. Set up Hugging Face access (required)

1. Create or sign in to a [Hugging Face](https://huggingface.co/) account.
2. Open [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept its access conditions. Follow any additional gated-model access instructions shown on that page.
3. Create a Hugging Face **read** access token in your account settings.
4. In the same folder as `app.py`, create a text file named **`.env`** (not `.env.txt`) containing:

```dotenv
HF_TOKEN=hf_your_token_here
```

The app checks for `HF_TOKEN` when it starts. Never upload the real token or the `.env` file to GitHub. Existing operating-system environment variables take precedence over values in `.env`.

### 4. Run the application (uploaded calls)

With the virtual environment active, run:

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit (normally **http://localhost:8501**). Upload an English-language WAV or MP3 customer-service call and select **Start analysis**. The application performs speaker diarization, emotion analysis, and—when speech-to-text is enabled—transcription and AI report generation. The first analysis may download WavLM, Whisper, the text encoder and Qwen model files, so allow extra time and sufficient disk space.

**Note:** For uploaded calls, the first detected speaker is automatically treated as the **Agent** and the second as the **Customer**. Check whether this assumption fits your recording.

### 5. Run the live-call feature (two terminals)

The live-call feature uses a separate local web server. Both commands must run from the same project folder and environment.

**Terminal 1 — live-call server:**

```powershell
.\.venv\Scripts\Activate.ps1
python live_server.py
```

The live-call server normally listens at **http://localhost:8765**. Wait for the emotion model to load.

**Terminal 2 — Streamlit interface:**

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

In Streamlit, open **Live Call**, create a room, and use the Agent and Customer links to join the call. Allow microphone access in both browsers. The server receives and relays the two audio streams, analyzes short audio chunks and stores local speaker recordings in `live_calls/`. Use headphones to reduce echo. End the call from the application to obtain the post-call analysis and report.

**For people on different devices/networks:** A `localhost` link cannot be opened on another person's device. One optional method is a temporary HTTPS tunnel using Cloudflare's `cloudflared` CLI (install it separately):

```powershell
cloudflared tunnel --url http://localhost:8765
```

Copy the generated `https://...trycloudflare.com` address into the local `.env` file:

```dotenv
LIVE_PUBLIC_URL=https://your-generated-address.trycloudflare.com
```

Restart **Streamlit** so it reads the updated URL. Keep the live server and tunnel running. Temporary tunnel addresses can change between sessions. Microphone access in remote browsers generally requires a secure HTTPS origin. A tunnel makes the call server reachable from the internet: share room links only with intended participants, obtain consent to recording, and do not treat a temporary tunnel as a production deployment.

### 6. Optional Telegram integration

To send call links or reports through Telegram, create a bot using [@BotFather](https://t.me/BotFather), obtain the intended recipient chat ID, and add these values to `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

Telegram is **optional**; the local upload analysis and browser call feature do not require it. The user must initiate contact with the bot or otherwise permit bot messages as required by Telegram.

### Optional configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | none | Required for gated speaker-diarization model access |
| `LIVE_PORT` | `8765` | Local live server port |
| `LIVE_SERVER_URL` | `http://localhost:8765` | Address Streamlit uses to contact the live server |
| `LIVE_PUBLIC_URL` | same as `LIVE_SERVER_URL` | Browser-facing URL for remote call participants |
| `LIVE_CALLS_DIR` | `live_calls` | Directory for live recordings and local admin key |
| `LIVE_ADMIN_KEY` | automatically generated in `live_calls/.admin_key` | Shared secret used for administrative live-call API requests |
| `TELEGRAM_BOT_TOKEN` | none | Optional Telegram bot token |
| `TELEGRAM_CHAT_ID` | none | Optional Telegram destination |

### Troubleshooting

| Problem | What to check |
|---|---|
| `HF_TOKEN is not set` | Ensure `.env` is beside `app.py`, is not named `.env.txt`, and contains a valid `HF_TOKEN` |
| Access denied to pyannote model | Accept the gated model conditions and confirm the Hugging Face token has appropriate access |
| `FileNotFoundError` for a `.joblib` file | Check that `model/emotion_classifier.joblib` exists; place the optional text classifier in the same folder |
| scikit-learn version warning | Install `scikit-learn==1.6.1` in the active environment |
| `streamlit` command not found | Activate `.venv` and install the packages; alternatively run `python -m streamlit run app.py` |
| Live call cannot connect | Start `python live_server.py`, check port 8765 and `LIVE_SERVER_URL`, and keep both terminals open |
| Customer cannot open a `localhost` link | Use a reachable HTTPS URL and set `LIVE_PUBLIC_URL` before restarting Streamlit |
| Microphone does not work | Grant browser microphone permission, use HTTPS for remote devices, and check the device's input settings |
| First run is very slow | Model weights are downloaded on demand and CPU inference can be resource-intensive |

**Verification note:** These steps are derived from the supplied project source code and configuration. They have not been validated by a full end-to-end installation on a fresh Windows computer.

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
