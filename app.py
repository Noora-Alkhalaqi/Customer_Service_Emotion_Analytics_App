import html
import json
import os
import tempfile

import requests

import numpy as np
import pandas as pd
import streamlit as st

from notifier import send_telegram_document, send_telegram_message, telegram_configured   # also loads the local .env file (HF_TOKEN, Telegram...)
from analysis import (
    COLORS,
    LIST_FIELDS,
    NOT_MENTIONED,
    REPORT_TITLES,
    agreement_stats,
    assign_roles,
    build_multimodal_table,
    build_call_summary_text,
    build_report_html,
    build_transcript,
    classification_coverage,
    format_coverage,
    generate_report,
    fmt_time,
    make_charts,
    make_llm,
    make_windows,
    speaker_stats,
    assign_texts_to_windows,
    text_probabilities,
    transcribe_segments,
    transcribe_windows,
)
from ui_theme import (
    APP_CSS,
    app_header_html,
    emotion_chip_html,
    live_status_html,
    empty_html,
    kpi_html,
    rcard_html,
    predictions_table_html,
    role_card_html,
    section_header_html,
    sidebar_nav_html,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SR = 16000
TEXT_ENCODER = "sentence-transformers/all-mpnet-base-v2"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_json(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


fusion = load_json("fusion_results.json")
text_model_available = os.path.exists(os.path.join(MODEL_DIR, "text_classifier.joblib"))
w_audio = float(fusion["chosen_audio_weight"]) if fusion and "chosen_audio_weight" in fusion else 0.4

st.set_page_config(page_title="Call Emotion Analytics", layout="wide")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
st.markdown(
    app_header_html(
        "Customer Service Speech Emotion Analytics",
        "Analyze customer service conversations using speaker diarization, emotion recognition, and AI-powered reporting.",
    ),
    unsafe_allow_html=True,
)

with st.sidebar:
    nav_slot = st.empty()          # filled at the end of the script, once we know which sections exist
    st.markdown('<div class="nav-title">Settings</div>', unsafe_allow_html=True)
    threshold = st.slider("Confidence threshold (below it the label is 'uncertain')", 0.0, 0.9, 0.40, 0.05)
    win = st.slider("Maximum analysis window (seconds)", 2.0, 8.0, 4.0, 0.5)
    chunk_seconds = st.slider("Live call: analyze every (seconds)", 3, 10, 5, key="live-chunk")
    st.markdown('<div class="side-note">The threshold is a user setting and has not been validated on real calls.</div>', unsafe_allow_html=True)
    use_text = st.checkbox(
        "Add speech-to-text cross-check (Whisper + text model)",
        value=text_model_available,
        disabled=not text_model_available,
        help="Transcribes each segment, predicts emotion from the words, and combines it with the audio prediction.",
    )
    if not text_model_available:
        st.caption("text_classifier.joblib was not found in the model folder, so this option is off.")
    asr_size = st.selectbox("Speech-to-text model size", ["base.en", "small.en"], index=0)
    include_transcript = st.checkbox("Include the transcript text in the downloaded report", value=False)

token = os.environ.get("HF_TOKEN")
if not token:
    st.error("HF_TOKEN is not set. Put the line HF_TOKEN=... in the .env file next to app.py (or set it in the terminal), then run the app again.")
    st.stop()


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner=False)
def load_diarization_pipeline():
    from pyannote.audio import Pipeline

    name = "pyannote/speaker-diarization-3.1"
    try:
        return Pipeline.from_pretrained(name, token=token)
    except TypeError:
        return Pipeline.from_pretrained(name, use_auth_token=token)


@st.cache_resource(show_spinner=False)
def load_emotion_models():
    import joblib
    from transformers import AutoFeatureExtractor, WavLMModel

    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
    wavlm.eval()
    clf = joblib.load(os.path.join(MODEL_DIR, "emotion_classifier.joblib"))
    return extractor, wavlm, clf


@st.cache_resource(show_spinner=False)
def load_asr(size):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner=False)
def load_text_stack():
    import joblib
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    enc = AutoModel.from_pretrained(TEXT_ENCODER).to(device).eval()
    tclf = joblib.load(os.path.join(MODEL_DIR, "text_classifier.joblib"))
    return tok, enc, tclf, device


@st.cache_resource(show_spinner=False)
def load_llm(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    model.eval()
    return tok, model


# ---------------------------------------------------------------- pipeline
def run_diarization(uploaded):
    import librosa
    import torch

    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
    try:
        samples, _ = librosa.load(tmp_path, sr=SR, mono=True)
    finally:
        os.remove(tmp_path)

    waveform = torch.from_numpy(samples).unsqueeze(0)
    pipeline = load_diarization_pipeline()
    output = pipeline({"waveform": waveform, "sample_rate": SR}, num_speakers=2)
    annotation = output.speaker_diarization if hasattr(output, "speaker_diarization") else output

    rows = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append({"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": speaker})
    turns = pd.DataFrame(rows)
    return samples.astype(np.float32), turns


def embed_windows(samples, windows, extractor, wavlm, progress=None):
    import torch

    vectors = []
    for i, (s, e) in enumerate(windows):
        audio = samples[int(s * SR): int(e * SR)]
        if len(audio) < 1600:
            audio = np.pad(audio, (0, 1600 - len(audio)))
        inputs = extractor(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            hidden = wavlm(**inputs).last_hidden_state
        vectors.append(hidden.mean(dim=1).squeeze(0).numpy())
        if progress is not None:
            progress(i + 1)
    return np.stack(vectors) if vectors else np.zeros((0, 768))


def analyze_speaker(samples, turns, speaker, role, models, win, threshold, progress_bar, base, total, asr=None, text_stack=None):
    extractor, wavlm, clf = models
    windows, skipped = make_windows(turns, speaker, win=win)
    if not windows:
        df = pd.DataFrame(columns=["start", "end", "emotion", "confidence", "mid", "time"])
        return {"df": df, "stats": speaker_stats(df, skipped), "timeline": None, "dist": None, "pie": None}

    n = len(windows)

    def update_audio(k):
        progress_bar.progress(min(1.0, (base + k) / total))

    def update_text(k):
        progress_bar.progress(min(1.0, (base + n + k) / total))

    X = embed_windows(samples, windows, extractor, wavlm, update_audio)
    probs = clf.predict_proba(X)
    classes = [str(c) for c in clf.classes_]

    texts = tprobs = None
    if asr is not None and text_stack is not None:
        texts = transcribe_windows(samples, windows, asr, sr=SR, progress=update_text)
        tprobs = text_probabilities(texts, text_stack, classes)

    df = build_multimodal_table(windows, probs, classes, threshold, texts, tprobs, w_audio)
    timeline, dist, pie = make_charts(df, role)
    return {
        "df": df,
        "stats": speaker_stats(df, skipped),
        "timeline": timeline,
        "dist": dist,
        "pie": pie,
    }


# ------------------------------------------------------------ AI call report
REPORT_PANEL_ICONS = {
    "key_discussion_points": ":material/format_list_bulleted:",
    "customer_requests": ":material/person:",
    "agent_actions": ":material/task_alt:",
    "outcome": ":material/flag:",
    "important_details": ":material/info:",
}


def show_bullets(items):
    if not items:
        st.write(NOT_MENTIONED)
    for item in items:
        st.markdown(f"- {item}")


def show_ai_report(result):
    report, meta = result["report"], result["meta"]
    if meta["parse"] == "failed" or meta["empty"]:
        st.warning("The model did not return a readable report for this call.")
        return
    if meta["parse"] == "partial":
        st.warning("Part of the model answer could not be read, so some sections may show 'Not mentioned'.")
    st.markdown(rcard_html("1. Call Overview", html.escape(report["call_overview"]), "file"), unsafe_allow_html=True)
    st.markdown(rcard_html("2. Reason for Contact", html.escape(report["reason_for_contact"]), "message"), unsafe_allow_html=True)
    with st.container(key="report-sections"):
        for number, (title, key) in enumerate(REPORT_TITLES[2:], start=3):
            with st.expander(f"{number}. {title}", expanded=True, icon=REPORT_PANEL_ICONS[key]):
                if key in LIST_FIELDS:
                    show_bullets(report[key])
                else:
                    st.write(report[key])
    removed = meta["removed_unsupported"] + meta["removed_misattributed"]
    if removed:
        st.caption(f"{removed} statement(s) were left out because they contained numbers not found in the transcript "
                   "or were attributed to the wrong speaker.")
    if meta["parts"] > 1:
        st.caption(f"This was a long call, so the transcript was processed in {meta['parts']} parts.")


# ------------------------------------------------------------ full analysis
def run_emotion_analysis(turns, samples):
    """Emotion analysis for the Agent and the Customer. Returns (results, text_mode)."""
    agent, customer = assign_roles(turns)
    with st.spinner(""):
        models = load_emotion_models()

    asr = text_stack = None
    text_mode = "off"
    if use_text:
        try:
            with st.spinner(""):
                asr = load_asr(asr_size)
                text_stack = load_text_stack()
            text_mode = f"on (Whisper {asr_size}, audio weight {w_audio})"
        except Exception as exc:  # keep the app usable with audio only
            asr = text_stack = None
            st.warning(f"The text cross-check could not be loaded, so this run uses audio only. Reason: {exc}")

    roles = [("Customer", customer)] if customer is not None else []
    roles.append(("Agent", agent))
    counts = [len(make_windows(turns, spk, win=win)[0]) for _, spk in roles]
    steps = 2 if asr is not None else 1
    total = max(1, sum(counts) * steps)
    progress_bar = st.progress(0.0)
    results = {}
    base = 0
    for (role, spk), n in zip(roles, counts):
        results[role] = analyze_speaker(samples, turns, spk, role, models, win, threshold, progress_bar, base, total, asr, text_stack)
        base += n * steps
    progress_bar.progress(1.0)
    return results, text_mode


def run_ai_report(results):
    """Generate the AI Call Report. Keeps the report, or the reason there is none, in the session."""
    st.session_state.pop("ai_report", None)
    st.session_state.pop("ai_report_note", None)
    if not any("transcript" in r["df"].columns for r in results.values()):
        st.session_state["ai_report_note"] = (
            "info", "The AI Call Report needs the transcript, so it is available only when speech-to-text is on.")
        return
    transcript = build_transcript(results)
    if not transcript:
        st.session_state["ai_report_note"] = ("warning", "No transcript text was produced, so there is nothing to report.")
        return
    try:
        with st.spinner(""):
            tok, llm = load_llm(LLM_NAME)
            st.session_state["ai_report"] = generate_report(transcript, make_llm(tok, llm))
    except Exception as exc:
        st.session_state["ai_report_note"] = ("error", f"The AI Call Report could not be generated. Reason: {exc}")


# ---------------------------------------------------------- page layout
SHOWN_SECTIONS = ["sec-upload"]


def section(anchor, title, subtitle, icon_name, visible=True):
    """Result sections are only drawn once there is something to show."""
    if visible:
        if anchor not in SHOWN_SECTIONS:
            SHOWN_SECTIONS.append(anchor)
        st.markdown(section_header_html(anchor, title, subtitle, icon_name), unsafe_allow_html=True)


def empty(text):
    st.markdown(empty_html(text), unsafe_allow_html=True)


def kpi_cards(items, columns=3):
    """items: (label, value, icon, unit, tooltip). Rows of equal cards."""
    for start in range(0, len(items), columns):
        for col, (label, value, icon_name, unit, tooltip) in zip(st.columns(columns), items[start:start + columns]):
            col.markdown(kpi_html(label, value, icon_name, unit, tooltip), unsafe_allow_html=True)


# ------------------------------------------------------------ upload and analysis
section("sec-upload", "Upload & Analysis", "", "upload")
with st.container(key="card-upload"):
    uploaded = st.file_uploader("Customer service call (English, WAV or MP3)", type=["wav", "mp3"])

    if uploaded is not None:
        st.audio(uploaded)

        # one click runs speaker separation, emotion analysis and the AI Call Report
        if st.button("Start analysis", type="primary", key="start-btn"):
            file_id = (uploaded.name, uploaded.size)
            for key in ("results", "ai_report", "ai_report_note", "fixed_roles", "telegram_note"):
                st.session_state.pop(key, None)
            if st.session_state.get("file_id") != file_id or "turns" not in st.session_state or st.session_state.get("live_done"):
                st.session_state.pop("live_done", None)
                for key in ("turns", "samples", "call_info"):
                    st.session_state.pop(key, None)
                with st.spinner(""):
                    samples, turns = run_diarization(uploaded)     # same file again: the speaker separation is reused
                st.session_state["samples"] = samples
                st.session_state["turns"] = turns
                st.session_state["file_id"] = file_id
                st.session_state["call_info"] = {"filename": uploaded.name, "duration": round(len(samples) / SR, 1)}
            results, text_mode = run_emotion_analysis(st.session_state["turns"], st.session_state["samples"])
            st.session_state["results"] = results
            info = st.session_state["call_info"]
            info["threshold"] = threshold
            info["win"] = win
            info["text_mode"] = text_mode
            run_ai_report(results)


# ------------------------------------------------------------ live call (phase A: connect and record)
LIVE_SERVER = os.environ.get("LIVE_SERVER_URL", "http://localhost:8765")            # where live_server.py runs (this computer)
LIVE_PUBLIC = os.environ.get("LIVE_PUBLIC_URL", LIVE_SERVER).rstrip("/")             # the address the customer can open (tunnel URL)


def live_key():
    key = os.environ.get("LIVE_ADMIN_KEY")
    if key:
        return key
    path = os.path.join(os.environ.get("LIVE_CALLS_DIR", "live_calls"), ".admin_key")
    return open(path).read().strip() if os.path.exists(path) else ""


def live_api(method, path, body=None):
    """Call the local live server. Returns (json or None, error text or None)."""
    try:
        r = requests.request(method, LIVE_SERVER + path, headers={"x-admin-key": live_key()}, json=body, timeout=130 if path.endswith("/end") else 5)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException:
        return None, "The live call server is not running. Start it in another terminal with: python live_server.py"


def live_results_from_payload(payload, threshold):
    """Turn the live server's per-chunk results into the same structure the upload flow produces."""
    results, turns = {}, []
    for role in ("Customer", "Agent"):
        rows = payload["results"][role.lower()]
        df = pd.DataFrame(rows, columns=["start", "end", "emotion", "confidence"])
        df["mid"] = (df["start"] + df["end"]) / 2
        df["time"] = df["start"].map(fmt_time) + "-" + df["end"].map(fmt_time)
        timeline, dist, pie = make_charts(df, role)
        results[role] = {"df": df, "stats": speaker_stats(df, {}), "timeline": timeline, "dist": dist, "pie": pie}
        # approximate speaking turns: neighbouring analyzed chunks of one person are one turn
        start = last = None
        for s, e in zip(df["start"], df["end"]):
            if start is not None and s - last > 0.5:
                turns.append((start, last, role.upper()))
                start = None
            start, last = (s if start is None else start), e
        if start is not None:
            turns.append((start, last, role.upper()))
    turns_df = pd.DataFrame(sorted(turns), columns=["start", "end", "speaker"])
    return results, turns_df


@st.fragment(run_every=2)
def live_monitor(room):
    """Refreshes by itself every 2 seconds while the call is running."""
    data, err = live_api("GET", f"/api/rooms/{room}/emotions")
    if err:
        st.error(err)
        return
    if data["status"].startswith("error"):
        st.error(f"The emotion model could not be started: {data['status'][7:]}")
    elif data["status"] == "starting":
        st.caption("Loading the emotion model (first call only)...")
    columns = st.columns(2)
    for col, role in zip(columns, ("customer", "agent")):
        info, rows = data["roles"][role], data["results"][role]
        last = rows[-1] if rows else None
        chip = (emotion_chip_html(last["emotion"], COLORS) + f'<span>{last["confidence"] * 100:.1f}%</span>') if last else ""
        with col:
            st.markdown(live_status_html(role.capitalize(), info["connected"], info["seconds"], chip), unsafe_allow_html=True)
            if rows:
                with st.expander(f"Recent segments ({len(rows)})"):
                    recent = pd.DataFrame(rows[-5:][::-1])
                    recent["time"] = recent["start"].map(fmt_time) + "-" + recent["end"].map(fmt_time)
                    st.markdown(predictions_table_html(recent, COLORS, include_transcript=False), unsafe_allow_html=True)
    st.caption(f"Each segment covers {data['settings']['chunk']:.0f} seconds. Labels are model predictions, not confirmed emotions.")


def add_live_transcripts(results, payload, room):
    """Transcribe each speaker's recording once and attach the text to that speaker's analyzed segments.
    Returns an error text (the analysis still works without text) or None."""
    import wave

    calls_dir = os.environ.get("LIVE_CALLS_DIR", "live_calls")
    try:
        with st.spinner(""):
            asr = load_asr(asr_size)
    except Exception as exc:
        return f"Speech-to-text could not be loaded, so there is no transcript. Reason: {exc}"
    bar = st.progress(0.0)
    done = 0
    for role, res in results.items():
        df = res["df"]
        path = os.path.join(calls_dir, room, f"{role.lower()}.wav")
        if df.empty or not os.path.exists(path):
            continue
        with wave.open(path, "rb") as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        offset = payload["roles"][role.lower()].get("offset", 0.0)             # call clock -> this person's own recording clock
        windows = [(float(s) - offset, float(e) - offset) for s, e in zip(df["start"], df["end"])]
        df["transcript"] = assign_texts_to_windows(transcribe_segments(samples, asr), windows)
        done += 1
        bar.progress(done / len(results))
    bar.progress(1.0)
    return None


section("sec-live", "Live Call", "", "phone")
with st.container(key="card-live"):
    room = st.session_state.get("live_room")
    if st.button("Create live call" if room is None else "Create a new live call", key="live-create"):
        data, err = live_api("POST", "/api/rooms", {"chunk": chunk_seconds, "threshold": threshold})
        if err:
            st.error(err)
        else:
            st.session_state["live_room"] = data["room"]
            st.session_state.pop("live_note", None)
            st.rerun()
    if room:
        customer_link = f"{LIVE_PUBLIC}/call/{room}/customer"
        agent_link = f"{LIVE_SERVER}/call/{room}/agent"
        link_customer, link_agent = st.columns(2)
        with link_customer:
            st.markdown('<div class="live-label">Customer link</div>', unsafe_allow_html=True)
            st.code(customer_link, language=None)
        with link_agent:
            st.markdown('<div class="live-label">Agent link</div>', unsafe_allow_html=True)
            st.code(agent_link, language=None)
        hints = []
        if LIVE_PUBLIC.startswith("http://localhost"):
            hints.append("The customer link only works on this computer. Set LIVE_PUBLIC_URL to your tunnel address (https) before sending it.")
        if not telegram_configured():
            hints.append("Telegram is off. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to the .env file to send the link from here.")
        if hints:
            st.caption(" ".join(hints))
        actions = st.columns([1, 1, 2])
        if telegram_configured() and actions[0].button("Send link to Telegram", key="live-telegram"):
            ok, msg = send_telegram_message(f"Please join your support call: {customer_link}")
            st.session_state["live_note"] = ("success" if ok else "error", msg)
        finish = actions[1].button("End call and analyze", type="primary", key="live-finish")
        note = st.session_state.get("live_note")
        if note:
            getattr(st, note[0])(note[1])
        live_monitor(room)
        if finish:
            done, err = live_api("POST", f"/api/rooms/{room}/end")
            payload, err2 = live_api("GET", f"/api/rooms/{room}/emotions") if not err else (None, err)
            if err or err2:
                st.error(err or err2)
            elif not any(payload["results"].values()):
                st.warning("No speech was analyzed in this call, so there is nothing to show yet.")
            else:
                results, turns_df = live_results_from_payload(payload, payload["settings"]["threshold"])
                duration = round(float(max(r["end"].max() for r in (v["df"] for v in results.values()) if len(r))), 1)
                for key in ("results", "ai_report", "ai_report_note", "samples", "turns", "call_info", "file_id", "telegram_note"):
                    st.session_state.pop(key, None)
                st.session_state["results"] = results
                st.session_state["turns"] = turns_df
                st.session_state["fixed_roles"] = ("AGENT", "CUSTOMER")
                st.session_state["live_done"] = True
                st.session_state["call_info"] = {"filename": f"Live call {room[:6]}", "duration": duration, "threshold": payload["settings"]["threshold"],
                                                 "win": payload["settings"]["chunk"], "text_mode": "off"}
                text_error = add_live_transcripts(results, payload, room)
                if text_error:
                    st.session_state["ai_report_note"] = ("warning", text_error)
                else:
                    run_ai_report(results)
                st.rerun()

# ------------------------------------------------------------ call overview and speaker roles
section("sec-overview", "Call Overview", "", "clock", visible="turns" in st.session_state)
if "turns" in st.session_state:
    turns = st.session_state["turns"]
    speakers = sorted(turns["speaker"].unique().tolist())
    st.session_state["call_info"]["n_speakers"] = len(speakers)
    kpi_cards([
        ("Recording Duration", st.session_state["call_info"]["duration"], "clock", "s", ""),
        ("Speakers Detected", len(speakers), "users", "", ""),
        ("Speaker Turns", len(turns), "repeat", "", ""),
    ])

section("sec-roles", "Speaker Roles", "", "users", visible="turns" in st.session_state)
if "turns" in st.session_state:
    fixed = st.session_state.get("fixed_roles")                 # live calls: roles are known from the link used
    agent, customer = fixed if fixed else assign_roles(turns)
    role_customer, role_agent = st.columns(2)
    role_customer.markdown(role_card_html("Customer", customer if customer is not None else "Not detected", "user"), unsafe_allow_html=True)
    role_agent.markdown(role_card_html("Agent", agent, "headset", note="" if fixed else "Spoke first"), unsafe_allow_html=True)

# ------------------------------------------------------------ emotion analytics
section("sec-analytics", "Emotion Analytics", "", "chart", visible="results" in st.session_state)
if "results" in st.session_state:
    results = st.session_state["results"]
    info = st.session_state["call_info"]
    tabs = st.tabs([f"{role} Report" for role in results])
    for tab, (role, res) in zip(tabs, results.items()):
        with tab:
            if res["timeline"] is None:
                empty(f"No speech segments were analyzed for the {role.lower()}.")
                continue
            s = res["stats"]
            ag = agreement_stats(res["df"])
            cov = classification_coverage(res["df"])
            kpi_cards([
                ("Analyzed Segments", s["segments"], "grid", "", "Number of analyzed speech windows."),
                ("Analyzed Speech Duration", f"{cov['analyzed_seconds']:.1f}", "clock", "s",
                 "Total time of the analyzed segments (overlaps counted once)."),
                ("Classification Coverage (% of time)", format_coverage(cov), "target", "",
                 "Share of the analyzed speech TIME that received an emotion label at or above the confidence threshold. "
                 "It is not model accuracy."),
                ("Uncertain (% of segments)", s["uncertain_pct"], "help", "%",
                 "Share of analyzed SEGMENTS (counted by number of segments, not by time) below the confidence threshold."),
                ("Most Frequent Label (by segments)", s["top"] if s["top"] else "n/a", "tag", "",
                 "The confident label that appears in the most segments."),
                ("Audio-Text Agreement (% of segments)",
                 ag["agree_pct"] if ag and ag["agree_pct"] is not None else "n/a", "link",
                 "%" if ag and ag["agree_pct"] is not None else "",
                 "Share of segments with a transcript where the audio model and the text model give the same top emotion."),
            ])
            st.write("")
            with st.container(key=f"card-timeline-{role.lower()}"):
                st.plotly_chart(res["timeline"])
            st.write("")
            col_bar, col_pie = st.columns(2)
            with col_bar:
                with st.container(key=f"card-bar-{role.lower()}"):
                    st.plotly_chart(res["dist"])
            with col_pie:
                with st.container(key=f"card-pie-{role.lower()}"):
                    st.plotly_chart(res["pie"])

# ------------------------------------------------------------ segment-level predictions
section("sec-predictions", "Segment-Level Predictions", "", "list", visible="results" in st.session_state)
if "results" in st.session_state:
    tabs = st.tabs([f"{role} Report" for role in results])
    for tab, (role, res) in zip(tabs, results.items()):
        with tab:
            df = res["df"]
            if df.empty:
                empty(f"No speech segments were analyzed for the {role.lower()}.")
                continue
            with st.container(key=f"card-table-{role.lower()}"):
                st.markdown(f'<p class="card-title">{html.escape(role)}: {len(df)} analyzed segments</p>', unsafe_allow_html=True)
                st.markdown(predictions_table_html(df, COLORS, include_transcript="transcript" in df.columns), unsafe_allow_html=True)

# ------------------------------------------------------------ AI call report
section("sec-report", "AI Call Report", "", "sparkles", visible="results" in st.session_state)
if "results" in st.session_state:
    report = st.session_state.get("ai_report")
    note = st.session_state.get("ai_report_note")
    if report:
        show_ai_report(report)
    if note:
        getattr(st, note[0])(note[1])
    failed = (note is not None and note[0] == "error") or (report is not None and (report["meta"]["parse"] == "failed" or report["meta"]["empty"]))
    if failed and st.button("Try the AI Call Report again"):
        run_ai_report(results)
        st.rerun()

# ------------------------------------------------------------ export
section("sec-export", "Export Report", "", "download", visible="results" in st.session_state)
if "results" in st.session_state:
    report_html = build_report_html(
        info,
        results,
        include_transcript=include_transcript,
        ai_report=st.session_state.get("ai_report"),
    )
    with st.container(key="card-export"):
        download_col, telegram_col, _ = st.columns([1.5, 2, 2.5])
        with download_col:
            st.download_button(
                "Download report",
                data=report_html,
                type="primary",
                key="download-btn",
                file_name="call_emotion_report.html",
                mime="text/html",
            )
        if telegram_configured():
            if telegram_col.button("Send summary to Telegram", key="telegram-summary"):
                ok, msg = send_telegram_message(build_call_summary_text(info, results, st.session_state.get("ai_report")))
                if ok:
                    ok, msg = send_telegram_document(report_html.encode("utf-8"), "call_emotion_report.html", "Full report (open it in a browser).")
                    msg = "Summary and report sent to Telegram." if ok else f"The summary was sent, but the report file was not. {msg}"
                st.session_state["telegram_note"] = ("success" if ok else "error", msg)
        note = st.session_state.get("telegram_note")
        if note and telegram_configured():
            getattr(st, note[0])(note[1])

nav_slot.markdown(sidebar_nav_html(SHOWN_SECTIONS), unsafe_allow_html=True)