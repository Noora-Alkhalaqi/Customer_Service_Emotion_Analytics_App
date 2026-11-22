"""Helper functions for the call emotion analytics app (heavy ML imports are lazy)."""
import html
import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import ui_theme as ui

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
UNCERTAIN = "uncertain"
NO_TEXT = "no text"
ORDER = EMOTIONS + [UNCERTAIN]
COLORS = {
    "anger": "#d62728",
    "disgust": "#8c564b",
    "fear": "#9467bd",
    "joy": "#2ca02c",
    "neutral": "#7f7f7f",
    "sadness": "#1f77b4",
    "surprise": "#ff7f0e",
    UNCERTAIN: "#c7c7c7",
    NO_TEXT: "#e6e6e6",
}


def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ----------------------------------------------------------------- windows
def make_windows(turns, speaker, win=4.0, min_len=1.0, max_overlap=0.2):
    """Cut one speaker's turns into short windows.

    - turns shorter than min_len are skipped
    - a long turn is split into equal pieces of at most `win` seconds
    - a window is skipped if more than max_overlap of it overlaps the other speaker
    """
    mine = turns[turns["speaker"] == speaker][["start", "end"]].values.tolist()
    others = turns[turns["speaker"] != speaker][["start", "end"]].values.tolist()
    windows = []
    skipped_short = 0
    skipped_overlap = 0
    for s, e in mine:
        length = e - s
        if length < min_len:
            skipped_short += 1
            continue
        n = max(1, math.ceil(length / win))
        size = length / n
        for i in range(n):
            w_start = s + i * size
            w_end = w_start + size
            overlap = sum(max(0.0, min(w_end, oe) - max(w_start, os_)) for os_, oe in others)
            if overlap / size > max_overlap:
                skipped_overlap += 1
            else:
                windows.append((w_start, w_end))
    return windows, {"skipped_short": skipped_short, "skipped_overlap": skipped_overlap}


def assign_roles(turns):
    """Roles from the diarization turns: the speaker who talks first is the Agent, the other one the Customer.

    Returns (agent, customer); customer is None when only one speaker was detected.
    """
    ordered = turns.sort_values(["start", "end"])
    agent = ordered.iloc[0]["speaker"]
    others = [s for s in dict.fromkeys(ordered["speaker"]) if s != agent]
    return agent, (others[0] if others else None)


# ------------------------------------------------------------------ labels
def decide_labels(probs, classes, threshold):
    """probs: (n, n_classes). Returns labels (with 'uncertain') and confidences."""
    probs = np.asarray(probs)
    best = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    labels = [classes[i] if c >= threshold else UNCERTAIN for i, c in zip(best, conf)]
    return labels, conf


def build_results_table(windows, probs, classes, threshold):
    labels, conf = decide_labels(probs, classes, threshold)
    df = pd.DataFrame(
        {
            "start": [round(w[0], 2) for w in windows],
            "end": [round(w[1], 2) for w in windows],
            "emotion": labels,
            "confidence": np.round(conf, 3),
        }
    )
    df["mid"] = (df["start"] + df["end"]) / 2
    df["time"] = df["start"].map(fmt_time) + "-" + df["end"].map(fmt_time)
    for i, c in enumerate(classes):
        df[f"p_{c}"] = np.round(np.asarray(probs)[:, i], 3)
    return df


def build_multimodal_table(windows, audio_probs, classes, threshold, texts=None, text_probs=None, w_audio=0.4):
    """Audio-only table, or audio + text table.

    `emotion` / `confidence` are the FINAL label: fused (w_audio * audio + (1 - w_audio) * text)
    when the segment has a transcript, audio only otherwise.
    """
    audio_probs = np.asarray(audio_probs)
    if texts is None:
        return build_results_table(windows, audio_probs, classes, threshold)
    text_probs = np.asarray(text_probs)
    has_text = np.array([bool(str(t).strip()) for t in texts])
    fused = np.where(has_text[:, None], w_audio * audio_probs + (1.0 - w_audio) * text_probs, audio_probs)
    df = build_results_table(windows, fused, classes, threshold)
    a_lab, a_conf = decide_labels(audio_probs, classes, threshold)
    t_lab, t_conf = decide_labels(text_probs, classes, threshold)
    agree = []
    for i in range(len(windows)):
        if not has_text[i]:
            agree.append("n/a")
        else:
            agree.append("yes" if audio_probs[i].argmax() == text_probs[i].argmax() else "no")
    df["audio_emotion"] = a_lab
    df["audio_confidence"] = np.round(a_conf, 3)
    df["transcript"] = [str(t).strip() for t in texts]
    df["text_emotion"] = [lab if h else NO_TEXT for lab, h in zip(t_lab, has_text)]
    df["text_confidence"] = np.where(has_text, np.round(t_conf, 3), 0.0)
    df["agree"] = agree
    return df


# ------------------------------------------------------- stats and summary
def union_seconds(starts, ends):
    """Total time covered by a set of intervals; overlapping parts are counted once."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(zip(map(float, starts), map(float, ends))):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def classification_coverage(df):
    """Coverage for ONE speaker's result table (not model accuracy).

    coverage % = time of segments that got one of the 7 emotion labels (confidence >= threshold)
                 / total analyzed speech time * 100
    'uncertain' segments stay in the denominator. Overlapping segments are counted once.
    Returns coverage_pct = None when there is no analyzed speech (shown as N/A).
    """
    if df.empty:
        return {"analyzed_seconds": 0.0, "classified_seconds": 0.0, "coverage_pct": None}
    classified = df[df["emotion"].isin(EMOTIONS)]
    analyzed = union_seconds(df["start"], df["end"])
    done = union_seconds(classified["start"], classified["end"])
    pct = 100.0 * done / analyzed if analyzed > 0 else None
    return {"analyzed_seconds": analyzed, "classified_seconds": done, "coverage_pct": pct}


def format_coverage(cov):
    return "N/A" if cov["coverage_pct"] is None else f"{cov['coverage_pct']:.1f}%"


def speaker_stats(df, skipped):
    if df.empty:
        return {"segments": 0, "seconds": 0.0, "uncertain_pct": 0.0, "top": None, **skipped}
    confident = df[df["emotion"] != UNCERTAIN]["emotion"]
    top = confident.value_counts().index[0] if len(confident) else None
    return {
        "segments": int(len(df)),
        "seconds": round(union_seconds(df["start"], df["end"]), 1),
        "uncertain_pct": round(100.0 * float((df["emotion"] == UNCERTAIN).mean()), 1),
        "top": top,
        **skipped,
    }


def agreement_stats(df):
    """Agreement between the audio and the text model (top prediction), or None without text."""
    if df.empty or "agree" not in df.columns:
        return None
    comparable = df[df["agree"] != "n/a"]
    if comparable.empty:
        return {"comparable": 0, "agree_pct": None, "with_text": 0}
    return {
        "comparable": int(len(comparable)),
        "agree_pct": round(100.0 * float((comparable["agree"] == "yes").mean()), 1),
        "with_text": int((df["transcript"].str.len() > 0).sum()),
    }


def style_figure(fig, height):
    """Lavender AI look for every chart. Emotion colors are set on the traces and are not touched here."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=ui.FONT, size=13, color=ui.NAVY),
        title=dict(font=dict(size=16, color=ui.NAVY), x=0.0, xanchor="left"),
        hoverlabel=dict(bgcolor="white", bordercolor=ui.BORDER, font=dict(family=ui.FONT, color=ui.NAVY)),
    )
    return fig


def tick_step(span):
    """A readable tick interval (seconds) for a time axis of `span` seconds."""
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800):
        if span / step <= 10:
            return step
    return 3600


# ------------------------------------------------------------------ charts
def make_strip(df, role):
    """One horizontal strip on a real time axis: each analyzed segment is a block from its start to its end.

    Gaps (turns of the other speaker, skipped short or overlapping parts) stay empty.
    """
    d = df.sort_values("start").reset_index(drop=True)
    fig = go.Figure()
    seen = set()
    for _, r in d.iterrows():
        emo = r["emotion"]
        fig.add_trace(
            go.Bar(
                x=[float(r["end"] - r["start"])],
                base=[float(r["start"])],
                y=[role],
                orientation="h",
                name=emo.capitalize(),
                legendgroup=emo,
                showlegend=emo not in seen,
                marker=dict(color=COLORS.get(emo, "#c7c7c7"), line=dict(color="white", width=1.5)),
                hovertemplate=f"{r['time']}<br>{emo} (confidence {float(r['confidence']) * 100:.1f}%)<br>{r['end'] - r['start']:.1f} s<extra></extra>",
            )
        )
        seen.add(emo)
    lo, hi = float(d["start"].min()), float(d["end"].max())
    step = tick_step(hi - lo)
    first = int(lo // step) * step
    ticks = [t for t in range(first, int(hi) + step + 1, step) if lo - step <= t <= hi + step]
    pad = max(0.5, (hi - lo) * 0.01)
    fig.update_layout(
        title=f"{role}: predicted emotion through the call",
        barmode="overlay",
        bargap=0.0,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.5, x=0, title_text=""),
        xaxis=dict(
            tickmode="array",
            tickvals=ticks,
            ticktext=[fmt_time(t) for t in ticks],
            range=[lo - pad, hi + pad],
            showgrid=True,
            gridcolor=ui.BORDER,
            zeroline=False,
            title=dict(text="Time in the call (mm:ss)", font=dict(size=12, color=ui.MUTED)),
        ),
        yaxis=dict(visible=False),
    )
    return style_figure(fig, 250)


CHART_EMOTIONS = ["anger", "joy", "sadness", "neutral", "fear", "disgust", "surprise"]
CHART_LABELS = CHART_EMOTIONS + [UNCERTAIN]   # 'uncertain' is shown as its own grey category


def emotion_durations(df):
    """Total seconds per label for ONE speaker's result table.

    - duration of a segment = end - start
    - every analyzed segment counts, including those below the confidence threshold ('uncertain')
    - percent = label seconds / analyzed seconds * 100 (sums to 100)
    """
    length = df["end"] - df["start"]
    seconds = length.groupby(df["emotion"]).sum()
    seconds = seconds.reindex(CHART_LABELS, fill_value=0.0).astype(float)
    total = float(seconds.sum())
    percent = seconds / total * 100.0 if total > 0 else seconds * 0.0
    return {
        "seconds": seconds,
        "percent": percent,
        "analyzed_seconds": total,
        "uncertain_seconds": float(seconds[UNCERTAIN]),
    }


def make_duration_chart(df, role):
    """Bar chart: total seconds of each emotion (and of 'uncertain') for one speaker."""
    secs = emotion_durations(df)["seconds"]
    fig = go.Figure(
        go.Bar(
            x=[e.capitalize() for e in CHART_LABELS],
            y=secs.values,
            marker_color=[COLORS[e] for e in CHART_LABELS],
            text=[f"{s:.1f} s" for s in secs.values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=ui.NAVY),
            hovertemplate="%{x}: %{y:.1f} s<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{role}: emotion duration",
        yaxis=dict(title=dict(text="Total duration (seconds)", standoff=12), range=[0, max(1.0, float(secs.max())) * 1.25],
                   gridcolor=ui.BORDER, zeroline=False, automargin=True),
        xaxis=dict(title=dict(text="Emotion", standoff=8), showgrid=False, automargin=True),
        margin=dict(l=70, r=20, t=50, b=40),
        showlegend=False,
    )
    return style_figure(fig, 400)


def make_percent_chart(df, role):
    """Donut chart: share of the speaker's analyzed speech time for each emotion (and 'uncertain')."""
    d = emotion_durations(df)
    shown = [e for e in CHART_LABELS if d["seconds"][e] > 0]
    fig = go.Figure(
        go.Pie(
            labels=[e.capitalize() for e in shown],
            values=[float(d["seconds"][e]) for e in shown],
            hole=0.58,
            marker=dict(colors=[COLORS[e] for e in shown], line=dict(color="white", width=2)),
            text=[f"{d['percent'][e]:.1f}%" for e in shown],
            textinfo="text",
            textposition="inside",
            insidetextorientation="horizontal",
            sort=False,
            direction="clockwise",
            hovertemplate="%{label}: %{value:.1f} s (%{text})<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{role}: share of speaking time by emotion",
        margin=dict(l=20, r=20, t=50, b=90),
        showlegend=True,
        legend=dict(orientation="v", y=0.5, x=1.0, xanchor="left", font=dict(size=13)),
        annotations=[
            dict(text=f"<b>{d['analyzed_seconds']:.1f} s</b><br><span style='font-size:12px;color:{ui.MUTED}'>analyzed</span>",
                 x=0.5, y=0.5, showarrow=False, font=dict(size=22, color=ui.NAVY)),
        ],
    )
    return style_figure(fig, 400)


def make_charts(df, role):
    """Returns (timeline strip, duration bar chart, percentage pie chart) for one speaker."""
    if df.empty:
        return None, None, None
    return make_strip(df, role), make_duration_chart(df, role), make_percent_chart(df, role)


# ------------------------------------------------- speech-to-text and text
def transcribe_windows(samples, windows, asr, sr=16000, progress=None):
    """Transcribe each window with a faster-whisper model. Returns one string per window."""
    texts = []
    for i, (s, e) in enumerate(windows):
        audio = samples[int(s * sr): int(e * sr)]
        segments, _ = asr.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
        parts = [seg.text.strip() for seg in segments if getattr(seg, "no_speech_prob", 0.0) < 0.6]
        texts.append(" ".join(p for p in parts if p))
        if progress is not None:
            progress(i + 1)
    return texts


def transcribe_segments(samples, asr, sr=16000):
    """Transcribe one speaker's whole recording once. Returns (start, end, word) for every spoken word, so a long sentence
    can be shared between the analysis windows it covers. Silence is skipped by the voice-activity filter."""
    def run(**extra):
        segments, _ = asr.transcribe(samples, language="en", beam_size=1, condition_on_previous_text=False,
                                     word_timestamps=True, **extra)
        items = []
        for seg in segments:
            text = seg.text.strip()
            if not text or getattr(seg, "no_speech_prob", 0.0) >= 0.6:
                continue
            words = getattr(seg, "words", None)
            if words:
                items += [(float(w.start), float(w.end), w.word.strip()) for w in words if w.word.strip()]
            else:                                           # no word times: spread the words evenly over the segment
                tokens = text.split()
                step = (float(seg.end) - float(seg.start)) / len(tokens)
                items += [(float(seg.start) + i * step, float(seg.start) + (i + 1) * step, t) for i, t in enumerate(tokens)]
        return items
    try:
        return run(vad_filter=True)
    except TypeError:                                       # a Whisper wrapper without the vad_filter option
        return run()


def assign_texts_to_windows(segments, windows):
    """Give every spoken word to the analysis window that contains its midpoint. Returns one string per window."""
    texts = [[] for _ in windows]
    for start, end, text in segments:
        mid = (start + end) / 2
        for i, (ws, we) in enumerate(windows):
            if ws <= mid < we or (i == len(windows) - 1 and mid >= we and mid - we < 1.0):
                texts[i].append(text)
                break
    return [" ".join(t) for t in texts]


def encode_texts(texts, tok, enc_model, device, batch_size=16):
    """Mean-pooled, L2-normalised sentence embeddings (same recipe used for training)."""
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            h = enc_model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        e = torch.nn.functional.normalize(e, dim=1)
        out.append(e.cpu().numpy())
    return np.vstack(out)


def text_probabilities(texts, stack, classes):
    """stack = (tokenizer, encoder, trained text classifier, device). Empty texts get zero rows."""
    tok, enc_model, tclf, device = stack
    out = np.zeros((len(texts), len(classes)), dtype=np.float32)
    idx = [i for i, t in enumerate(texts) if str(t).strip()]
    if idx:
        X = encode_texts([texts[i] for i in idx], tok, enc_model, device)
        P = tclf.predict_proba(X)
        tc = [str(c) for c in tclf.classes_]
        order = [tc.index(c) for c in classes]
        out[idx] = P[:, order]
    return out


# ------------------------------------------------------------ AI call report
REPORT_FIELDS = [
    "call_overview",
    "reason_for_contact",
    "key_discussion_points",
    "customer_requests",
    "agent_actions",
    "outcome",
    "important_details",
]
LIST_FIELDS = {"key_discussion_points", "customer_requests", "agent_actions", "important_details"}
NOT_MENTIONED = "Not mentioned"
MAX_CHARS_PER_PART = 10000      # longer transcripts are processed in several parts, never cut
MAX_LIST_ITEMS = 12

REPORT_SYSTEM = (
    "You write factual reports of customer service calls. You receive an automatic transcript (speech recognition, "
    "it may contain errors and may be incomplete). Every line starts with the speaker role, either 'Customer:' or "
    "'Agent:'. Use these labels exactly and never swap them.\n\n"
    "Return ONE JSON object and nothing else, with exactly these keys:\n"
    '- "call_overview": string, 3 to 5 sentences: the main issue, what was discussed and what happened in the call.\n'
    '- "reason_for_contact": string: why the Customer contacted support (list every reason if there are several).\n'
    '- "key_discussion_points": array of short strings with the most important information discussed.\n'
    '- "customer_requests": array of strings: only what the Customer asked for.\n'
    '- "agent_actions": array of strings: only what the Agent did or said they did. Start each item with '
    '"Completed:" (done or confirmed during the call), "Promised:" (said they will do it) or "Discussed:" '
    "(only talked about or offered).\n"
    '- "outcome": string: what the transcript shows happened at the end of the call, and whether a speaker explicitly '
    "confirmed that the issue was resolved. If the call ends without a stated outcome, say that.\n"
    '- "important_details": array of strings: order or reference numbers, dates, deadlines, amounts and product names '
    "that were said. Never include full card numbers, passwords, addresses, phone numbers or email addresses.\n\n"
    "Rules:\n"
    "- Use only information that is in the transcript. Never invent or guess details.\n"
    '- If a section has no information, use "Not mentioned" for a string and [] for an array.\n'
    "- Do not give advice, recommendations or next steps. Do not decide whether the problem is resolved; only report "
    "what the speakers said.\n"
    "- Do not describe or judge anyone's emotions, satisfaction or performance.\n"
    "- Write in the third person, in plain English. Refer to the speakers as the Customer and the Agent."
)
REPORT_TITLES = [
    ("Call Overview", "call_overview"),
    ("Reason for Contact", "reason_for_contact"),
    ("Key Discussion Points", "key_discussion_points"),
    ("Customer Requests", "customer_requests"),
    ("Actions Taken by Agent", "agent_actions"),
    ("Outcome / Resolution", "outcome"),
    ("Important Details", "important_details"),
]
JSON_PREFILL = '{\n  "call_overview": "'      # the model continues from here, which keeps the answer valid JSON


def build_transcript(results):
    """Full dialogue ordered by time. Speaker labels are the roles chosen in the app (Customer / Agent)."""
    rows = []
    for role, res in results.items():
        df = res["df"]
        if df.empty or "transcript" not in df.columns:
            continue
        for start, text in zip(df["start"], df["transcript"]):
            if str(text).strip():
                rows.append((float(start), role, str(text).strip()))
    rows.sort(key=lambda r: r[0])
    lines = []
    for _, role, text in rows:
        if lines and lines[-1][0] == role:
            lines[-1][1] += " " + text
        else:
            lines.append([role, text])
    return "\n".join(f"{role}: {text}" for role, text in lines)


def make_llm(tok, model):
    """Wrap a Hugging Face chat model as llm(system, user, prefill, max_new_tokens) -> generated text."""

    def call(system, user, prefill="", max_new_tokens=800):
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return call


# ---- reading the model answer safely
def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```\s*$", "", text)


def parse_json_object(text):
    """Return a dict from the model text, repairing a truncated answer when possible; None if impossible."""
    text = _strip_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    cand = text[start:]
    depth, in_str, esc, end = 0, False, False, None
    for i, ch in enumerate(cand):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    attempts = [cand[: end + 1]] if end is not None else []
    for quote in ("", '"'):                      # the answer was cut off: close the string, array and object
        for tail in ("", "]", "}", "]}"):
            attempts.append(cand + quote + tail)
    for attempt in attempts:
        attempt = re.sub(r",\s*([\]}])", r"\1", attempt)
        try:
            obj = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def salvage_fields(text):
    """Last resort when the JSON is broken: pick out the fields that can still be read."""
    out = {}
    for key in REPORT_FIELDS:
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % key, text)
        if m and key not in LIST_FIELDS:
            out[key] = m.group(1)
        m = re.search(r'"%s"\s*:\s*\[(.*?)\]' % key, text, flags=re.S)
        if m and key in LIST_FIELDS:
            out[key] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return out


# ---- cleaning and checking the content
_EMPTY = {"", "n/a", "na", "none", "null", "unknown", "not mentioned", "not available", "not specified", "-"}
_WORD_NUMBERS = {w: str(n) for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORD_NUMBERS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                      "eighty": "80", "ninety": "90", "hundred": "100", "first": "1", "second": "2", "third": "3"})


def _clean_text(value):
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip(" \t\n-*•")
    return "" if text.lower().strip(" .") in _EMPTY else text


def _number_tokens(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    tokens = set(re.findall(r"\d+", text))
    tokens.update(_WORD_NUMBERS[w] for w in re.findall(r"[a-z]+", text) if w in _WORD_NUMBERS)
    return tokens


def _supported(text, transcript_numbers):
    """True when every number in `text` also appears in the transcript."""
    return all(n in transcript_numbers for n in re.findall(r"\d+", re.sub(r"(?<=\d),(?=\d)", "", text)))


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def redact_sensitive(text):
    return _CARD.sub("[number removed]", _EMAIL.sub("[email removed]", text))


_CUSTOMER_SUBJECT = re.compile(     # the Customer is the one doing something: not an Agent action
    r"^(?:the\s+)?customer\s+(?:asked|requested|wants?|wanted|said|says|stated|reported|complained|called|checked|"
    r"provided|gave|mentioned|explained|thanked|agreed|confirmed)\b", re.I)
_AGENT_SUBJECT = re.compile(r"^(?:the\s+)?agent\b", re.I)


def normalize_report(obj, transcript):
    """Make a model answer safe to display. Returns (report, counts of removed items)."""
    numbers = _number_tokens(transcript)
    removed = {"unsupported": 0, "misattributed": 0}
    report = {}
    for key in REPORT_FIELDS:
        raw = obj.get(key) if isinstance(obj, dict) else None
        if key in LIST_FIELDS:
            if isinstance(raw, str):
                raw = re.split(r"\n+|\s*[•]\s*", raw)
            items, seen = [], set()
            for item in (raw if isinstance(raw, (list, tuple)) else []):
                text = _clean_text(item)
                if not text or text.lower() in seen:
                    continue
                if not _supported(text, numbers):
                    removed["unsupported"] += 1
                    continue
                if (key == "agent_actions" and _CUSTOMER_SUBJECT.match(re.sub(r"^(Completed|Promised|Discussed):\s*", "", text))) or \
                        (key == "customer_requests" and _AGENT_SUBJECT.match(text)):
                    removed["misattributed"] += 1
                    continue
                seen.add(text.lower())
                items.append(redact_sensitive(text))
            report[key] = items[:MAX_LIST_ITEMS]
        else:
            text = _clean_text(raw)
            kept = [s for s in _sentences(text) if _supported(s, numbers)]
            removed["unsupported"] += len(_sentences(text)) - len(kept)
            text = " ".join(kept)
            report[key] = redact_sensitive(text) if text else NOT_MENTIONED
    return report, removed


def split_transcript(transcript, max_chars=MAX_CHARS_PER_PART):
    """Split at speaker turns into parts of at most max_chars; no text is dropped."""
    if len(transcript) <= max_chars:
        return [transcript]
    pieces = []
    for line in transcript.split("\n"):
        while len(line) > max_chars:                       # a very long turn is cut at a sentence end if possible
            cut = line.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 0 else max_chars
            pieces.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            pieces.append(line)
    parts, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) > max_chars:
            parts.append(cur)
            cur = piece
        else:
            cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        parts.append(cur)
    return parts


def _report_for_part(part, llm, note=""):
    """One model call (plus one retry when nothing readable came back). Returns (report, removed, how)."""
    user = f"{note}Transcript:\n{part}\n\nReturn the JSON object now."
    for attempt in range(2):
        raw = JSON_PREFILL + llm(REPORT_SYSTEM, user, JSON_PREFILL, 800)
        obj = parse_json_object(raw)
        how = "json"
        if obj is None:
            obj, how = salvage_fields(raw), "partial"
        present = sum(1 for k in REPORT_FIELDS if k in obj)
        if present >= 2:                                   # a refusal or chatter gives at most one field: not a report
            if present < len(REPORT_FIELDS):
                how = "partial"
            report, removed = normalize_report(obj, part)
            return report, removed, how
        user += "\n\nYour previous answer was not valid JSON. Answer with one valid JSON object only."
    report, removed = normalize_report({}, part)
    return report, removed, "failed"


def _merge_lists(reports, key):
    out, seen = [], set()
    for r in reports:
        for item in r[key]:
            if item.lower() not in seen:
                seen.add(item.lower())
                out.append(item)
    return out[: MAX_LIST_ITEMS * 2]


def generate_report(transcript, llm, max_chars=MAX_CHARS_PER_PART):
    """Structured AI call report from the speaker-labelled transcript.

    Returns {"report": {7 fields}, "meta": {...}}. Nothing here changes any emotion model or result.
    """
    parts = split_transcript(transcript, max_chars)
    results, removed_total, hows = [], {"unsupported": 0, "misattributed": 0}, []
    for i, part in enumerate(parts):
        note = "" if len(parts) == 1 else f"This is part {i + 1} of {len(parts)} of one call. Report only this part.\n\n"
        report, removed, how = _report_for_part(part, llm, note)
        results.append(report)
        hows.append(how)
        for k in removed_total:
            removed_total[k] += removed[k]
    if len(results) == 1:
        final = results[0]
    else:
        numbers = _number_tokens(transcript)
        reasons = [r["reason_for_contact"] for r in results if r["reason_for_contact"] != NOT_MENTIONED]
        outcomes = [r["outcome"] for r in results if r["outcome"] != NOT_MENTIONED]
        overviews = [r["call_overview"] for r in results if r["call_overview"] != NOT_MENTIONED]
        final = {key: _merge_lists(results, key) for key in LIST_FIELDS}
        final["reason_for_contact"] = " ".join(dict.fromkeys(reasons)) or NOT_MENTIONED
        final["outcome"] = outcomes[-1] if outcomes else NOT_MENTIONED      # the end of the call comes last
        final["call_overview"] = " ".join(overviews) or NOT_MENTIONED
        if len(overviews) > 1:                                              # fuse the part overviews into one
            joined = "\n".join(f"Part {i + 1}: {o}" for i, o in enumerate(overviews))
            text = _clean_text(llm("Combine the part summaries of ONE customer service call into a single overview "
                                   "of 3 to 5 sentences in the third person. Use only the information given. Do not "
                                   "add advice or new details.", joined + "\n\nWrite the overview now.", "", 250))
            kept = [s for s in _sentences(text) if _supported(s, numbers)]
            if kept:
                final["call_overview"] = redact_sensitive(" ".join(kept))
    empty = all((not final[k]) if k in LIST_FIELDS else final[k] == NOT_MENTIONED for k in REPORT_FIELDS)
    return {
        "report": final,
        "meta": {
            "parts": len(parts),
            "parse": "failed" if all(h == "failed" for h in hows) else ("json" if all(h == "json" for h in hows) else "partial"),
            "removed_unsupported": removed_total["unsupported"],
            "removed_misattributed": removed_total["misattributed"],
            "empty": empty,
        },
    }


# ------------------------------------------------------------------ report
def build_call_summary_text(call_info, results, ai_report=None):
    """Short plain-text summary of one analyzed call (numbers + the AI overview). It contains NO transcript lines."""
    lines = ["Call analysis summary", f"File: {call_info.get('filename', '')}", f"Duration: {call_info.get('duration', '')} s", ""]
    for role, res in results.items():
        st_, df = res["stats"], res["df"]
        cov = format_coverage(classification_coverage(df))
        top = st_["top"] if st_["top"] else "n/a"
        lines.append(f"{role}: {st_['segments']} segments, {st_['seconds']} s of speech")
        lines.append(f"  Most frequent label: {top} | Uncertain: {st_['uncertain_pct']}% of segments | Classification coverage: {cov}")
    if ai_report and ai_report["meta"]["parse"] != "failed" and not ai_report["meta"]["empty"]:
        rep = ai_report["report"]
        lines += ["", "AI Call Report", f"Overview: {rep['call_overview']}", f"Reason for contact: {rep['reason_for_contact']}", f"Outcome: {rep['outcome']}"]
    lines += ["", "Emotion labels are model predictions, not confirmed emotions."]
    text = "\n".join(lines)
    return text if len(text) <= 3900 else text[:3890].rstrip() + "..."


def build_report_html(call_info, results, include_transcript=False, ai_report=None):
    """Self-contained HTML report (Lavender AI identity, A4 print CSS). The numbers are the ones shown in the app."""
    e = html.escape
    fixed = lambda fig, height: go.Figure(fig).update_layout(width=620, height=height)   # fits the width of an A4 page
    row = lambda label, value: f"<dt>{e(label)}</dt><dd>{e(str(value))}</dd>"
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Customer Service Speech Emotion Report</title>",
        f"<style>{ui.REPORT_CSS}</style></head><body><div class='page'>",
        "<div class='hero'><h1>Customer Service Speech Emotion Report</h1>"
        f"<div class='meta'>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div></div>",
        "<h2 class='sec-h'>Call information</h2>",
        "<div class='card'><dl class='info'>"
        + row("File", call_info.get("filename", ""))
        + row("Duration", f"{call_info.get('duration', '')} seconds")
        + row("Speakers detected", call_info.get("n_speakers", ""))
        + row("Confidence threshold for labels", call_info.get("threshold", ""))
        + row("Analysis window length (max)", f"{call_info.get('win', '')} seconds")
        + "</dl></div>",
    ]
    first_plot = True
    for role, res in results.items():
        st_ = res["stats"]
        df = res["df"]
        cov = classification_coverage(df)
        ag = agreement_stats(df)
        parts.append(f"<h2 class='sec-h'>{e(role)} report</h2>")
        parts.append(
            "<div class='kpis'>"
            + ui.kpi_html("Analyzed Segments", st_["segments"], "grid")
            + ui.kpi_html("Analyzed Speech Duration", f"{st_['seconds']}", "clock", "s")
            + ui.kpi_html("Classification Coverage (% of time)", format_coverage(cov), "target")
            + ui.kpi_html("Uncertain (% of segments)", st_["uncertain_pct"], "help", "%")
            + ui.kpi_html("Most Frequent Label (by segments)", st_["top"] if st_["top"] else "n/a", "tag")
            + ui.kpi_html("Audio-Text Agreement (% of segments)",
                          ag["agree_pct"] if ag and ag["agree_pct"] is not None else "n/a", "link",
                          "%" if ag and ag["agree_pct"] is not None else "")
            + "</div>"
        )
        for fig, height in ((res["timeline"], 230), (res["dist"], 320), (res.get("pie"), 330)):
            if fig is not None:
                chart = fixed(fig, height).to_html(full_html=False, include_plotlyjs=True if first_plot else False,
                                           config={"displayModeBar": False, "responsive": False})
                parts.append(f"<div class='card'><div class='chart'>{chart}</div></div>")
                first_plot = False
        if not df.empty:
            parts.append("<div class='card'><h3>Segment-level predictions</h3>"
                         + ui.predictions_table_html(df, COLORS, include_transcript=include_transcript) + "</div>")
    if ai_report:
        rep = ai_report["report"]
        parts.append("<h2 class='sec-h'>AI Call Report</h2>")
        icons = {"call_overview": "file", "reason_for_contact": "message", "key_discussion_points": "list",
                 "customer_requests": "user", "agent_actions": "headset", "outcome": "check", "important_details": "info"}
        for number, (title, key) in enumerate(REPORT_TITLES, start=1):
            if key in LIST_FIELDS:
                body = ("<ul class='items'>" + "".join(f"<li>{e(i)}</li>" for i in rep[key]) + "</ul>") if rep[key] \
                    else f"<p>{NOT_MENTIONED}</p>"
            else:
                body = f"<p>{e(rep[key])}</p>"
            parts.append(f"<div class='rcard'><div class='rcard-title'><span>{ui.icon(icons[key], 17)}</span>"
                         f"{number}. {e(title)}</div>{body}</div>")
    parts.append("<div class='foot'>Customer Service Speech Emotion Analytics</div></div></body></html>")
    return "\n".join(parts) + "\n"