"""Live call receiver (phase A): a small web server that serves the call page and receives the microphone audio.

Run it next to the Streamlit app:   python live_server.py
Each call has a random room id. Each person (customer / agent) opens their own link in a browser, allows the
microphone, and the page streams 16 kHz mono audio to this server over a WebSocket.
It records one WAV per speaker, relays each person's voice to the other, and (phase B) analyzes the emotion of each speaker every few seconds.
Uses only packages Streamlit already installs (starlette, uvicorn, websockets).
"""
import asyncio
import json
import os
import re
import secrets
import time
import wave
from pathlib import Path

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from notifier import load_env_file   # reads the local .env file (LIVE_* settings)
import live_analysis

SAMPLE_RATE = 16000
ROLES = ("customer", "agent")
OTHER = {"customer": "agent", "agent": "customer"}
CALLS_DIR = Path(os.environ.get("LIVE_CALLS_DIR", "live_calls"))
PORT = int(os.environ.get("LIVE_PORT", "8765"))
ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

SILENCE_RMS = 0.01               # chunks quieter than this are not analyzed (nobody is speaking)
MIN_FLUSH_SECONDS = 1.0          # at the end of the call, a leftover piece shorter than this is dropped
ANALYZER_FACTORY = live_analysis.EmotionAnalyzer      # replaced in tests
_ANALYZER = None
MAX_SECONDS = 60 * 60            # per speaker; protects the computer's memory
MAX_ROOMS = 20


def load_admin_key():
    """The Streamlit app and this server run on the same computer and share this key (env var or a local file).
    The API (create room / status / end) needs it, so people who only have a call link cannot use the API."""
    key = os.environ.get("LIVE_ADMIN_KEY")
    if key:
        return key
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    path = CALLS_DIR / ".admin_key"
    if not path.exists():
        path.write_text(secrets.token_urlsafe(24))
    return path.read_text().strip()


ADMIN_KEY = load_admin_key()


def authorized(request):
    return secrets.compare_digest(request.headers.get("x-admin-key", ""), ADMIN_KEY)


ROOMS = {}          # room id -> {"created": t, "roles": {role: {"pcm": bytearray, "connected": bool, "first": t}}}


def new_room(chunk=5.0, threshold=0.4):
    """Create a call room with an unguessable id."""
    room = secrets.token_urlsafe(12)
    ROOMS[room] = {
        "created": time.time(),
        "roles": {r: {"pcm": bytearray(), "connected": False, "first": None, "ws": None} for r in ROLES},
        "settings": {"chunk": chunk, "threshold": threshold},
        "results": {r: [] for r in ROLES},      # one row per analyzed chunk: start, end (seconds on the call clock), emotion, confidence
        "cursor": {r: 0 for r in ROLES},        # number of samples already analyzed
        "status": "starting",                   # starting -> ready (models loaded) | error: ...
        "ended": False,
        "task": None,
    }
    return room


def get_analyzer():
    """Load the models once per server process."""
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = ANALYZER_FACTORY()
    return _ANALYZER


def call_clock_offset(room, role):
    """Seconds between the first person joining and this person joining (puts both speakers on one time axis)."""
    firsts = [s["first"] for s in ROOMS[room]["roles"].values() if s["first"]]
    mine = ROOMS[room]["roles"][role]["first"]
    return (mine - min(firsts)) if (firsts and mine) else 0.0


async def analyze_new_audio(room, role, final=False):
    """Analyze every full chunk that has arrived since the last time (and the leftover piece when the call ends)."""
    from analysis import decide_labels            # the same label rule as the main app
    r = ROOMS[room]
    slot = r["roles"][role]
    chunk_n = int(r["settings"]["chunk"] * live_analysis.SAMPLE_RATE)
    loop = asyncio.get_running_loop()
    while True:
        total = len(slot["pcm"]) // 2
        start_i = r["cursor"][role]
        available = total - start_i
        if available >= chunk_n:
            end_i = start_i + chunk_n
        elif final and available >= int(MIN_FLUSH_SECONDS * live_analysis.SAMPLE_RATE):
            end_i = total
        else:
            return
        r["cursor"][role] = end_i
        audio = np.frombuffer(bytes(slot["pcm"][start_i * 2:end_i * 2]), dtype=np.int16).astype(np.float32) / 32768.0
        if live_analysis.rms(audio) < SILENCE_RMS:
            continue                                                    # silence: no label
        analyzer = get_analyzer()
        probs = await loop.run_in_executor(None, analyzer.predict, audio)
        labels, conf = decide_labels(np.asarray(probs)[None, :], analyzer.classes, r["settings"]["threshold"])
        offset = call_clock_offset(room, role)
        r["results"][role].append({
            "start": round(offset + start_i / live_analysis.SAMPLE_RATE, 2),
            "end": round(offset + end_i / live_analysis.SAMPLE_RATE, 2),
            "emotion": labels[0],
            "confidence": round(float(conf[0]), 3),
        })


async def analysis_loop(room):
    """Background task of one call: load the models, then analyze new audio about once per second."""
    r = ROOMS[room]
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, get_analyzer)
        r["status"] = "ready"
    except Exception as exc:                                            # e.g. model files missing
        r["status"] = f"error: {type(exc).__name__}: {exc}"
        return
    try:
        while not r["ended"]:
            for role in ROLES:
                await analyze_new_audio(room, role)
            await asyncio.sleep(1.0)
        for role in ROLES:
            await analyze_new_audio(room, role, final=True)
    except Exception as exc:
        r["status"] = f"error: {type(exc).__name__}: {exc}"


def valid(room, role):
    return bool(ROOM_RE.match(room)) and role in ROLES


def seconds_of(pcm):
    return len(pcm) / 2 / SAMPLE_RATE            # int16 mono


def save_wav(room, role):
    """Write what was received so far for one speaker to live_calls/<room>/<role>.wav."""
    pcm = ROOMS[room]["roles"][role]["pcm"]
    if not pcm:
        return None
    folder = CALLS_DIR / room
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{role}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(pcm))
    return path


PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Support call</title>
<style>
:root{--navy:#292447;--purple:#7565B5;--purple-dark:#5E4F9A;--soft:#B7A9E5;--pale:#EDE9FA;--bg:#F7F6FC;--muted:#625B79;--border:#E4E0ED;--green:#2ca02c}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--navy);font-family:Inter,"Segoe UI",system-ui,sans-serif;
display:flex;flex-direction:column;align-items:center}
header{width:100%;background:var(--navy);color:#fff;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px;border-bottom:4px solid var(--soft)}
header .brand{display:flex;align-items:center;gap:10px;font-weight:650}
header .brand svg{width:24px;height:24px;stroke:var(--soft)}
.chip{background:rgba(255,255,255,.12);color:var(--pale);border-radius:999px;padding:4px 12px;font-size:.82rem;font-weight:600}
main{width:100%;max-width:560px;padding:24px 18px 40px 18px}
.card{background:#fff;border:1px solid var(--border);border-radius:18px;padding:22px;box-shadow:0 2px 12px rgba(41,36,71,.08)}
h1{font-size:1.35rem;margin:0 0 6px 0}p{color:var(--muted);margin:6px 0;line-height:1.5}
ol.steps{margin:14px 0 4px 0;padding:0;list-style:none}
ol.steps li{display:flex;align-items:center;gap:12px;margin:10px 0;color:var(--navy)}
ol.steps b{width:28px;height:28px;border-radius:50%;background:var(--pale);color:var(--purple-dark);display:inline-flex;align-items:center;justify-content:center;font-size:.9rem;flex:none}
button{font:inherit;cursor:pointer;border:0;border-radius:12px;padding:12px 22px;font-size:1rem;font-weight:600}
.primary{background:var(--purple);color:#fff;width:100%;margin-top:14px}.primary:hover{background:var(--purple-dark)}
.primary:disabled{opacity:.55;cursor:default}
.ghost{background:#fff;color:var(--navy);border:1px solid var(--border)}.ghost:hover{background:var(--pale)}
.end{background:#B3261E;color:#fff}.end:hover{background:#8F1E18}
#msg{margin-top:12px;font-weight:600;color:var(--navy);min-height:1.4em}
#call{display:none}
.banner{display:flex;align-items:center;justify-content:center;gap:10px;border-radius:14px;padding:12px 14px;font-weight:650;margin-bottom:16px;transition:background .3s}
.banner.wait{background:var(--pale);color:var(--purple-dark)}.banner.joined{background:#E3F4E6;color:#1F6B2E}.banner.left{background:#FBE9E7;color:#8F1E18}
.banner .dot{width:10px;height:10px;border-radius:50%;background:currentColor}
.banner.wait .dot{animation:pulse 1.4s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(94,79,154,.5)}100%{box-shadow:0 0 0 10px rgba(94,79,154,0)}}
.people{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.person{background:var(--bg);border:1px solid var(--border);border-radius:16px;padding:16px 10px;text-align:center;transition:border-color .2s,box-shadow .2s}
.person.talking{border-color:var(--purple);box-shadow:0 0 0 3px rgba(117,101,181,.25)}
.person.absent{opacity:.55}
.avatar{width:64px;height:64px;border-radius:50%;background:var(--pale);color:var(--purple-dark);margin:0 auto 8px auto;display:flex;align-items:center;justify-content:center}
.avatar svg{width:30px;height:30px}
.name{font-weight:650}.state{font-size:.82rem;color:var(--muted);margin-top:2px;min-height:1.2em}
.bars{display:flex;gap:3px;justify-content:center;align-items:flex-end;height:22px;margin-top:10px}
.bars i{width:5px;border-radius:3px;background:var(--soft);height:4px;transition:height .08s}
.bottom{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:18px}
#timer{font-variant-numeric:tabular-nums;font-size:1.3rem;font-weight:700}
.controls{display:flex;gap:10px}
.small{font-size:.82rem;color:var(--muted);text-align:center;margin-top:14px}
.hidden-status{position:absolute;left:-9999px}
</style></head><body>
<header><div class="brand"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h2M6 8v8M10 4v16M14 7v10M18 9v6M22 11v2"/></svg>Support call</div>
<span class="chip" id="rolechip"></span></header>
<main>
<section class="card" id="prejoin">
<h1>Join the support call</h1><p id="who"></p>
<ol class="steps"><li><b>1</b>Use headphones to avoid echo</li><li><b>2</b>Press Join and allow the microphone</li><li><b>3</b>Wait for the other person, then talk</li></ol>
<button class="primary" id="go">Join call</button><div id="msg" role="status" aria-live="polite"></div>
</section>
<section id="call">
<div class="card">
<div class="banner wait" id="banner" role="status" aria-live="polite"><span class="dot"></span><span id="bannertext">Waiting for the other person to join...</span></div>
<div class="people">
<div class="person" id="me"><div class="avatar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/></svg></div>
<div class="name" id="myname">You</div><div class="state" id="mystate">Connected</div><div class="bars" id="mybars"><i></i><i></i><i></i><i></i><i></i></div></div>
<div class="person absent" id="other"><div class="avatar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-7 8-7s8 3 8 7"/></svg></div>
<div class="name" id="othername">Other person</div><div class="state" id="otherstate">Not joined yet</div><div class="bars" id="otherbars"><i></i><i></i><i></i><i></i><i></i></div></div>
</div>
<div class="bottom"><div><div class="small" style="margin:0;text-align:left">Call time</div><div id="timer">00:00</div></div>
<div class="controls"><button class="ghost" id="mute">Mute</button><button class="end" id="stop">End call</button></div></div>
</div>
<p class="small">Use headphones to avoid echo.</p>
</section>
<div class="hidden-status"><span id="status">Not connected</span><span id="hear"></span><i id="level"></i></div>
</main>
<script>
const parts = location.pathname.split("/");              // /call/<room>/<role>
const room = parts[2], role = parts[3];
const label = r => r === "agent" ? "Agent" : "Customer";
const otherRole = role === "agent" ? "customer" : "agent";
const $ = id => document.getElementById(id);
$("rolechip").textContent = "You are the " + label(role);
$("who").textContent = "You are joining as the " + label(role).toLowerCase() + ".";
$("myname").textContent = "You (" + label(role) + ")";
$("othername").textContent = label(otherRole);
let ws, ctx, stream, node, playAt = 0, joinedAt = 0, timerId = null, muted = false, otherPresent = false, otherTimer = null;
window.__received = 0; window.__otherPresent = false; window.__chimes = 0;

function setBars(id, level){                              // level 0..1 -> five bars
  const bars = $(id).children;
  for (let i = 0; i < bars.length; i++) bars[i].style.height = (level > i / 5 ? 6 + Math.min(1, level * 1.6) * (8 + i * 2) : 4) + "px";
}
function chime(){                                         // short two-note sound when the other person joins
  try {
    const t = ctx.currentTime;
    [660, 880].forEach((f, i) => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.frequency.value = f; o.connect(g); g.connect(ctx.destination);
      g.gain.setValueAtTime(0.0001, t + i * 0.16); g.gain.exponentialRampToValueAtTime(0.15, t + i * 0.16 + 0.03);
      g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.16 + 0.22);
      o.start(t + i * 0.16); o.stop(t + i * 0.16 + 0.25);
    });
    window.__chimes++;
  } catch (e) {}
}
function setOther(present){
  const was = otherPresent; otherPresent = present; window.__otherPresent = present;
  const b = $("banner");
  if (present) {
    b.className = "banner joined"; $("bannertext").textContent = label(otherRole) + " joined the call";
    $("other").classList.remove("absent"); $("otherstate").textContent = "Connected";
    if (!was) { chime(); startTimer(); }
  } else {
    const left = was;
    b.className = "banner " + (left ? "left" : "wait");
    $("bannertext").textContent = left ? label(otherRole) + " left the call" : "Waiting for the other person to join...";
    $("other").classList.add("absent"); $("otherstate").textContent = left ? "Left the call" : "Not joined yet"; setBars("otherbars", 0);
  }
}
function startTimer(){
  if (timerId) return; joinedAt = Date.now();
  timerId = setInterval(() => { const s = Math.floor((Date.now() - joinedAt) / 1000);
    $("timer").textContent = String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); }, 500);
}
const workletSrc = `
class Pcm extends AudioWorkletProcessor {
  constructor(){ super(); this.buf = []; this.ratio = sampleRate / 16000; this.pos = 0; }
  process(inputs){
    const ch = inputs[0][0]; if(!ch) return true;
    let peak = 0;
    for (let i = 0; i < ch.length; i++) { peak = Math.max(peak, Math.abs(ch[i])); }
    while (this.pos < ch.length) {
      const i = Math.floor(this.pos), f = this.pos - i, a = ch[i], b = ch[Math.min(i + 1, ch.length - 1)];
      this.buf.push(a + (b - a) * f); this.pos += this.ratio;
    }
    this.pos -= ch.length;
    if (this.buf.length >= 1600) {
      const out = new Int16Array(this.buf.length);
      for (let i = 0; i < out.length; i++) { const s = Math.max(-1, Math.min(1, this.buf[i])); out[i] = s < 0 ? s * 32768 : s * 32767; }
      this.port.postMessage({pcm: out.buffer, peak}, [out.buffer]); this.buf = [];
    }
    return true;
  }
}
registerProcessor("pcm", Pcm);`;
function play(buf){                                       // play the other person's audio (16 kHz int16) with a small jitter buffer
  const pcm = new Int16Array(buf), f = new Float32Array(pcm.length); let peak = 0;
  for (let i = 0; i < pcm.length; i++) { f[i] = pcm[i] / 32768; peak = Math.max(peak, Math.abs(f[i])); }
  const ab = ctx.createBuffer(1, f.length, 16000); ab.copyToChannel(f, 0);
  const src = ctx.createBufferSource(); src.buffer = ab; src.connect(ctx.destination);
  const now = ctx.currentTime;
  if (playAt < now + 0.05) playAt = now + 0.2;
  src.start(playAt); playAt += ab.duration;
  window.__received += pcm.length;
  $("hear").textContent = "Hearing the other person: " + (window.__received / 16000).toFixed(0) + " s received.";
  setBars("otherbars", Math.min(1, peak * 3)); $("other").classList.toggle("talking", peak > 0.03);
  clearTimeout(otherTimer); otherTimer = setTimeout(() => { setBars("otherbars", 0); $("other").classList.remove("talking"); }, 400);
}
$("go").onclick = async () => {
  $("go").disabled = true; $("msg").textContent = "Connecting..."; $("status").textContent = "Connecting...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: false}});
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([workletSrc], {type: "application/javascript"})));
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/" + room + "/" + role);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      node = new AudioWorkletNode(ctx, "pcm");
      node.port.onmessage = e => { if (ws.readyState === 1) ws.send(e.data.pcm);
        const lvl = muted ? 0 : Math.min(1, e.data.peak * 3); $("level").style.width = (lvl * 100) + "%"; setBars("mybars", lvl); $("me").classList.toggle("talking", lvl > 0.1); };
      ctx.createMediaStreamSource(stream).connect(node);
      $("status").textContent = "Connected - you can speak"; $("prejoin").style.display = "none"; $("call").style.display = "block";
      $("go").style.display = "none";
    };
    ws.onmessage = e => {
      if (typeof e.data === "string") { try { const m = JSON.parse(e.data); if (m.type === "presence") setOther(!!m.other); } catch (err) {} }
      else play(e.data);
    };
    ws.onclose = () => {
      $("status").textContent = "Call ended"; clearInterval(timerId); timerId = null;
      $("banner").className = "banner left"; $("bannertext").textContent = "Call ended"; $("stop").style.display = "none"; $("mute").style.display = "none";
      $("me").classList.remove("talking"); setBars("mybars", 0);
    };
    ws.onerror = () => { $("msg").textContent = "Connection failed. Please try again."; $("go").disabled = false; };
  } catch (err) { $("msg").textContent = "Microphone not available: " + err.message; $("status").textContent = "Microphone not available"; $("go").disabled = false; }
};
$("mute").onclick = () => {
  muted = !muted; stream.getAudioTracks().forEach(t => t.enabled = !muted);
  $("mute").textContent = muted ? "Unmute" : "Mute"; $("mystate").textContent = muted ? "Muted" : "Connected";
};
$("stop").onclick = () => { try { ws.close(); stream.getTracks().forEach(t => t.stop()); ctx.close(); } catch (e) {} };
window.__ready = true;
</script></body></html>"""


async def page(request):
    room, role = request.path_params["room"], request.path_params["role"]
    if not valid(room, role) or room not in ROOMS:
        return HTMLResponse("This call link is not valid or has expired.", status_code=404)
    return HTMLResponse(PAGE)


async def send_presence(ws, present):
    """Small JSON message: is the other person in the call?"""
    try:
        await ws.send_text(json.dumps({"type": "presence", "other": bool(present)}))
    except Exception:
        pass


async def audio_socket(ws: WebSocket):
    room, role = ws.path_params["room"], ws.path_params["role"]
    if not valid(room, role) or room not in ROOMS:
        await ws.close(code=4404)
        return
    await ws.accept()
    slot = ROOMS[room]["roles"][role]
    other = ROOMS[room]["roles"][OTHER[role]]
    old = slot["ws"]
    slot["ws"] = ws                                                  # a newer connection of the same role replaces the old one
    if old is not None:
        try:
            await old.close(code=4409)
        except Exception:
            pass
    slot["connected"] = True
    slot["first"] = slot["first"] or time.time()
    await send_presence(ws, other["ws"] is not None)                 # tell the new person whether the other one is already here
    if other["ws"] is not None:
        await send_presence(other["ws"], True)                       # tell the other one that this person just joined
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) % 2 == 0 and len(data) <= 64000:            # int16 samples only, at most 2 s per message
                if seconds_of(slot["pcm"]) < MAX_SECONDS:
                    slot["pcm"].extend(data)                         # recording (phase B analyses this)
                peer = other["ws"]
                if peer is not None:                                 # the other person hears it
                    try:
                        await peer.send_bytes(data)
                    except Exception:
                        other["ws"] = None
    except WebSocketDisconnect:
        pass
    finally:
        if slot["ws"] is ws:                                         # not replaced by a newer connection
            slot["ws"] = None
            slot["connected"] = False
            if other["ws"] is not None:
                await send_presence(other["ws"], False)              # the other person sees that this one left
        save_wav(room, role)


async def create_room(request):
    if not authorized(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        body = {}
    chunk = min(15.0, max(3.0, float(body.get("chunk", 5.0))))
    threshold = min(0.95, max(0.0, float(body.get("threshold", 0.4))))
    while len(ROOMS) >= MAX_ROOMS:                                 # drop the oldest room
        old = ROOMS.pop(min(ROOMS, key=lambda r: ROOMS[r]["created"]))
        old["ended"] = True
        if old["task"] is not None:
            old["task"].cancel()
    room = new_room(chunk, threshold)
    ROOMS[room]["task"] = asyncio.create_task(analysis_loop(room))
    return JSONResponse({"room": room, "chunk": chunk, "threshold": threshold})


async def room_status(request):
    if not authorized(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    room = request.path_params["room"]
    if not ROOM_RE.match(room) or room not in ROOMS:
        return JSONResponse({"error": "unknown room"}, status_code=404)
    return JSONResponse({r: {"connected": s["connected"], "seconds": round(seconds_of(s["pcm"]), 1)}
                         for r, s in ROOMS[room]["roles"].items()})


async def room_emotions(request):
    if not authorized(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    room = request.path_params["room"]
    if not ROOM_RE.match(room) or room not in ROOMS:
        return JSONResponse({"error": "unknown room"}, status_code=404)
    r = ROOMS[room]
    return JSONResponse({
        "status": r["status"], "ended": r["ended"], "settings": r["settings"],
        "roles": {k: {"connected": s["connected"], "seconds": round(seconds_of(s["pcm"]), 1),
                      "offset": round(call_clock_offset(room, k), 2)} for k, s in r["roles"].items()},
        "results": r["results"],
    })


async def end_room(request):
    if not authorized(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    room = request.path_params["room"]
    if not ROOM_RE.match(room) or room not in ROOMS:
        return JSONResponse({"error": "unknown room"}, status_code=404)
    r = ROOMS[room]
    r["ended"] = True
    if r["task"] is not None:
        try:
            await asyncio.wait_for(asyncio.shield(r["task"]), timeout=120)      # the loop analyzes the leftover audio and stops
        except (asyncio.TimeoutError, Exception):
            pass
    saved = {k: str(p) for k in ROLES if (p := save_wav(room, k))}
    return JSONResponse({"saved": saved, "status": r["status"]})


app = Starlette(routes=[
    Route("/call/{room}/{role}", page),
    WebSocketRoute("/ws/{room}/{role}", audio_socket),
    Route("/api/rooms", create_room, methods=["POST"]),
    Route("/api/rooms/{room}", room_status),
    Route("/api/rooms/{room}/emotions", room_emotions),
    Route("/api/rooms/{room}/end", end_room, methods=["POST"]),
])

if __name__ == "__main__":
    print(f"Live call server on http://localhost:{PORT}  (recordings are saved in '{CALLS_DIR}')")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")