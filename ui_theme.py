"""Lavender AI design system: palette, icons, CSS and small HTML helpers.

Pure Python (no Streamlit import at module level) so that analysis.py can reuse the palette and the CSS for the
downloadable HTML report. The emotion colors are NOT part of this theme: they stay in analysis.COLORS.
"""
import html

# ------------------------------------------------------------------ palette
NAVY = "#292447"        # primary dark, main text
PURPLE = "#7565B5"      # primary accent
PURPLE_DARK = "#5E4F9A" # hover state, purple text on pale backgrounds (contrast)
SOFT = "#B7A9E5"        # secondary accent
PALE = "#EDE9FA"        # light accent
BG = "#F7F6FC"          # main background
WHITE = "#FFFFFF"       # cards
MUTED = "#625B79"       # secondary text
BORDER = "#E4E0ED"      # borders
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, 'Helvetica Neue', Arial, sans-serif"

# ------------------------------------------------------------------ icons (simple 24x24 stroke icons, Feather style)
_ICONS = {
    "waveform": '<line x1="4" y1="9" x2="4" y2="15"/><line x1="8" y1="5" x2="8" y2="19"/><line x1="12" y1="8" x2="12" y2="16"/>'
                '<line x1="16" y1="3" x2="16" y2="21"/><line x1="20" y1="9" x2="20" y2="15"/>',
    "upload": '<polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>'
              '<path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "repeat": '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/>'
              '<path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "headset": '<path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3z'
               'M3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/>',
    "chart": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/>'
            '<line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
    "sparkles": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "tag": '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
            '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>'
            '<line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
}


def icon(name, size=20, color="currentColor"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICONS[name]}</svg>')


# ------------------------------------------------------------------ navigation (anchor links, no JavaScript)
NAV_ITEMS = [
    ("sec-upload", "Upload & Analysis", "upload"),
    ("sec-live", "Live Call", "phone"),
    ("sec-overview", "Call Overview", "clock"),
    ("sec-roles", "Speaker Roles", "users"),
    ("sec-analytics", "Emotion Analytics", "chart"),
    ("sec-predictions", "Segment-Level Predictions", "list"),
    ("sec-report", "AI Call Report", "sparkles"),
    ("sec-export", "Export Report", "download"),
]

# ------------------------------------------------------------------ app CSS
# Selectors rely on the documented `key=` class (.st-key-<key>), semantic HTML (details, table) and our own classes.
SHARED_CSS = f"""
:root {{ --navy:{NAVY}; --purple:{PURPLE}; --purple-dark:{PURPLE_DARK}; --soft:{SOFT}; --pale:{PALE}; --bg:{BG};
        --muted:{MUTED}; --border:{BORDER}; --font:{FONT}; --shadow:0 1px 2px rgba(41,36,71,.05),0 8px 22px rgba(41,36,71,.07); }}
/* KPI cards */
.kpi {{ display:flex; align-items:center; gap:14px; background:#fff; border:1px solid var(--border); border-radius:14px;
       padding:16px 18px; min-height:92px; box-shadow:var(--shadow); height:100%; box-sizing:border-box; margin-bottom:14px; }}
.kpi-icon {{ width:40px; height:40px; border-radius:12px; background:var(--pale); color:var(--purple-dark);
            display:flex; align-items:center; justify-content:center; flex:none; }}
.kpi-label {{ font-size:.8rem; font-weight:500; color:var(--muted); line-height:1.25; }}
.kpi-value {{ font-size:1.65rem; font-weight:700; color:var(--navy); line-height:1.2; letter-spacing:-.01em; text-transform:capitalize; }}
.kpi-unit {{ font-size:.9rem; font-weight:500; color:var(--muted); margin-left:4px; text-transform:none; }}

/* live call monitor */
.live {{ background:var(--bg); border:1px solid var(--border); border-radius:14px; padding:14px 16px; }}
.live-head {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }}
.live-name {{ display:flex; align-items:center; gap:10px; font-weight:650; color:var(--navy); font-size:1.05rem; }}
.live-name span {{ width:32px; height:32px; border-radius:10px; background:var(--pale); color:var(--purple-dark);
                  display:inline-flex; align-items:center; justify-content:center; }}
.pill {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }}
.pill i {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.pill.on {{ background:#E3F4E6; color:#1F6B2E; }} .pill.on i {{ background:#2ca02c; }}
.pill.off {{ background:var(--pale); color:var(--muted); }} .pill.off i {{ background:#9a94ad; }}
.live-value {{ font-size:1.6rem; font-weight:700; color:var(--navy); line-height:1.2; }}
.live-value small {{ font-size:.85rem; font-weight:500; color:var(--muted); margin-left:4px; }}
.live-sub {{ color:var(--muted); font-size:.85rem; margin-top:8px; }}
.live-last {{ margin-top:4px; min-height:30px; display:flex; align-items:center; gap:8px; color:var(--muted); }}
.live-label {{ color:var(--muted); font-size:.82rem; font-weight:600; margin:2px 0 4px 0; }}
.st-key-card-live .card-title {{ margin-top:0; }}

/* predictions table */
.pred-wrap {{ max-height:460px; overflow:auto; border:1px solid var(--border); border-radius:12px; }}
.pred {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
.pred th {{ position:sticky; top:0; background:var(--pale); color:var(--navy); text-align:left; font-weight:600;
           padding:10px 14px; border-bottom:1px solid var(--border); z-index:1; }}
.pred td {{ padding:10px 14px; border-bottom:1px solid var(--border); vertical-align:top; color:var(--navy); }}
.pred tr:nth-child(even) td {{ background:#FBFAFE; }}
.pred td.t {{ white-space:nowrap; color:var(--muted); font-variant-numeric:tabular-nums; }}
.pred td.c {{ font-variant-numeric:tabular-nums; }}
.pred td.tx {{ min-width:280px; white-space:normal; word-break:break-word; }}
.chip {{ display:inline-flex; align-items:center; gap:7px; padding:3px 11px 3px 9px; border-radius:999px; font-weight:600;
        font-size:.82rem; text-transform:capitalize; }}
.chip i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}

/* AI report */
.rcard {{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:18px 22px; box-shadow:var(--shadow);
         border-top:4px solid var(--soft); margin-bottom:14px; }}
.rcard-title {{ margin:0 0 8px 0; display:flex; align-items:center; gap:10px; font-size:1.05rem; font-weight:650; color:var(--navy); }}
.rcard-title span {{ width:30px; height:30px; border-radius:9px; background:var(--pale); color:var(--purple-dark);
                 display:flex; align-items:center; justify-content:center; }}
.rcard p {{ margin:0; color:var(--navy); line-height:1.6; }}
"""

APP_CSS = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');""" + SHARED_CSS + f"""
.stApp {{ font-family: var(--font); color: var(--navy); background: var(--bg); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp button, .stApp input {{ font-family: var(--font); }}
.block-container {{ max-width: 1240px; padding-top: 4.2rem; padding-bottom: 4rem; }}
[data-testid='stSpinner'] [role='status'] {{ display: none; }}

/* header */
.app-header {{ display:flex; align-items:center; gap:18px; padding:6px 0 18px 0; }}
.app-logo {{ width:56px; height:56px; border-radius:16px; display:flex; align-items:center; justify-content:center;
            background:linear-gradient(135deg,var(--purple),var(--navy)); color:#fff; flex:none; box-shadow:var(--shadow); }}
.app-title {{ display:block; margin:0; font-size:2rem; line-height:1.15; font-weight:700; color:var(--navy); letter-spacing:-.02em; }}
.app-subtitle {{ display:block; margin:6px 0 0 0; font-size:1rem; color:var(--muted); max-width:760px; }}

/* section headers */
.sec {{ display:flex; align-items:center; gap:14px; margin:2.2rem 0 .9rem 0; padding-bottom:.8rem;
       border-bottom:1px solid var(--border); scroll-margin-top:70px; }}
.sec-icon {{ width:38px; height:38px; border-radius:11px; background:var(--pale); color:var(--purple-dark);
            display:flex; align-items:center; justify-content:center; flex:none; }}
.sec-title {{ margin:0; padding:0; font-size:1.3rem; font-weight:650; color:var(--navy); line-height:1.2; }}
.sec-sub {{ margin:2px 0 0 0; font-size:.9rem; color:var(--muted); }}
.sec:target .sec-title {{ color:var(--purple-dark); }}
.sec:target .sec-icon {{ background:var(--purple); color:#fff; }}

/* cards (st.container(key="card-...")) */
[class*="st-key-card-"] {{ background:#fff; border:1px solid var(--border); border-radius:16px; padding:20px 22px;
                          box-shadow:var(--shadow); }}
.card-title {{ margin:0 0 4px 0; font-size:1.05rem; font-weight:650; color:var(--navy); }}
.card-note {{ margin:0 0 10px 0; font-size:.85rem; color:var(--muted); }}
.empty {{ background:#fff; border:1px dashed var(--soft); border-radius:14px; padding:18px 20px; color:var(--muted); font-size:.95rem; }}

/* speaker role cards */
.role {{ display:flex; align-items:center; gap:16px; background:#fff; border:1px solid var(--border); border-radius:16px;
        padding:20px 22px; box-shadow:var(--shadow); border-left:6px solid var(--purple); box-sizing:border-box; height:100%; margin-bottom:12px; }}
.role.agent {{ border-left-color:var(--navy); }}
.role-icon {{ width:48px; height:48px; border-radius:14px; background:var(--pale); color:var(--purple-dark);
             display:flex; align-items:center; justify-content:center; flex:none; }}
.role.agent .role-icon {{ background:var(--navy); color:#fff; }}
.role-name {{ font-size:1.1rem; font-weight:650; color:var(--navy); }}
.role-id {{ font-size:.9rem; color:var(--muted); }}
.role-id code {{ background:var(--pale); color:var(--navy); padding:2px 8px; border-radius:8px; font-size:.85rem; }}
.role-tag {{ display:inline-block; margin-left:8px; font-size:.72rem; font-weight:600; padding:2px 8px; border-radius:999px;
            background:var(--pale); color:var(--purple-dark); }}

/* buttons */
.st-key-start-btn button, .st-key-download-btn button {{ background:var(--purple); color:#fff; border:none; border-radius:12px; font-weight:600;
                            padding:.6rem 1.4rem; transition:background .15s; }}
.st-key-start-btn button:hover, .st-key-start-btn button:focus-visible,
.st-key-download-btn button:hover, .st-key-download-btn button:focus-visible {{ background:var(--purple-dark); color:#fff; }}
.st-key-start-btn button p, .st-key-download-btn button p {{ color:#fff; }}

.st-key-report-sections details {{ background:#fff; border:1px solid var(--border) !important; border-radius:14px; margin-bottom:10px;
                                   box-shadow:var(--shadow); }}
.st-key-report-sections summary {{ font-weight:600; color:var(--navy); }}

/* sidebar */
[data-testid='stSidebar'] {{ background:{NAVY}; }}
[data-testid='stSidebar'] * {{ color:{PALE}; }}
.brand {{ display:flex; align-items:center; gap:12px; padding:4px 0 14px 0; }}
.brand .app-logo {{ width:40px; height:40px; border-radius:12px; background:var(--purple); box-shadow:none; }}
.brand b {{ display:block; color:#fff; font-size:1rem; line-height:1.2; }}
.brand small {{ color:{SOFT}; font-size:.78rem; }}
.nav-title {{ font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; color:{SOFT}; margin:8px 0 8px 4px; font-weight:600; }}
a.nav {{ display:flex; align-items:center; gap:12px; padding:9px 12px; margin-bottom:3px; border-radius:10px; text-decoration:none;
        color:{PALE} !important; font-size:.92rem; font-weight:500; border-left:3px solid transparent; }}
a.nav:hover, a.nav:focus-visible {{ background:rgba(183,169,229,.16); color:#fff !important; }}
.stApp:has(#sec-upload:target) a.nav[href="#sec-upload"], .stApp:has(#sec-overview:target) a.nav[href="#sec-overview"],
.stApp:has(#sec-roles:target) a.nav[href="#sec-roles"], .stApp:has(#sec-analytics:target) a.nav[href="#sec-analytics"],
.stApp:has(#sec-predictions:target) a.nav[href="#sec-predictions"], .stApp:has(#sec-report:target) a.nav[href="#sec-report"],
.stApp:has(#sec-export:target) a.nav[href="#sec-export"] {{ background:rgba(183,169,229,.22); border-left-color:{SOFT}; color:#fff !important; font-weight:600; }}
.side-note {{ background:rgba(183,169,229,.14); border:1px solid rgba(183,169,229,.35); border-radius:10px; padding:10px 12px; font-size:.82rem; line-height:1.45; color:{PALE}; margin:6px 0 12px 0; }}
.side-rule {{ border:none; border-top:1px solid rgba(183,169,229,.25); margin:14px 0; }}
"""


# ------------------------------------------------------------------ downloadable report CSS (screen + A4 print)
REPORT_CSS = SHARED_CSS + f"""
body {{ margin:0; background:var(--bg); color:var(--navy); font-family:var(--font); line-height:1.5; }}
.page {{ max-width:960px; margin:0 auto; padding:28px 20px 48px 20px; }}
.hero {{ background:{NAVY}; color:#fff; border-radius:18px; padding:26px 30px; border-bottom:5px solid {SOFT}; }}
.hero h1 {{ margin:0 0 6px 0; font-size:1.7rem; font-weight:700; letter-spacing:-.01em; color:#fff; }}
.hero p {{ margin:0; color:{PALE}; font-size:.95rem; }}
.hero .meta {{ margin-top:12px; font-size:.82rem; color:{SOFT}; }}
h2.sec-h {{ margin:34px 0 4px 0; font-size:1.3rem; color:var(--navy); font-weight:650; padding-bottom:8px; border-bottom:2px solid {SOFT}; }}
p.sec-p {{ margin:6px 0 14px 0; color:var(--muted); font-size:.9rem; }}
.card {{ background:#fff; border:1px solid var(--border); border-radius:14px; padding:16px 20px; margin:12px 0;
        box-shadow:var(--shadow); }}
.card h3 {{ margin:0 0 8px 0; font-size:1rem; color:var(--navy); font-weight:650; }}
.band {{ background:var(--pale); border-radius:14px; padding:14px 16px; margin:12px 0; }}
.kpis {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
.kpis .kpi {{ margin-bottom:0; box-shadow:none; padding:12px 14px; }}
.kpis .kpi-label {{ font-size:.74rem; }}
.kpis .kpi-value {{ font-size:1.4rem; }}
.chart {{ display:flex; justify-content:center; overflow:hidden; }}
.info {{ display:grid; grid-template-columns:210px 1fr; gap:8px 14px; margin:0; font-size:.92rem; }}
.info dt {{ color:var(--muted); }}
.info dd {{ margin:0; font-weight:600; word-break:break-word; }}
.note {{ margin:8px 0 0 0; font-size:.82rem; color:var(--muted); }}
ul.items {{ margin:4px 0 0 0; padding-left:20px; }}
ul.items li {{ margin:4px 0; }}
.foot {{ margin-top:34px; padding-top:12px; border-top:1px solid var(--border); font-size:.8rem; color:var(--muted); text-align:center; }}
.pred-wrap {{ max-height:none; overflow:visible; }}
.pred th {{ position:static; }}

@page {{ size:A4; margin:14mm; }}
@media print {{
  body {{ background:#fff; }}
  .page {{ max-width:none; padding:0; }}
  .hero {{ border-radius:0; }}
  .card, .kpi, .rcard {{ box-shadow:none; }}
  .card, .rcard, .kpi, .band, .chart {{ break-inside:avoid; page-break-inside:avoid; }}
  h2.sec-h {{ break-after:avoid; page-break-after:avoid; }}
  .kpis {{ grid-template-columns:repeat(3,1fr); }}
  .pred tr {{ break-inside:avoid; page-break-inside:avoid; }}
  .pred thead {{ display:table-header-group; }}
  .chip, .chip i, .hero {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  .modebar-container {{ display:none !important; }}
}}
"""


# ------------------------------------------------------------------ HTML helpers
def app_header_html(title, subtitle):
    return (f'<div class="app-header"><div class="app-logo">{icon("waveform", 30)}</div>'
            f'<div><div class="app-title" role="heading" aria-level="1">{html.escape(title)}</div><div class="app-subtitle">{html.escape(subtitle)}</div></div></div>')


def section_header_html(anchor, title, subtitle, icon_name):
    return (f'<div class="sec" id="{anchor}"><div class="sec-icon">{icon(icon_name, 20)}</div>'
            f'<div><div class="sec-title" role="heading" aria-level="2">{html.escape(title)}</div>{f'<div class="sec-sub">{html.escape(subtitle)}</div>' if subtitle else ''}</div></div>')


def kpi_html(label, value, icon_name, unit="", tooltip=""):
    unit_html = f'<span class="kpi-unit">{html.escape(unit)}</span>' if unit else ""
    title = f' title="{html.escape(tooltip, quote=True)}"' if tooltip else ""
    return (f'<div class="kpi"{title}><div class="kpi-icon">{icon(icon_name, 20)}</div><div>'
            f'<div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(str(value))}{unit_html}</div></div></div>')


def live_status_html(role, connected, seconds, last_chip_html=""):
    """One speaker's live status: connection pill, seconds received and the latest predicted emotion."""
    pill = ('<span class="pill on"><i></i>Connected</span>' if connected else '<span class="pill off"><i></i>Not connected</span>')
    last = last_chip_html or "Waiting for speech"
    return (f'<div class="live"><div class="live-head"><div class="live-name"><span>{icon("headset" if role == "Agent" else "user", 18)}</span>'
            f'{html.escape(role)}</div>{pill}</div>'
            f'<div class="live-value">{seconds:.1f}<small>s received</small></div>'
            f'<div class="live-sub">Latest segment</div><div class="live-last">{last}</div></div>')


def role_card_html(role, speaker_id, icon_name, note=""):
    cls = "agent" if role == "Agent" else "customer"
    tag = f'<span class="role-tag">{html.escape(note)}</span>' if note else ""
    return (f'<div class="role {cls}"><div class="role-icon">{icon(icon_name, 24)}</div><div>'
            f'<div class="role-name">{html.escape(role)}{tag}</div>'
            f'<div class="role-id">Assigned speaker ID: <code>{html.escape(str(speaker_id))}</code></div></div></div>')


def empty_html(text):
    return f'<div class="empty">{html.escape(text)}</div>'


def sidebar_nav_html(shown=None):
    links = "".join(f'<a class="nav" href="#{a}">{icon(i, 18)}<span>{html.escape(t)}</span></a>'
                    for a, t, i in NAV_ITEMS if shown is None or a in shown)
    return (f'<div class="brand"><div class="app-logo">{icon("waveform", 22)}</div>'
            f'<div><b>Emotion Analytics</b><small>Customer service calls</small></div></div>'
            f'<div class="nav-title">Navigation</div><nav>{links}</nav><hr class="side-rule">')


def rcard_html(title, body_html, icon_name):
    return f'<div class="rcard"><div class="rcard-title" role="heading" aria-level="3"><span>{icon(icon_name, 17)}</span>{html.escape(title)}</div><p>{body_html}</p></div>'


def _tint(hex_color, alpha=0.14):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def emotion_chip_html(emotion, colors):
    color = colors.get(emotion, "#c7c7c7")
    return (f'<span class="chip" style="background:{_tint(color)};color:{NAVY}"><i style="background:{color}"></i>'
            f'{html.escape(str(emotion))}</span>')


def predictions_table_html(df, colors, include_transcript=True):
    """Segment-level predictions as a styled, scrollable table. Values are shown exactly as stored (no truncation)."""
    show_tx = include_transcript and "transcript" in df.columns
    head = "<tr><th>Time</th><th>Emotion</th><th>Confidence</th>" + ("<th>Transcript</th>" if show_tx else "") + "</tr>"
    rows = []
    for _, r in df.iterrows():
        cells = (f'<td class="t">{html.escape(str(r["time"]))}</td><td>{emotion_chip_html(r["emotion"], colors)}</td>'
                 f'<td class="c">{float(r["confidence"]) * 100:.1f}%</td>')
        if show_tx:
            cells += f'<td class="tx">{html.escape(str(r["transcript"]))}</td>'
        rows.append(f"<tr>{cells}</tr>")
    return f'<div class="pred-wrap"><table class="pred"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'