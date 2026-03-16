"""
Streamlit RAG interface — chat UI + embedded PDF viewer
Vehicle Manual Edition — Automotive Dashboard Theme

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from urllib.parse import quote
import streamlit as st
from streamlit.components.v1 import html as st_html
from core.chat_memory import ChatMemoryManager
from core.storage_manager import StorageManager
from core.retriever import SmartRetriever, MultiCollectionRetriever
from core.metadata_manager import MetadataManager
from tools.pdf_server import get_viewer_url, SERVER_PORT, start_server_background

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="BIAK — Vehicle Manual AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Start pdf_server ─────────────────────────────────────────────────────────
@st.cache_resource
def _boot_pdf_server():
    start_server_background()

_boot_pdf_server()

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
PDF_HTTP_BASE = "http://localhost:8000"


# ─── Session state defaults ───────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "selected_collection": None,
    "pdf_filename": None,
    "pdf_page": 1,
    "query_count": 0,
    "chat_memory": ChatMemoryManager(),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helpers ──────────────────────────────────────────────────────────────────
def pdf_exists_on_disk(filename: str) -> bool:
    return bool(filename) and (PDF_DIR / filename).exists()


def get_pdf_http_url(filename: str, page: int) -> str:
    return f"{PDF_HTTP_BASE}/{quote(filename)}#page={int(page)}"


def render_pdf_viewer_pdfjs(filename: str, page: int, height: int = 720) -> None:
    """
    Renders PDF by injecting bytes directly as a JS Uint8Array literal.
    Vehicle-themed viewer with dark instrument panel styling.
    """
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        st.warning(f"PDF not found: `{filename}` (expected under `{PDF_DIR}`)")
        return

    raw_bytes = pdf_path.read_bytes()
    hex_str = raw_bytes.hex()

    # Dark automotive theme for embedded viewer
    v_bg = "#0f1117"
    v_bar = "#161b27"
    v_text = "#e8eaf0"
    v_border = "#2a3045"
    v_btn = "#1e2535"
    v_btn_h = "#2a3450"
    v_accent = "#f59e0b"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{
      height:100vh; overflow:hidden; background:{v_bg};
      font-family:'Rajdhani', sans-serif;
      display:flex; flex-direction:column;
    }}
    .bar {{
      background:{v_bar};
      border-bottom:1px solid {v_border};
      padding:8px 14px;
      display:flex; align-items:center; gap:12px; flex:0 0 auto;
      position:relative;
    }}
    .bar::after {{
      content:''; position:absolute; bottom:-2px; left:0; right:0; height:1px;
      background:linear-gradient(90deg, transparent, {v_accent}55, transparent);
    }}
    .logo {{
      font-size:11px; font-weight:700; letter-spacing:.2em;
      color:{v_accent}; flex-shrink:0; font-family:'Share Tech Mono', monospace;
      text-transform:uppercase;
    }}
    .fname {{
      flex:1; font-size:12px; color:{v_text}; overflow:hidden;
      text-overflow:ellipsis; white-space:nowrap; opacity:.7;
      font-family:'Share Tech Mono', monospace;
    }}
    .nav {{ display:flex; align-items:center; gap:6px; flex-shrink:0; }}
    .nav input {{
      width:56px; padding:3px 6px; font-size:11px;
      border:1px solid {v_border}; border-radius:3px; outline:none;
      background:#0a0d14; color:{v_text}; text-align:center;
      font-family:'Share Tech Mono', monospace;
      transition: border-color .15s;
    }}
    .nav input:focus {{ border-color:{v_accent}; box-shadow: 0 0 6px {v_accent}44; }}
    .total {{ font-size:10px; color:{v_text}; opacity:.45; white-space:nowrap;
      font-family:'Share Tech Mono', monospace; }}
    button {{
      padding:3px 12px; font-size:10px; font-weight:700; letter-spacing:.1em;
      border:1px solid {v_border}; border-radius:3px;
      background:{v_btn}; color:{v_text}; cursor:pointer;
      font-family:'Rajdhani', sans-serif; text-transform:uppercase;
      transition:all .14s;
    }}
    button:hover {{
      background:{v_btn_h}; border-color:{v_accent};
      color:{v_accent}; box-shadow:0 0 8px {v_accent}33;
    }}
    .wrap {{
      flex:1 1 auto; overflow:auto; padding:12px;
      position:relative; display:flex; flex-direction:column; align-items:center;
      background: radial-gradient(ellipse at 50% 0%, #1a2035 0%, {v_bg} 70%);
    }}
    canvas {{
      display:none; background:white;
      border:1px solid {v_border}; border-radius:6px; max-width:100%;
      box-shadow:0 4px 24px rgba(0,0,0,.5), 0 0 0 1px {v_border};
    }}
    .loader {{
      display:flex; flex-direction:column; align-items:center;
      justify-content:center; gap:12px; width:100%; flex:1;
      color:{v_text}; font-size:11px; letter-spacing:.1em; opacity:.7;
      font-family:'Share Tech Mono', monospace; text-transform:uppercase;
    }}
    .spinner {{
      width:28px; height:28px;
      border:2px solid {v_border}; border-top-color:{v_accent};
      border-radius:50%; animation:spin .8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
    .err {{
      margin:12px; padding:12px 14px; color:#f87171;
      background:#1a0a0a; border:1px solid #7f1d1d;
      border-radius:6px; font-size:11px; line-height:1.6; width:100%;
      white-space:pre-wrap; font-family:'Share Tech Mono', monospace;
    }}
  </style>
</head>
<body>
  <div class="bar">
    <div class="logo">▶ MANUAL</div>
    <div class="fname" title="{filename}">{filename}</div>
    <div class="nav">
      <input id="pg" type="number" min="1" value="{int(page)}"/>
      <span class="total" id="total"></span>
      <button id="go">GO</button>
    </div>
  </div>
  <div class="wrap" id="wrap">
    <div class="loader" id="loader">
      <div class="spinner"></div>
      <div id="loadmsg">Loading PDF.js…</div>
    </div>
    <canvas id="cv"></canvas>
    <div id="err" class="err" style="display:none;"></div>
  </div>

<script>
var HEX = "{hex_str}";
var START_PAGE = {int(page)};

function showErr(msg) {{
  document.getElementById('loader').style.display = 'none';
  document.getElementById('cv').style.display     = 'none';
  var el = document.getElementById('err');
  el.style.display = 'block';
  el.textContent   = '⚠ ' + msg;
}}

function hexToUint8(hex) {{
  var len   = hex.length / 2;
  var bytes = new Uint8Array(len);
  for (var i = 0; i < len; i++) {{
    bytes[i] = parseInt(hex.substr(i * 2, 2), 16);
  }}
  return bytes;
}}

function loadScript(url, cb, errCb) {{
  var s    = document.createElement('script');
  s.src    = url;
  s.onload = cb;
  s.onerror = errCb;
  document.head.appendChild(s);
}}

function startViewer() {{
  document.getElementById('loadmsg').textContent = 'Decoding PDF…';

  var pdfBytes;
  try {{
    pdfBytes = hexToUint8(HEX);
  }} catch(e) {{
    showErr('Hex decode failed: ' + e.message);
    return;
  }}

  var workerCode = 'importScripts("https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js");';
  var workerBlob = new Blob([workerCode], {{ type: 'application/javascript' }});
  var workerUrl  = URL.createObjectURL(workerBlob);

  pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

  document.getElementById('loadmsg').textContent = 'Rendering…';

  var loadTask = pdfjsLib.getDocument({{ data: pdfBytes }});

  loadTask.promise.then(function(pdf) {{
    var pgInput = document.getElementById('pg');
    var totalEl = document.getElementById('total');
    var goBtn   = document.getElementById('go');
    var loader  = document.getElementById('loader');
    var canvas  = document.getElementById('cv');

    pgInput.max         = pdf.numPages;
    totalEl.textContent = '/ ' + pdf.numPages;

    var p = Math.min(Math.max(START_PAGE, 1), pdf.numPages);
    pgInput.value = p;

    function renderPage(num) {{
      loader.style.display = 'flex';
      canvas.style.display = 'none';
      document.getElementById('loadmsg').textContent = 'Loading page ' + num + '…';

      pdf.getPage(num).then(function(page) {{
        var wrap       = document.getElementById('wrap');
        var containerW = wrap.clientWidth - 24;
        var vp1        = page.getViewport({{ scale: 1 }});
        var scale      = Math.min(2.5, Math.max(1.0, containerW / vp1.width));
        var vp         = page.getViewport({{ scale: scale }});

        canvas.width  = Math.floor(vp.width);
        canvas.height = Math.floor(vp.height);

        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        page.render({{ canvasContext: ctx, viewport: vp }}).promise.then(function() {{
          loader.style.display = 'none';
          canvas.style.display = 'block';
        }}).catch(function(e) {{
          showErr('Render error: ' + e.message);
        }});
      }}).catch(function(e) {{
        showErr('Page load error: ' + e.message);
      }});
    }}

    renderPage(p);

    goBtn.addEventListener('click', function() {{
      var v = parseInt(pgInput.value, 10);
      if (!isFinite(v) || v < 1) v = 1;
      if (v > pdf.numPages) v = pdf.numPages;
      pgInput.value = v;
      renderPage(v);
    }});

    pgInput.addEventListener('keydown', function(e) {{
      if (e.key === 'Enter') goBtn.click();
    }});

  }}).catch(function(e) {{
    showErr('PDF load error: ' + (e && e.message ? e.message : String(e)));
  }});
}}

loadScript(
  'https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.min.js',
  function() {{ startViewer(); }},
  function() {{
    loadScript(
      'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js',
      function() {{ startViewer(); }},
      function() {{ showErr('Could not load PDF.js from unpkg or jsdelivr.\\nCheck your internet connection.'); }}
    );
  }}
);
</script>
</body>
</html>
"""
    st_html(html, height=height, scrolling=False)


def render_source_pills(nodes, *, key_prefix: str) -> None:
    if not nodes:
        return

    mm = MetadataManager()
    pages = mm.extract_pages_from_nodes(nodes)
    if not pages:
        return

    ranges = mm.merge_consecutive_pages(pages)
    fname = mm.extract_filename_from_nodes(nodes)

    if not fname:
        return

    per_row = 6
    for r in range(0, len(ranges), per_row):
        row = ranges[r : r + per_row]
        cols = st.columns(len(row))
        for i, (start, end) in enumerate(row):
            label = mm.format_page_range(start, end)
            k = f"{key_prefix}_p_{start}_{end}"
            if cols[i].button(f"⬡ {label}", key=k, use_container_width=True):
                if pdf_exists_on_disk(fname):
                    st.session_state.pdf_filename = fname
                    st.session_state.pdf_page = int(start)
                    st.rerun()
                else:
                    st.warning(
                        f"Source PDF `{fname}` not found in `{PDF_DIR}`. "
                        f"Available: {[f.name for f in PDF_DIR.glob('*.pdf')]}"
                    )


# ─── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_resource
def get_storage():
    return StorageManager()


@st.cache_resource
def get_collections():
    return get_storage().list_collections()


@st.cache_resource
def get_retriever(collection_name: str | None):
    if collection_name:
        return SmartRetriever(collection_name, verbose=False)
    return MultiCollectionRetriever(verbose=False)


# ─── CSS — Automotive Dashboard Theme ─────────────────────────────────────────
# Palette: Deep navy cockpit + amber instrument glow + carbon fibre texture
BG       = "#0f1117"
SIDEBAR  = "#0d1020"
PANEL    = "#141824"
TEXT     = "#dde2f0"
BORDER   = "#252d42"
ACCENT   = "#f59e0b"       # amber — instrument cluster orange
ACCENT2  = "#38bdf8"       # ice blue — secondary highlights
CHIP     = "#1a2035"
DANGER   = "#ef4444"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&family=Outfit:wght@300;400;500;600&display=swap');

:root {{
  --bg:{BG}; --sidebar:{SIDEBAR}; --panel:{PANEL};
  --text:{TEXT}; --border:{BORDER}; --accent:{ACCENT};
  --accent2:{ACCENT2}; --chip:{CHIP};
}}

/* ── Reset & Base ─────────────────────────────────── */
#MainMenu, footer {{ visibility:hidden; }}

.stApp {{
  background-color:var(--bg) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(245,158,11,.07) 0%, transparent 60%),
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 39px,
      rgba(255,255,255,.012) 40px
    ),
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 39px,
      rgba(255,255,255,.012) 40px
    );
}}

.block-container {{
  padding:1.2rem 1.8rem 2rem 1.8rem !important;
  background-color:transparent !important;
}}

html, body, [class*="css"] {{
  font-family:'Outfit', sans-serif !important;
  color:var(--text) !important;
}}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0a0d18 0%, {SIDEBAR} 100%) !important;
  border-right:1px solid var(--border) !important;
  box-shadow: 4px 0 24px rgba(0,0,0,.4) !important;
}}

[data-testid="stSidebar"]::before {{
  content:'';
  position:absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, transparent, {ACCENT}, transparent);
}}

[data-testid="stSidebar"] * {{ color:var(--text) !important; }}

[data-testid="collapsedControl"] {{ display:block !important; visibility:visible !important; opacity:1 !important; z-index:9999 !important; }}
[data-testid="collapsedControl"] button {{
  border:1px solid var(--border) !important;
  background:var(--sidebar) !important;
}}

header[data-testid="stHeader"] {{ background:transparent !important; }}

/* ── Sidebar selectbox ────────────────────────────── */
[data-testid="stSidebar"] .stSelectbox > div > div {{
  background: #0d1120 !important;
  border:1px solid var(--border) !important;
  border-radius:6px !important;
  color:var(--text) !important;
  font-family:'Share Tech Mono', monospace !important;
  font-size:12px !important;
}}

/* ── Buttons ──────────────────────────────────────── */
.stButton > button {{
  background: linear-gradient(135deg, #151c2e 0%, #1a2238 100%) !important;
  color:var(--text) !important;
  border:1px solid var(--border) !important;
  font-family:'Rajdhani', sans-serif !important;
  font-size:12px !important;
  font-weight:600 !important;
  letter-spacing:.08em !important;
  text-transform:uppercase !important;
  border-radius:4px !important;
  transition:all 0.18s ease !important;
}}
.stButton > button:hover {{
  background: linear-gradient(135deg, #1e2a44 0%, #253050 100%) !important;
  color:{ACCENT} !important;
  border-color:{ACCENT} !important;
  box-shadow: 0 0 12px rgba(245,158,11,.25) !important;
}}

/* ── Chat messages ────────────────────────────────── */
[data-testid="stChatMessage"] {{
  border-radius:8px !important;
  margin-bottom:8px !important;
  background: linear-gradient(135deg, #141824 0%, #111520 100%) !important;
  border:1px solid var(--border) !important;
  backdrop-filter: blur(4px);
  transition: border-color .2s;
}}
[data-testid="stChatMessage"]:hover {{
  border-color: #3a4560 !important;
}}
[data-testid="stChatMessage"] > div > div {{
  padding:2px 12px 2px 12px !important;
}}

/* User message — subtle left accent bar */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
  border-left:3px solid {ACCENT2} !important;
}}
/* Assistant message — amber left accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
  border-left:3px solid {ACCENT} !important;
}}

/* ── Chat input ───────────────────────────────────── */
[data-testid="stChatInput"] {{
  background: #0d1020 !important;
  border:1px solid var(--border) !important;
  border-radius:8px !important;
  transition: border-color .2s, box-shadow .2s;
}}
[data-testid="stChatInput"]:focus-within {{
  border-color:{ACCENT} !important;
  box-shadow: 0 0 16px rgba(245,158,11,.15) !important;
}}
[data-testid="stChatInput"] textarea {{
  background:transparent !important;
  color:var(--text) !important;
  font-family:'Outfit', sans-serif !important;
  font-size:14px !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
  color:var(--text) !important;
  opacity:0.35 !important;
}}

/* ── Section headers ──────────────────────────────── */
h1, h2, h3, h4 {{
  font-family:'Rajdhani', sans-serif !important;
  font-weight:600 !important;
  letter-spacing:.05em !important;
  color:var(--text) !important;
}}
h3 {{
  font-size:1.1rem !important;
  text-transform:uppercase !important;
  letter-spacing:.12em !important;
  padding-bottom:8px !important;
  border-bottom:1px solid var(--border) !important;
  margin-bottom:10px !important;
}}

/* ── Collection badge ─────────────────────────────── */
.coll-badge {{
  display:inline-flex; align-items:center; gap:5px;
  background: linear-gradient(135deg, #1a2235, #1e2840);
  border:1px solid {ACCENT}55;
  color:{ACCENT};
  font-family:'Share Tech Mono', monospace;
  font-size:10px;
  padding:3px 10px;
  border-radius:3px;
  letter-spacing:.1em;
  text-transform:uppercase;
  margin-bottom:6px;
  box-shadow: 0 0 8px {ACCENT}22;
}}

/* ── Stat cards ───────────────────────────────────── */
.stat-row {{ display:flex; gap:8px; margin-bottom:16px; }}
.stat-card {{
  flex:1;
  background: linear-gradient(135deg, #111520 0%, #141824 100%);
  border:1px solid var(--border);
  border-radius:8px;
  padding:12px 8px;
  text-align:center;
  position:relative;
  overflow:hidden;
  transition: border-color .2s, box-shadow .2s;
}}
.stat-card::before {{
  content:''; position:absolute;
  top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, transparent, {ACCENT}88, transparent);
}}
.stat-card:hover {{
  border-color:{ACCENT}55;
  box-shadow: 0 0 16px {ACCENT}18;
}}
.stat-num {{
  font-family:'Rajdhani', sans-serif;
  font-size:22px; font-weight:700;
  color:{ACCENT}; line-height:1;
  text-shadow: 0 0 12px {ACCENT}66;
}}
.stat-label {{
  font-size:9px; color:var(--text);
  letter-spacing:.1em; text-transform:uppercase;
  margin-top:4px; opacity:0.5;
  font-family:'Share Tech Mono', monospace;
}}

/* ── PDF empty state ──────────────────────────────── */
.empty-pdf {{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  height:500px; gap:16px;
  color:var(--text); font-family:'Share Tech Mono', monospace; font-size:12px;
  letter-spacing:.06em; border:1px dashed #2a3045; border-radius:12px;
  background: radial-gradient(ellipse at 50% 40%, #141e35 0%, #0d1020 100%);
  text-transform:uppercase;
}}
.empty-pdf .ei {{ font-size:48px; opacity:.3; filter:grayscale(1); }}
.empty-pdf .hint {{ opacity:.4; font-size:10px; }}

/* ── Page pill source buttons ─────────────────────── */
div[data-testid="stHorizontalBlock"] .stButton > button {{
  font-size:10px !important;
  padding:3px 8px !important;
  line-height:1.2 !important;
  min-height:0px !important;
  height:auto !important;
  border-radius:20px !important;
  background: linear-gradient(135deg, #161e32 0%, #1a2438 100%) !important;
  color:{ACCENT} !important;
  border:1px solid {ACCENT}55 !important;
  letter-spacing:.05em !important;
  font-family:'Share Tech Mono', monospace !important;
  font-weight:400 !important;
  text-transform:none !important;
  box-shadow: 0 0 6px {ACCENT}15 !important;
  transition:all .16s !important;
}}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
  background: {ACCENT} !important;
  color:#0f1117 !important;
  border-color:{ACCENT} !important;
  box-shadow: 0 0 14px {ACCENT}55 !important;
}}
div[data-testid="stHorizontalBlock"] .stButton > button div {{
  font-size:11px !important;
}}

/* ── Number input (page jump) ─────────────────────── */
[data-testid="stNumberInput"] input {{
  background: #0d1020 !important;
  border:1px solid var(--border) !important;
  color:var(--text) !important;
  border-radius:4px !important;
  font-family:'Share Tech Mono', monospace !important;
  font-size:12px !important;
}}
[data-testid="stNumberInput"] input:focus {{
  border-color:{ACCENT} !important;
  box-shadow: 0 0 8px {ACCENT}33 !important;
}}

/* ── Spinner ──────────────────────────────────────── */
[data-testid="stSpinner"] {{
  color:{ACCENT} !important;
}}

/* ── Sidebar app title ────────────────────────────── */
.sidebar-title {{
  font-family:'Rajdhani', sans-serif;
  font-size:22px;
  font-weight:700;
  letter-spacing:.2em;
  color:var(--text);
  text-transform:uppercase;
  line-height:1;
  margin-bottom:2px;
}}
.sidebar-sub {{
  font-family:'Share Tech Mono', monospace;
  font-size:9px;
  letter-spacing:.18em;
  color:{ACCENT};
  text-transform:uppercase;
  opacity:.8;
}}

/* ── PDF meta bar ─────────────────────────────────── */
.pdf-meta {{
  font-family:'Share Tech Mono', monospace;
  font-size:11px; color:var(--text);
  padding:5px 0;
  display:flex; align-items:center; gap:6px;
}}
.pdf-dot {{ color:{ACCENT}; }}
.pdf-name {{ opacity:.8; }}
.pdf-sep {{ opacity:.3; }}
.pdf-pg {{ color:{ACCENT2}; }}

/* ── Link buttons (open in tab) ───────────────────── */
.ext-link {{
  font-family:'Share Tech Mono', monospace;
  font-size:10px; color:var(--text) !important;
  text-decoration:none;
  border:1px solid var(--border);
  padding:4px 14px; border-radius:4px;
  display:inline-block; margin-top:8px;
  text-transform:uppercase; letter-spacing:.08em;
  transition:all .15s;
}}
.ext-link:hover {{
  border-color:{ACCENT};
  color:{ACCENT} !important;
  box-shadow: 0 0 8px {ACCENT}33;
}}

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{
  background: #2a3555; border-radius:4px;
}}
::-webkit-scrollbar-thumb:hover {{ background:{ACCENT}88; }}

/* ── Warning/error boxes ──────────────────────────── */
[data-testid="stAlert"] {{
  background: #1a0e0e !important;
  border:1px solid #7f1d1d !important;
  border-radius:6px !important;
  color:#fca5a5 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 0 16px 0;">
          <div class="sidebar-title">BIAC</div>
          <div class="sidebar-sub">▶ Vehicle Manual AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="height:1px;background:linear-gradient(90deg,transparent,{ACCENT}88,transparent);margin-bottom:16px;"></div>',
        unsafe_allow_html=True,
    )

    collections = get_collections()
    if not collections:
        st.error("No collections found.\nRun `python scripts/process_pdfs.py` first.")
        st.stop()

    options = ["— All manuals —"] + collections
    idx = 0
    if st.session_state.selected_collection in collections:
        idx = collections.index(st.session_state.selected_collection) + 1

    st.markdown(
        '<div style="font-family:\'Share Tech Mono\',monospace;font-size:9px;letter-spacing:.12em;color:#888;text-transform:uppercase;margin-bottom:4px;">Select Manual</div>',
        unsafe_allow_html=True,
    )
    chosen = st.selectbox("Manual", options, index=idx, label_visibility="collapsed")
    selected = None if chosen == "— All manuals —" else chosen

    if selected != st.session_state.selected_collection:
        st.session_state.selected_collection = selected
        st.session_state.messages = []
        st.session_state.pdf_filename = None
        st.session_state.pdf_page = 1
        st.rerun()

    st.markdown(
        f'<div style="height:1px;background:var(--border);margin:16px 0;"></div>',
        unsafe_allow_html=True,
    )

    total_chunks = 0
    try:
        for c in collections:
            total_chunks += get_storage().get_collection_info(c).get("count", 0)
    except Exception:
        pass

    st.markdown(
        f"""
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-num">{len(collections)}</div>
        <div class="stat-label">Manuals</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{total_chunks}</div>
        <div class="stat-label">Chunks</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{st.session_state.query_count}</div>
        <div class="stat-label">Queries</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="height:1px;background:var(--border);margin:4px 0 14px 0;"></div>',
        unsafe_allow_html=True,
    )

    if st.button("⬡  Clear Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_memory.clear()
        st.session_state.pdf_filename = None
        st.session_state.pdf_page = 1
        st.session_state.query_count = 0
        st.rerun()

    st.markdown(
        f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:9px;
      color:var(--text);margin-top:16px;opacity:.4;letter-spacing:.08em;
      text-transform:uppercase; line-height:1.8;">
      ● PDF Server · Port {SERVER_PORT}<br>
      ● LlamaIndex · ChromaDB · Azure OpenAI
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─── Main columns ─────────────────────────────────────────────────────────────
col_chat, col_pdf = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Chat
# ══════════════════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown("### 🚗 Ask Your Manual")

    CHAT_HEIGHT = 650
    chat_area = st.container(height=CHAT_HEIGHT)

    with chat_area:
        for mi, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    if msg.get("collection"):
                        st.markdown(
                            f'<span class="coll-badge">▶ {msg["collection"]}</span>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(msg["content"])

                    nodes = msg.get("nodes", [])
                    if nodes:
                        render_source_pills(nodes, key_prefix=f"hist_{mi}")

                else:
                    st.markdown(msg["content"])

        tail = st.empty()

    query = st.chat_input("Search your vehicle manual…  e.g. 'How do I reset the service light?'")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        memory = st.session_state.chat_memory
        history = memory.get_context()

        context = ""
        if history:
            context = "\n".join(
                f"{msg.role.capitalize()}: {msg.content}" for msg in history
            )
        if context:
            query_with_memory = f"""
            You are answering questions about vehicle manuals.

            Conversation history:
            {context}

            User question:
            {query}
            """
        else:
            query_with_memory = query

        with tail.container():
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                try:
                    retriever = get_retriever(st.session_state.selected_collection)
                    is_multi = isinstance(retriever, MultiCollectionRetriever)
                    coll_label = st.session_state.selected_collection

                    if coll_label:
                        st.markdown(
                            f'<span class="coll-badge">▶ {coll_label}</span>',
                            unsafe_allow_html=True,
                        )

                    with st.spinner("Searching manual…"):
                        if is_multi:
                            response = retriever.query_best(query_with_memory)
                            coll_label = response.collection_name
                        else:
                            response = retriever.query(query_with_memory)

                    if getattr(response, "retrieval_successful", False):
                        answer = response.answer
                        nodes = response.source_nodes
                        memory.add_user_message(query)
                        memory.add_assistant_message(answer)
                        st.markdown(answer)

                        render_source_pills(nodes, key_prefix=f"live_{st.session_state.query_count}")

                        # auto-set viewer to first page
                        mm = MetadataManager()
                        pages = mm.extract_pages_from_nodes(nodes)
                        fname = mm.extract_filename_from_nodes(nodes)
                        if pages and fname and pdf_exists_on_disk(fname):
                            st.session_state.pdf_filename = fname
                            st.session_state.pdf_page = int(pages[0])

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "nodes": nodes,
                                "collection": coll_label,
                            }
                        )
                        st.session_state.query_count += 1
                    else:
                        err = getattr(response, "error_message", "Unknown error")
                        st.error(f"Retrieval failed: {err}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"⚠️ {err}", "nodes": []}
                        )

                except Exception as e:
                    logger.exception("Query error")
                    st.error(f"Error: {e}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"⚠️ {e}", "nodes": []}
                    )

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — PDF Viewer
# ══════════════════════════════════════════════════════════════════════════════
with col_pdf:
    st.markdown("### 📋 Source Document")

    fname = st.session_state.pdf_filename
    page = int(st.session_state.pdf_page or 1)

    if fname:
        col_info, col_jump = st.columns([3, 1])

        with col_info:
            st.markdown(
                f'<div class="pdf-meta">'
                f'<span class="pdf-dot">●</span>'
                f'<span class="pdf-name">{fname}</span>'
                f'<span class="pdf-sep">·</span>'
                f'<span class="pdf-pg">pg {page}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_jump:
            new_page = st.number_input(
                "page",
                min_value=1,
                value=page,
                step=1,
                label_visibility="collapsed",
                key=f"pjump_{fname}_{page}",
            )
            if int(new_page) != page:
                st.session_state.pdf_page = int(new_page)
                st.rerun()

        render_pdf_viewer_pdfjs(fname, page, height=720)

        viewer_url = get_viewer_url(fname, page)
        raw_url = get_pdf_http_url(fname, page)
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" class="ext-link">↗ Open Viewer</a>'
            f'&nbsp;&nbsp;'
            f'<a href="{raw_url}" target="_blank" class="ext-link">↗ Raw PDF</a>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            """
        <div class="empty-pdf">
          <div class="ei">🚗</div>
          <div>Ask a question to load the source page</div>
          <div class="hint">The relevant manual page will appear here automatically</div>
        </div>
        """,
            unsafe_allow_html=True,
        )