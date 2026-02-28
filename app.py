import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import json

from src.law_fetcher import fetch_law_summary
from src.linker import link_comments_to_law
from src.sentiment_engine import analyze_sentiment
from src.insight_engine import get_groq_insight

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PolicyIQ",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root variables ───────────────────────────────────────────────────── */
:root {
  --bg:          #060912;
  --bg2:         #0c1120;
  --surface:     rgba(255,255,255,0.04);
  --surface-h:   rgba(255,255,255,0.07);
  --border:      rgba(255,255,255,0.08);
  --border-h:    rgba(255,255,255,0.18);
  --text:        #e8ecf4;
  --muted:       #6b7280;
  --accent1:     #7c6aff;
  --accent2:     #3ecfcf;
  --accent3:     #f4736e;
  --green:       #34d399;
  --red:         #f87171;
  --gold:        #fbbf24;
  --radius:      14px;
  --radius-lg:   20px;
  --transition:  all 0.3s cubic-bezier(0.4,0,0.2,1);
}

/* ── Global reset ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
  margin: 0;
  padding: 0;
}

/* Hide Streamlit chrome completely to remove the top blank gap */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* FIX: 10% margins for content, but allow absolute full width for navbar */
.block-container {
  padding-top: 100px !important; /* Pushes content down so fixed navbar doesn't cover it */
  padding-bottom: 2rem !important;
  padding-left: 10% !important;
  padding-right: 10% !important;
  max-width: 100% !important;
  margin: 0 auto;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,106,255,0.4); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,106,255,0.7); }

/* ── Animated mesh background ─────────────────────────────────────────── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  z-index: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(124,106,255,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 85% 90%,  rgba(62,207,207,0.08) 0%, transparent 55%),
    radial-gradient(ellipse 50% 60% at 50% 50%,  rgba(244,115,110,0.04) 0%, transparent 70%);
  animation: meshDrift 18s ease-in-out infinite alternate;
  pointer-events: none;
}
@keyframes meshDrift {
  0%   { opacity: 1; transform: scale(1) translateY(0); }
  50%  { opacity: 0.85; transform: scale(1.04) translateY(-15px); }
  100% { opacity: 1; transform: scale(1) translateY(0); }
}

/* ── Page wrappers ────────────────────────────────────────────────────── */
.page-wrap {
  position: relative;
  z-index: 1;
  width: 100%;
  margin: 0 auto;
  padding: 0 0 100px;
}
.page-wrap-wide {
  position: relative;
  z-index: 1;
  width: 100%;
  margin: 0 auto;
  padding: 0 0 100px;
}

/* ── Navbar (Fixed edge-to-edge) ──────────────────────────────────────── */
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  width: 100vw;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 5%;
  background: rgba(6,9,18,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
}
.navbar-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'Syne', sans-serif;
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.4px;
  color: var(--text);
}
.navbar-mark {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, var(--accent1), var(--accent2));
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px;
  box-shadow: 0 0 20px rgba(124,106,255,0.4);
}
.navbar-badge {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.06em;
  padding: 4px 11px;
  background: rgba(124,106,255,0.12);
  border: 1px solid rgba(124,106,255,0.28);
  border-radius: 99px;
  color: var(--accent1);
}

/* ── 4-Step progress ──────────────────────────────────────────────────── */
.progress-bar-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 36px 0 12px;
  animation: fadeSlide .5s ease forwards;
}
.step-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.step-circle {
  width: 40px; height: 40px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-size: 13px; font-weight: 700;
  border: 2px solid var(--border);
  background: var(--bg2);
  color: var(--muted);
  transition: var(--transition);
  position: relative;
  z-index: 1;
}
.step-circle.active {
  border-color: var(--accent1);
  background: rgba(124,106,255,0.18);
  color: var(--accent1);
  box-shadow: 0 0 18px rgba(124,106,255,0.45);
}
.step-circle.done {
  border-color: var(--green);
  background: rgba(52,211,153,0.12);
  color: var(--green);
  box-shadow: 0 0 12px rgba(52,211,153,0.25);
}
.step-label {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--muted);
  white-space: nowrap;
  text-align: center;
  max-width: 80px;
  line-height: 1.3;
}
.step-label.active { color: var(--accent1); }
.step-label.done   { color: var(--green); }
.step-connector {
  width: 70px; height: 2px;
  background: var(--border);
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
  flex-shrink: 0;
}
.step-connector.done::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, var(--green), var(--accent2));
  animation: connFill .7s ease forwards;
}
.step-connector.active-fill::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, var(--accent1), rgba(124,106,255,0.3));
  width: 50%;
}
@keyframes connFill { from { width: 0; } to { width: 100%; } }

/* ── Animations ───────────────────────────────────────────────────────── */
@keyframes fadeSlide {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 4px 20px rgba(124,106,255,0.3); }
  50%       { box-shadow: 0 4px 35px rgba(124,106,255,0.6); }
}
@keyframes insightSlideIn {
  from { opacity: 0; transform: translateX(-20px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes insightSlideInRight {
  from { opacity: 0; transform: translateX(20px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes borderGlow {
  0%, 100% { box-shadow: 0 0 20px rgba(248,113,113,0.15); }
  50%       { box-shadow: 0 0 40px rgba(248,113,113,0.35); }
}
@keyframes borderGlowGreen {
  0%, 100% { box-shadow: 0 0 20px rgba(52,211,153,0.15); }
  50%       { box-shadow: 0 0 40px rgba(52,211,153,0.35); }
}

.card-d1 { animation: fadeSlide .5s .05s ease both; }
.card-d2 { animation: fadeSlide .5s .15s ease both; }
.card-d3 { animation: fadeSlide .5s .25s ease both; }
.card-d4 { animation: fadeSlide .5s .35s ease both; }

/* ── Card base ────────────────────────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 28px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.025) 0%, transparent 60%);
  pointer-events: none;
}
.card:hover {
  border-color: var(--border-h);
  background: var(--surface-h);
  transform: translateY(-3px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, var(--accent1) 0%, var(--accent2) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 99px !important;
  padding: 12px 32px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  cursor: pointer !important;
  transition: var(--transition) !important;
  animation: glow-pulse 3s ease-in-out infinite !important;
  width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
  transform: scale(1.04) translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(124,106,255,0.55) !important;
}
div[data-testid="stButton"] > button:active {
  transform: scale(0.98) !important;
}

/* ── Text inputs ──────────────────────────────────────────────────────── */
div[data-testid="stTextInput"] input {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  padding: 14px 16px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent1) !important;
  box-shadow: 0 0 0 3px rgba(124,106,255,0.2) !important;
  outline: none !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stFileUploader"] label {
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  margin-bottom: 8px !important;
}

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px dashed rgba(124,106,255,0.35) !important;
  border-radius: var(--radius) !important;
  transition: var(--transition) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  background: rgba(124,106,255,0.05) !important;
  border-color: var(--accent1) !important;
}

/* ── Loading Spinner & Wrap ───────────────────────────────────────────── */
.loading-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 0 36px;
  gap: 16px;
  animation: fadeSlide .4s ease forwards;
}
.loading-title {
  font-family: 'Syne', sans-serif;
  font-size: 22px; font-weight: 700;
  color: var(--text);
  margin-top: 10px;
}
.loading-sub { font-size: 13px; color: var(--muted); }

/* Custom Circular Loader */
.spinner {
  width: 50px;
  height: 50px;
  border: 4px solid rgba(255, 255, 255, 0.05);
  border-top: 4px solid var(--accent1);
  border-right: 4px solid var(--accent2);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  box-shadow: 0 0 15px rgba(124, 106, 255, 0.2);
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* ── Report header ────────────────────────────────────────────────────── */
.report-header {
  padding: 20px 0 20px;
  animation: fadeSlide .4s ease forwards;
}
.report-title {
  font-family: 'Syne', sans-serif;
  font-size: 32px; font-weight: 800;
  letter-spacing: -1px; color: var(--text);
  margin-bottom: 6px;
}
.policy-pill {
  display: inline-block;
  padding: 3px 12px;
  background: rgba(124,106,255,.12);
  border: 1px solid rgba(124,106,255,.25);
  border-radius: 99px;
  font-size: 12px; font-weight: 500;
  color: var(--accent1); letter-spacing: .02em;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
.divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 32px 0;
}

/* ── INSIGHT CARDS (premium animated) ────────────────────────────────── */
.insights-section-title {
  font-family: 'Syne', sans-serif;
  font-size: 18px; font-weight: 700;
  color: var(--text); letter-spacing: -.3px;
  margin-bottom: 20px;
  margin-top: 20px;
  animation: fadeSlide .5s .4s ease both;
  opacity: 0;
}
.insight-card {
  border-radius: var(--radius-lg);
  padding: 28px 30px;
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: transform .3s cubic-bezier(0.4,0,0.2,1), box-shadow .3s ease;
  position: relative;
  overflow: hidden;
  height: 100%;
}
.insight-card::after {
  content: '';
  position: absolute;
  top: -30px; right: -30px;
  width: 100px; height: 100px;
  border-radius: 50%;
  pointer-events: none;
}
.insight-card:hover {
  transform: translateY(-6px);
}
/* Opposed card */
.insight-opposed {
  background: linear-gradient(135deg, rgba(248,113,113,0.10) 0%, rgba(248,113,113,0.03) 100%);
  border: 1px solid rgba(248,113,113,0.30);
  animation: insightSlideIn .6s .5s ease both, borderGlow 3s 1.2s ease-in-out infinite;
}
.insight-opposed::after {
  background: radial-gradient(circle, rgba(248,113,113,0.12), transparent 70%);
}
.insight-opposed:hover {
  box-shadow: 0 20px 50px rgba(248,113,113,0.20);
}
/* Supported card */
.insight-supported {
  background: linear-gradient(135deg, rgba(52,211,153,0.10) 0%, rgba(52,211,153,0.03) 100%);
  border: 1px solid rgba(52,211,153,0.30);
  animation: insightSlideInRight .6s .6s ease both, borderGlowGreen 3s 1.3s ease-in-out infinite;
}
.insight-supported::after {
  background: radial-gradient(circle, rgba(52,211,153,0.12), transparent 70%);
}
.insight-supported:hover {
  box-shadow: 0 20px 50px rgba(52,211,153,0.20);
}
/* Insight card header bar */
.insight-header-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  border-radius: 10px;
  margin-bottom: 18px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.insight-header-bar-red {
  background: rgba(248,113,113,0.15);
  border: 1px solid rgba(248,113,113,0.25);
  color: #f87171;
}
.insight-header-bar-green {
  background: rgba(52,211,153,0.15);
  border: 1px solid rgba(52,211,153,0.25);
  color: #34d399;
}
.insight-body {
  font-size: 14px;
  color: #c8d0e0;
  line-height: 1.80;
  font-weight: 300;
}

/* ── Plotly container rounding ────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}

/* Suppress original h3 global pulse */
h3 { animation: none !important; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND  — DO NOT TOUCH
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_cached_analysis(law_name, file_bytes):
    temp_path = os.path.join("data", "raw", "temp_upload.csv")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    df = pd.read_csv(temp_path)

    status_text.markdown("### 🔍 1/4: Fetching Law Context (Groq Llama-3)...")
    progress_bar.progress(25)
    law_data = fetch_law_summary(law_name)

    law_context_path = os.path.join("data", "processed", "law_context.json")
    os.makedirs(os.path.dirname(law_context_path), exist_ok=True)
    with open(law_context_path, "w") as f:
        json.dump(law_data, f)

    status_text.markdown("### 🔗 2/4: Linking Comments to Clauses (MiniLM)...")
    progress_bar.progress(50)
    linked_df = link_comments_to_law(df)

    status_text.markdown("### 🧠 3/4: Analyzing Sentiment (RoBERTa)...")
    progress_bar.progress(75)
    final_df = analyze_sentiment(linked_df)

    status_text.markdown("### 💡 4/4: Generating Insights...")
    progress_bar.progress(90)
    clean_df = final_df[
        (final_df['Sentiment_Label'] != 'N/A') &
        (final_df['Linked_Clause'] != 'Irrelevant')
    ]
    summary = clean_df.groupby(['Linked_Clause', 'Sentiment_Label']).size().unstack(fill_value=0)
    for col in ['negative', 'positive']:
        if col not in summary.columns:
            summary[col] = 0

    insights = {}
    if summary['negative'].sum() > 0:
        opp_sec = summary['negative'].idxmax()
        opp_comments = clean_df[
            (clean_df['Linked_Clause'] == opp_sec) &
            (clean_df['Sentiment_Label'] == 'negative')
        ]['Comment'].tolist()
        insights['opposed_sec'] = opp_sec
        insights['opposed_text'] = get_groq_insight(opp_comments, law_name, opp_sec, "negative")

    if summary['positive'].sum() > 0:
        sup_sec = summary['positive'].idxmax()
        sup_comments = clean_df[
            (clean_df['Linked_Clause'] == sup_sec) &
            (clean_df['Sentiment_Label'] == 'positive')
        ]['Comment'].tolist()
        insights['supported_sec'] = sup_sec
        insights['supported_text'] = get_groq_insight(sup_comments, law_name, sup_sec, "positive")

    return clean_df, insights

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  — DO NOT TOUCH
# ─────────────────────────────────────────────────────────────────────────────
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'final_data' not in st.session_state:
    st.session_state.final_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = {}

def set_step(step_num):
    st.session_state.step = step_num

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def navbar():
    st.markdown("""
    <div class="navbar">
      <div class="navbar-brand">
        <div class="navbar-mark">⚖️</div>
        PolicyIQ
      </div>
      <span class="navbar-badge">Public Comment Intelligence</span>
    </div>
    """, unsafe_allow_html=True)


def step_progress(current: int, placeholder=None):
    """4-step progress bar: 1=Law Context, 2=Linking, 3=Sentiment, 4=Insights."""
    steps = [
        ("01", "Law Context"),
        ("02", "Linking"),
        ("03", "Sentiment"),
        ("04", "Insights"),
    ]
    html = ""
    for i, (ico, lbl) in enumerate(steps, start=1):
        if i < current:
            cc, lc, ico = "step-circle done", "step-label done", "✓"
        elif i == current:
            cc, lc = "step-circle active", "step-label active"
        else:
            cc, lc = "step-circle", "step-label"

        conn = ""
        if i < len(steps):
            if i < current:
                ck = "step-connector done"
            elif i == current:
                ck = "step-connector active-fill"
            else:
                ck = "step-connector"
            conn = f'<div class="{ck}"></div>'

        html += f"""
        <div class="step-node">
          <div class="{cc}">{ico}</div>
          <span class="{lc}">{lbl}</span>
        </div>{conn}"""

    html_out = f'<div class="progress-bar-wrap">{html}</div>'
    
    if placeholder is not None:
        placeholder.markdown(html_out, unsafe_allow_html=True)
    else:
        st.markdown(html_out, unsafe_allow_html=True)


def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Cleaned up Plotly base settings
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#6b7280", size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.07)",
        tickfont=dict(color="#6b7280"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.07)",
        tickfont=dict(color="#6b7280"),
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER NAVBAR — always visible
# ─────────────────────────────────────────────────────────────────────────────
navbar()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INPUT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step == 1:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:60px 0 0; text-align:center; animation:fadeSlide .5s ease forwards;">
      <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;
                  letter-spacing:-.6px;color:var(--text);margin-bottom:12px;">
        Analyse Public Comments
      </div>
      <div style="font-size:16px;color:var(--muted);max-width:600px;
                  margin:0 auto 40px;line-height:1.65;">
        Upload a CSV of public comments and name the policy. Our pipeline links each
        comment to a clause, classifies sentiment, and synthesises executive insights.
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        # ABSOLUTE FIX: No HTML card wrapper here at all to prevent the empty ghost box above inputs.
        
        law_name = st.text_input(
            "Policy / Law Name",
            value="Personal Data Protection Bill, 2019",
        )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Comments (CSV)", type=["csv"])

        if uploaded_file is not None:
            st.markdown(f"""
            <div style="margin-top:12px;padding:12px 16px;
                        background:rgba(52,211,153,0.08);
                        border:1px solid rgba(52,211,153,0.22);
                        border-radius:10px;font-size:13px;color:#34d399;">
              ✓ &nbsp;<strong>{uploaded_file.name}</strong> ready
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            # PERFECT CENTERING FIX: using 3 equal columns guarantees the button sits flawlessly in the middle
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col2:
                if st.button("Run Analysis", use_container_width=True):
                    st.session_state.file_bytes = uploaded_file.getvalue()
                    st.session_state.law_name   = law_name
                    set_step(2)
                    st.rerun()
        else:
            st.markdown("""
            <div style="padding:10px 0;font-size:13px;color:var(--muted);text-align:center;">
              Upload a CSV file to unlock analysis
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == 2:
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    tracker_placeholder = st.empty()
    step_progress(1, tracker_placeholder)

    st.markdown("""
    <div class="loading-wrap">
      <div class="spinner"></div>
      <div class="loading-title">Running NLP Pipeline</div>
      <div class="loading-sub">Law Context → Linking → Sentiment Analysis → Insights</div>
    </div>
    """, unsafe_allow_html=True)

    class DynamicStatus:
        def markdown(self, text, **kwargs):
            if "1/4" in text or "Initialising" in text:
                step_progress(1, tracker_placeholder)
            elif "2/4" in text:
                step_progress(2, tracker_placeholder)
            elif "3/4" in text:
                step_progress(3, tracker_placeholder)
            elif "4/4" in text or "Analysis complete" in text:
                step_progress(4, tracker_placeholder)

    class HiddenProgress:
        def progress(self, value):
            pass 

    status_text = DynamicStatus()
    progress_bar = HiddenProgress()

    try:
        status_text.markdown("Initialising pipeline…")
        
        clean_df, insights = run_cached_analysis(
            st.session_state.law_name,
            st.session_state.file_bytes,
        )
        
        status_text.markdown("Analysis complete…")
        time.sleep(1)
        
        st.session_state.final_data = clean_df
        st.session_state.insights   = insights
        set_step(3)
        st.rerun()

    except Exception as e:
        st.markdown(f"""
        <div style="margin-top:24px;padding:18px 22px;
                    background:rgba(248,113,113,0.08);
                    border:1px solid rgba(248,113,113,0.25);
                    border-radius:14px;font-size:14px;color:#f87171;text-align:center;">
          ⚠️ Pipeline error: <strong>{e}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if st.button("← Try Again"):
                set_step(1)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — IMPACT REPORT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == 3:
    df       = st.session_state.final_data
    law_name = st.session_state.law_name

    st.markdown('<div class="page-wrap-wide">', unsafe_allow_html=True)

    # ── Report header ─────────────────────────────────────────────────────────
    head_col, btn_col = st.columns([5, 1])
    with head_col:
        st.markdown(f"""
        <div class="report-header">
          <div class="report-title">📊 Impact Report</div>
          <div style="font-size:14px;color:var(--muted);display:flex;align-items:center;gap:8px;margin-top:4px;">
            Policy:&nbsp;<span class="policy-pill">{law_name}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with btn_col:
        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
        if st.button("🔄 New Analysis"):
            set_step(1)
            st.rerun()

    divider()

    # ── ROW 1: PIE CHART (Overall Sentiment) ──────────────────────────────────
    # Replaced CSS classes with pure inline styles to guarantee NO empty border boxes
    st.markdown(
        '<div style="text-align: center; margin-top: 10px; animation: fadeSlide .5s .05s ease both;">'
        '<div style="font-family: \'Syne\', sans-serif; font-size: 20px; font-weight: 700; color: #e8ecf4; margin-bottom: 4px;">Overall Sentiment</div>'
        '<div style="font-size: 13px; color: #6b7280; margin-bottom: 20px;">Total % breakdown across all comments</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    pie_data = df['Sentiment_Label'].value_counts().reset_index()
    pie_data.columns = ['Sentiment', 'Count']

    pie_colors = []
    color_map  = {'negative': '#f87171', 'positive': '#34d399', 'neutral': '#7c6aff'}
    for s in pie_data['Sentiment']:
        pie_colors.append(color_map.get(s, '#6b7280'))

    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_data['Sentiment'],
        values=pie_data['Count'],
        hole=0.45,
        marker=dict(
            colors=pie_colors,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        textinfo='percent',
        textposition='inside',
        textfont=dict(family="DM Sans", color="#fff", size=14, weight=600),
        hovertemplate="<b>%{label}</b><br>%{value} comments (%{percent})<extra></extra>",
        direction='clockwise',
        sort=False,
    )])
    
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#6b7280", size=14),
        height=380,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#9ca3af", size=14),
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            itemsizing="constant",
        ),
        margin=dict(l=0, r=80, t=20, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})


    # ── ROW 2: BAR CHART (Sentiment by Clause) ────────────────────────────────
    # Pure inline styles to guarantee NO empty border boxes
    st.markdown(
        '<div style="text-align: center; margin-top: 50px; animation: fadeSlide .5s .15s ease both;">'
        '<div style="font-family: \'Syne\', sans-serif; font-size: 20px; font-weight: 700; color: #e8ecf4; margin-bottom: 4px;">Sentiment by Clause</div>'
        '<div style="font-size: 13px; color: #6b7280; margin-bottom: 20px;">Volume of comments per policy section — grouped by sentiment</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    bar_data = df.groupby(['Linked_Clause', 'Sentiment_Label']).size().reset_index(name='Count')

    fig_bar = px.bar(
        bar_data,
        x='Linked_Clause',
        y='Count',
        color='Sentiment_Label',
        barmode='group',
        color_discrete_map={
            'negative': '#f87171',
            'neutral':  '#7c6aff',
            'positive': '#34d399',
        },
        labels={'Linked_Clause': '', 'Count': 'Volume', 'Sentiment_Label': 'Sentiment_Label'},
    )
    
    fig_bar.update_layout(
        **_PL,
        height=380,
        xaxis_title="",
        yaxis_title="Volume",
        bargap=0.20,
        bargroupgap=0.04,
        legend=dict(
            title="Sentiment_Label",
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af", size=12),
            orientation="v",
            yanchor="top", y=1,
            xanchor="right", x=1,
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    divider()

    # ── ROW 3: INSIGHT CARDS ──────────────────────────────────────────────────
    st.markdown("""
    <div class="insights-section-title">
      💡 Insights
    </div>
    """, unsafe_allow_html=True)

    opp_sec  = st.session_state.insights.get('opposed_sec',   'N/A')
    opp_text = st.session_state.insights.get('opposed_text',  'Not enough negative data to generate insight.')
    sup_sec  = st.session_state.insights.get('supported_sec', 'N/A')
    sup_text = st.session_state.insights.get('supported_text','Not enough positive data to generate insight.')

    ic1, ic2 = st.columns(2, gap="large")

    with ic1:
        st.markdown(f"""
        <div class="insight-card insight-opposed card-d3">
          <div class="insight-header-bar insight-header-bar-red">
            🚨 &nbsp;Most Opposed: {opp_sec}
          </div>
          <div class="insight-body">{opp_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with ic2:
        st.markdown(f"""
        <div class="insight-card insight-supported card-d4">
          <div class="insight-header-bar insight-header-bar-green">
            ✅ &nbsp;Most Supported: {sup_sec}
          </div>
          <div class="insight-body">{sup_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close page-wrap-wide