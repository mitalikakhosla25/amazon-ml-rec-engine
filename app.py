"""
app.py
------
Amazon-Style Multi-Modal Search & Recommendation System
Streamlit front-end that wires together:
  • search_engine.py  — semantic search via sentence-transformer embeddings + cosine similarity
  • recommender.py    — personalised recommendations via SVD matrix factorisation

Run with:
    streamlit run app.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
import pandas as pd

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Discover — ML Search & Recommend",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state.view = "landing"


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e8e4df;
}
.stApp {
    background: #08090c;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(110, 231, 183, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 10%, rgba(167, 139, 250, 0.06) 0%, transparent 55%),
        radial-gradient(ellipse 50% 30% at 50% 100%, rgba(232, 93, 76, 0.04) 0%, transparent 50%);
}
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1100px; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(110, 231, 183, 0.15); }
    50%      { box-shadow: 0 0 40px rgba(110, 231, 183, 0.3); }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes line-grow {
    from { width: 0; }
    to   { width: 100%; }
}
@keyframes dot-pulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50%      { opacity: 1; transform: scale(1.4); }
}

.anim-fade-up   { animation: fadeUp 0.7s ease both; }
.anim-fade-up-1 { animation: fadeUp 0.7s 0.1s ease both; }
.anim-fade-up-2 { animation: fadeUp 0.7s 0.2s ease both; }
.anim-fade-up-3 { animation: fadeUp 0.7s 0.35s ease both; }
.anim-fade-up-4 { animation: fadeUp 0.7s 0.5s ease both; }
.anim-fade-up-5 { animation: fadeUp 0.7s 0.65s ease both; }
.anim-slide-r   { animation: slideRight 0.6s ease both; }

/* ── Landing hero ── */
.landing-hero {
    min-height: 72vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 2rem 0 3rem;
}
.landing-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6ee7b7;
    margin-bottom: 1.2rem;
}
.landing-title {
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    color: #faf8f5;
    margin: 0 0 1.4rem;
    max-width: 720px;
}
.landing-title .accent {
    background: linear-gradient(135deg, #6ee7b7 0%, #a78bfa 50%, #e85d4c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.landing-lead {
    font-size: 1.1rem;
    line-height: 1.75;
    color: #9a9590;
    max-width: 580px;
    margin: 0 0 2.4rem;
    font-weight: 400;
}

/* ── Tower cards (landing) ── */
.tower-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1.5rem;
    align-items: stretch;
    margin: 2.5rem 0 3rem;
}
@media (max-width: 768px) {
    .tower-grid { grid-template-columns: 1fr; }
    .tower-connector { display: none !important; }
}
.tower-card {
    background: rgba(20, 24, 32, 0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s, transform 0.3s;
    position: relative;
    overflow: hidden;
}
.tower-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--tower-accent), transparent);
    opacity: 0.6;
}
.tower-card:hover {
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-4px);
}
.tower-card.search-tower { --tower-accent: #6ee7b7; }
.tower-card.rec-tower    { --tower-accent: #a78bfa; }
.tower-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 1rem;
}
.tower-card.search-tower .tower-icon { background: rgba(110,231,183,0.12); }
.tower-card.rec-tower    .tower-icon { background: rgba(167,139,250,0.12); }
.tower-name {
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--tower-accent);
    margin-bottom: 0.5rem;
}
.tower-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #faf8f5;
    margin: 0 0 0.7rem;
}
.tower-desc {
    font-size: 0.85rem;
    line-height: 1.65;
    color: #7a7570;
    margin: 0 0 1rem;
}
.tower-spec {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.tower-spec span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    padding: 3px 9px;
    border-radius: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    color: #6a6560;
}
.tower-connector {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0 0.5rem;
}
.connector-line {
    width: 2px;
    height: 40px;
    background: linear-gradient(180deg, #6ee7b7, #a78bfa);
    border-radius: 1px;
    animation: line-grow 1s 0.8s ease both;
}
.connector-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: #4a4540;
    writing-mode: vertical-rl;
    text-orientation: mixed;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    margin: 0 0 2.5rem;
    padding: 1.5rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.stat-item { }
.stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #faf8f5;
    letter-spacing: -0.02em;
    line-height: 1;
}
.stat-num .unit { font-size: 0.9rem; color: #6ee7b7; font-weight: 600; }
.stat-label {
    font-size: 0.75rem;
    color: #5a5550;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── CTA button (custom) ── */
.cta-wrap { margin-top: 0.5rem; }
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6ee7b7 0%, #34d399 100%) !important;
    color: #08090c !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2.2rem !important;
    border: none !important;
    border-radius: 50px !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(110, 231, 183, 0.25) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(110, 231, 183, 0.35) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important;
    color: #9a9590 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 50px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── App header (inner page) ── */
.app-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 2rem;
}
.app-logo {
    font-size: 1rem;
    font-weight: 800;
    color: #faf8f5;
    letter-spacing: -0.02em;
}
.app-logo span { color: #6ee7b7; }
.app-nav-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #4a4540;
    text-transform: uppercase;
}

/* ── Section headers ── */
.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6ee7b7;
    margin-bottom: 0.4rem;
}
.section-heading {
    font-size: 1.5rem;
    font-weight: 700;
    color: #faf8f5;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem;
}
.section-sub {
    font-size: 0.88rem;
    color: #6a6560;
    margin: 0 0 1.4rem;
    line-height: 1.6;
}

/* ── Search input ── */
.stTextInput > div > div > input {
    background: rgba(20, 24, 32, 0.8) !important;
    border: 1.5px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: #faf8f5 !important;
    font-size: 1rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.85rem 1.2rem !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6ee7b7 !important;
    box-shadow: 0 0 0 3px rgba(110, 231, 183, 0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #4a4540 !important; }

/* ── Result cards ── */
.result-card {
    background: rgba(20, 24, 32, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.85rem;
    position: relative;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
    animation: fadeUp 0.5s ease both;
}
.result-card:hover {
    border-color: rgba(110, 231, 183, 0.2);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.result-card .rank-chip {
    position: absolute;
    top: 1.1rem; right: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: #6ee7b7;
    background: rgba(110, 231, 183, 0.08);
    border: 1px solid rgba(110, 231, 183, 0.2);
    padding: 2px 10px;
    border-radius: 20px;
}
.result-card .card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #faf8f5;
    margin: 0 0 0.5rem;
    padding-right: 3.5rem;
    line-height: 1.4;
}
.result-card .meta-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 0.7rem;
    flex-wrap: wrap;
}
.meta-pill {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
}
.pill-cat   { background: rgba(167,139,250,0.12); color: #c4b5fd; }
.pill-price { background: rgba(110,231,183,0.1); color: #6ee7b7; font-family: 'IBM Plex Mono', monospace; }
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.6rem;
}
.score-bar-bg {
    flex: 1;
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6ee7b7, #a78bfa);
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.score-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6ee7b7;
    min-width: 58px;
    text-align: right;
}
.card-desc {
    font-size: 0.82rem;
    color: #5a5550;
    line-height: 1.6;
    margin: 0;
}

/* ── Recommendation panel ── */
.rec-panel {
    background: rgba(20, 24, 32, 0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 1.5rem;
    position: sticky;
    top: 1rem;
}
.rec-card {
    background: rgba(12, 14, 18, 0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
    animation: fadeUp 0.4s ease both;
}
.rec-card:hover { border-color: rgba(167, 139, 250, 0.2); }
.rec-card .rec-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #a78bfa;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.rec-card .rec-title {
    font-size: 0.84rem;
    font-weight: 600;
    color: #e8e4df;
    margin-bottom: 0.35rem;
    line-height: 1.35;
}
.rec-card .rec-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.3rem;
}
.rec-card .rec-cat   { font-size: 0.67rem; color: #7a7570; }
.rec-card .rec-price { font-size: 0.67rem; color: #6ee7b7; font-family: 'IBM Plex Mono', monospace; }
.rec-card .rec-score {
    font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.2);
    color: #c4b5fd;
    padding: 1px 8px;
    border-radius: 10px;
}

/* ── Profile selector ── */
.profile-block {
    background: rgba(12, 14, 18, 0.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 1.2rem;
}
.profile-block .profile-label {
    font-size: 0.72rem;
    color: #5a5550;
    margin-bottom: 0.2rem;
}
.profile-block .profile-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #faf8f5;
}
.profile-block .cluster-badge {
    display: inline-block;
    margin-top: 0.5rem;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 3px 11px;
    border-radius: 20px;
}

/* ── Empty states ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1.5rem;
}
.empty-icon {
    font-size: 2.2rem;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}
.empty-state p {
    font-size: 0.9rem;
    color: #4a4540;
    line-height: 1.7;
    margin: 0;
}

/* ── How it works strip ── */
.how-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
@media (max-width: 700px) { .how-strip { grid-template-columns: 1fr; } }
.how-step {
    background: rgba(20, 24, 32, 0.4);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 1.2rem;
}
.how-step .step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: #6ee7b7;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.how-step .step-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #e8e4df;
    margin-bottom: 0.35rem;
}
.how-step .step-desc {
    font-size: 0.78rem;
    color: #5a5550;
    line-height: 1.55;
    margin: 0;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(12, 14, 18, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e8e4df !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(20, 24, 32, 0.5) !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: #9a9590 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #6ee7b7 !important; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    margin: 1.5rem 0;
}

/* ── Results count ── */
.results-meta {
    font-size: 0.8rem;
    color: #5a5550;
    margin-bottom: 1rem;
}
.results-meta strong { color: #a78bfa; font-weight: 600; }

/* ── Live dots ── */
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: #6ee7b7;
    letter-spacing: 0.08em;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #6ee7b7;
    animation: dot-pulse 2s ease infinite;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cluster_label(user_id: int) -> tuple[str, str]:
    """Return (cluster_name, css_colour_pair) for a user_id."""
    if 1  <= user_id <= 11: return "Electronics",  "rgba(110,231,183,0.12)|#6ee7b7"
    if 12 <= user_id <= 22: return "Apparel",       "rgba(232,93,76,0.12)|#e85d4c"
    if 23 <= user_id <= 36: return "Home",          "rgba(167,139,250,0.12)|#c4b5fd"
    if 37 <= user_id <= 50: return "Books",         "rgba(251,191,36,0.12)|#fbbf24"
    return "All Categories", "rgba(255,255,255,0.06)|#9a9590"


def _score_bar(score: float, max_score: float = 1.0, delay: int = 0) -> str:
    pct = min(100, max(0, (score / max_score) * 100))
    return (
        f'<div class="score-bar-wrap" style="animation-delay:{delay*0.08}s">'
        f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct:.1f}%"></div></div>'
        f'<span class="score-label">{score:.3f}</span>'
        f'</div>'
    )


def _result_card(rank: int, row: pd.Series, score: float, desc: str, delay: int = 0) -> str:
    title  = row["title"]
    cat    = row["category"]
    price  = row["price"]
    desc_s = (desc[:200] + "…") if len(desc) > 200 else desc
    bar    = _score_bar(score, delay=delay)
    return f"""
<div class="result-card" style="animation-delay:{delay * 0.1}s">
  <span class="rank-chip">#{rank}</span>
  <div class="card-title">{title}</div>
  <div class="meta-row">
    <span class="meta-pill pill-cat">{cat}</span>
    <span class="meta-pill pill-price">₹{price:.2f}</span>
  </div>
  {bar}
  <p class="card-desc">{desc_s}</p>
</div>"""


def _rec_card(rank: int, row: pd.Series, delay: int = 0) -> str:
    title = (row["title"][:58] + "…") if len(row["title"]) > 58 else row["title"]
    return f"""
<div class="rec-card" style="animation-delay:{delay * 0.08}s">
  <div class="rec-rank">#{rank}</div>
  <div class="rec-title">{title}</div>
  <div class="rec-meta">
    <span class="rec-cat">{row['category']}</span>
    <span class="rec-price">₹{row['price']:.2f}</span>
    <span class="rec-score">★ {row['predicted_rating']:.2f}</span>
  </div>
</div>"""


# ── Cached resource loading ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_search_engine():
    import search_engine as se
    return se.init_search_engine()


@st.cache_resource(show_spinner=False)
def load_recommender():
    import recommender as rec
    return rec.build_recommender()


@st.cache_data(show_spinner=False)
def load_products_meta():
    return pd.read_csv("products.csv")


# ── Landing page ───────────────────────────────────────────────────────────────

def render_landing():
    st.markdown("""
<div class="landing-hero">
  <div class="landing-eyebrow anim-fade-up">E-Commerce Discovery Engine</div>
  <h1 class="landing-title anim-fade-up-1">
    Find products by <span class="accent">meaning</span>,<br>
  not just keywords.
  </h1>
  <p class="landing-lead anim-fade-up-2">
    This is a live demo of a two-tower retrieval system — the same architecture
    used by large-scale marketplaces. One tower understands <em>what you say</em>
    through semantic embeddings. The other learns <em>what you like</em>
    from past behaviour via matrix factorisation.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="tower-grid anim-fade-up-3">
  <div class="tower-card search-tower">
    <div class="tower-icon">◎</div>
    <div class="tower-name">Tower 1 · NLP</div>
    <div class="tower-heading">Semantic Search</div>
    <p class="tower-desc">
      Encodes product titles and descriptions into a 384-dimensional vector space
      using a Transformer encoder. Your natural-language query is mapped to the
      same space — cosine similarity finds intent, not keyword overlap.
    </p>
    <div class="tower-spec">
      <span>all-MiniLM-L6-v2</span>
      <span>384-dim</span>
      <span>200 products</span>
      <span>O(N) retrieval</span>
    </div>
  </div>

  <div class="tower-connector">
    <div class="connector-line"></div>
    <div class="connector-label">TWO-TOWER</div>
    <div class="connector-line"></div>
  </div>

  <div class="tower-card rec-tower">
    <div class="tower-icon">◈</div>
    <div class="tower-name">Tower 2 · CF</div>
    <div class="tower-heading">Personalised Feed</div>
    <p class="tower-desc">
      Builds a sparse user–item rating matrix and decomposes it with truncated SVD
      into latent taste factors. Mean-centering removes rating-scale bias so
      recommendations reflect true preference patterns.
    </p>
    <div class="tower-spec">
      <span>SVD k=10</span>
      <span>50 users</span>
      <span>91% sparse</span>
      <span>masked leaks</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="stats-row anim-fade-up-4">
  <div class="stat-item">
    <div class="stat-num">200</div>
    <div class="stat-label">Products across 4 categories</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">854</div>
    <div class="stat-label">Simulated user interactions</div>
  </div>
  <div class="stat-item">
    <div class="stat-num">384<span class="unit">d</span></div>
    <div class="stat-label">Embedding dimensions per item</div>
  </div>
  <div class="stat-item">
  <div class="stat-num">0.69</div>
    <div class="stat-label">SVD reconstruction RMSE</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="how-strip anim-fade-up-5">
  <div class="how-step">
    <div class="step-num">STEP 01</div>
    <div class="step-title">Type naturally</div>
    <p class="step-desc">Search "something to read at night" — no exact keywords needed. The encoder maps your intent into vector space.</p>
  </div>
  <div class="how-step">
    <div class="step-num">STEP 02</div>
    <div class="step-title">Vectors match</div>
    <p class="step-desc">A single matrix–vector multiply scores all 200 products by cosine similarity in under a millisecond.</p>
  </div>
  <div class="how-step">
    <div class="step-num">STEP 03</div>
    <div class="step-title">Taste predicts</div>
    <p class="step-desc">Switch demo profiles to see SVD-generated picks based on each user's rating history and latent clusters.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Enter the demo →", type="primary", use_container_width=True):
            st.session_state.view = "app"
            st.rerun()


# ── Main app ───────────────────────────────────────────────────────────────────

def render_app():
    st.markdown("""
<div class="app-nav anim-fade-up">
  <div class="app-logo">discover<span>.</span></div>
  <div class="live-indicator"><span class="live-dot"></span> LIVE DEMO</div>
</div>
""", unsafe_allow_html=True)

    col_main, col_side = st.columns([2.2, 1])

    # ── Left: Search ──────────────────────────────────────────────────────────
    with col_main:
        st.markdown("""
<div class="anim-slide-r">
  <div class="section-eyebrow">Semantic Search Tower</div>
  <div class="section-heading">What are you looking for?</div>
  <p class="section-sub">Describe what you want in plain language — the NLP encoder
  finds products by meaning, not keyword matching.</p>
</div>
""", unsafe_allow_html=True)

        query = st.text_input(
            label="search",
            placeholder='Try "cozy reading light" or "portable workstation for travel"…',
            label_visibility="collapsed",
        )

        search_ready = False
        try:
            with st.spinner("Loading embedding model…"):
                products_df_se, embedding_matrix, model = load_search_engine()
            search_ready = True
        except Exception as e:
            st.warning(f"Search engine unavailable: {e}")
            products_df_se = load_products_meta()

        if query.strip() == "":
            st.markdown("""
<div class="empty-state anim-fade-up-2">
  <div class="empty-icon">◎</div>
  <p>Start typing above to search 200 products by semantic similarity.<br>
  Your query becomes a 384-dimensional vector — results rank by cosine distance.</p>
</div>""", unsafe_allow_html=True)

        elif not search_ready:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#e85d4c;font-size:0.82rem;">⚠ Semantic model offline — showing keyword matches.</p>',
                unsafe_allow_html=True,
            )
            mask    = products_df_se["title"].str.contains(query, case=False, na=False)
            results = products_df_se[mask].head(5)
            if results.empty:
                st.markdown("""
<div class="empty-state"><div class="empty-icon">—</div>
<p>No keyword matches. Try a different query.</p></div>""", unsafe_allow_html=True)
            else:
                for i, (_, row) in enumerate(results.iterrows()):
                    st.markdown(
                        _result_card(i + 1, row, score=0.0, desc=row["description"], delay=i),
                        unsafe_allow_html=True,
                    )
        else:
            import search_engine as se
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            with st.spinner(f'Searching "{query}"…'):
                results = se.semantic_search(
                    query_text=query,
                    products_df=products_df_se,
                    embedding_matrix=embedding_matrix,
                    model=model,
                    top_k=5,
                )
            st.markdown(
                f'<p class="results-meta">Top 5 matches for <strong>"{query}"</strong> — ranked by cosine similarity</p>',
                unsafe_allow_html=True,
            )
            max_sim = float(results["similarity_score"].max()) if not results.empty else 1.0
            for i, (_, row) in enumerate(results.iterrows()):
                score = float(row["similarity_score"])
                pid   = int(row["product_id"])
                desc_row = products_df_se.loc[products_df_se["product_id"] == pid, "description"]
                desc = str(desc_row.values[0]) if not desc_row.empty else ""
                bar  = _score_bar(score, max_score=max_sim if max_sim > 0 else 1.0, delay=i)
                card = f"""
<div class="result-card" style="animation-delay:{i * 0.1}s">
  <span class="rank-chip">#{i+1}</span>
  <div class="card-title">{row['title']}</div>
  <div class="meta-row">
    <span class="meta-pill pill-cat">{row['category']}</span>
    <span class="meta-pill pill-price">₹{row['price']:.2f}</span>
  </div>
  {bar}
  <p class="card-desc">{(desc[:220] + '…') if len(desc) > 220 else desc}</p>
</div>"""
                st.markdown(card, unsafe_allow_html=True)

            with st.expander("How does semantic search work under the hood?"):
                st.markdown(f"""
**Pipeline for** `"{query}"`

1. Query encoded by `all-MiniLM-L6-v2` → unit vector **q ∈ ℝ³⁸⁴**
2. Product catalogue pre-encoded as matrix **E ∈ ℝ²⁰⁰ˣ³⁸⁴**
3. Cosine similarity via single dot product: **s = E @ q**
4. Top-5 retrieved with `np.argpartition` in **O(N)** time

| Component | Shape | Operation |
|-----------|-------|-----------|
| Query embedding | (384,) | encode once |
| Product matrix | (200, 384) | pre-computed |
| Scores | (200,) | argsort top-5 |
                """)

    # ── Right: Recommendations ────────────────────────────────────────────────
    with col_side:
        st.markdown('<div class="rec-panel anim-fade-up-2">', unsafe_allow_html=True)
        st.markdown("""
  <div class="section-eyebrow" style="color:#a78bfa;">Personalisation Tower</div>
  <div class="section-heading" style="font-size:1.15rem;">For You</div>
  <p class="section-sub" style="font-size:0.8rem;margin-bottom:1rem;">
    Pick a demo profile to see SVD collaborative-filtering picks.
  </p>
""", unsafe_allow_html=True)

        user_id = st.selectbox(
            "Demo profile",
            options=list(range(1, 51)),
            index=4,
            format_func=lambda x: f"Profile {x:02d}",
            label_visibility="collapsed",
        )

        cluster_name, colour_pair = _cluster_label(user_id)
        bg_col, fg_col = colour_pair.split("|")
        st.markdown(f"""
<div class="profile-block">
  <div class="profile-label">Active profile</div>
  <div class="profile-name">User {user_id:02d}</div>
  <span class="cluster-badge" style="background:{bg_col};color:{fg_col};">
    {cluster_name} affinity
  </span>
</div>
""", unsafe_allow_html=True)

        rec_ready = False
        try:
            with st.spinner("Building SVD model…"):
                R_hat, R_obs, user_ids, product_ids, products_df_rec = load_recommender()
            rec_ready = True
        except Exception as e:
            st.warning(f"Recommender unavailable: {e}")

        if rec_ready:
            import recommender as rec_mod
            try:
                recs = rec_mod.get_collaborative_recommendations(
                    user_id=user_id,
                    R_hat=R_hat,
                    R_observed=R_obs,
                    user_ids=user_ids,
                    product_ids=product_ids,
                    products_df=products_df_rec,
                    top_k=6,
                )
                cards_html = "".join(
                    _rec_card(i + 1, row, delay=i)
                    for i, (_, row) in enumerate(recs.iterrows())
                )
                st.markdown(cards_html, unsafe_allow_html=True)
                n_rated = recs.attrs.get("n_rated", "?")
                st.markdown(
                    f'<p style="font-size:0.68rem;color:#4a4540;margin-top:0.6rem;">'
                    f'From {n_rated} ratings · k=10 latent factors</p>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Could not generate recommendations: {e}")
        else:
            st.markdown("""
<div class="empty-state" style="padding:2rem 0.5rem;">
  <p>Recommender not loaded.<br>Ensure ratings.csv is present.</p>
</div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)
    if st.button("← Back to overview", type="secondary"):
        st.session_state.view = "landing"
        st.rerun()


# ── Route ──────────────────────────────────────────────────────────────────────

if st.session_state.view == "landing":
    render_landing()
else:
    render_app()
