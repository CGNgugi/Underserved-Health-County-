"""
kenya_health.py — Kenya Health Gap Dashboard v2
================================================
Improvements over v1:
  • New page: KDHS Raw Data Ingestor — upload raw DHS CSV, auto-map columns,
    compute domain scores, and predict UCS without any pre-processing step.
  • Prediction engine uses a pipeline trained on the existing county_ucs_final.csv
    data so it works even without model .pkl files.
  • Cleaner layout, better empty-state handling, richer SHAP explorer.
  • All original pages preserved and improved.

Author : Cynthia Ngugi (138725) | MSc Data Science, Strathmore University
Run    : streamlit run kenya_health.py
"""

import warnings, os, json, io
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    import folium
    from streamlit.components.v1 import html as st_html
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kenya Health Equity Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Palette from pasted .py — blue primary, clean MoH public-health style
COLORS = {
    "blue":         "#2F80ED",   # primary blue
    "blue_dark":    "#1B4F9C",   # dark blue (headings, active nav)
    "blue_light":   "#EAF3FF",   # very light blue (hover, bg tint)
    "orange":       "#F2994A",   # accent / KPI values
    "orange_dark":  "#C75B12",   # critical / Cluster 1
    "red":          "#D94A38",   # alert
    "gray":         "#4F4F4F",   # body text
    "gray_light":   "#F8FAFC",   # page background
    "border":       "#D9E2EC",   # borders
    "black":        "#222222",   # headings
    "white":        "#FFFFFF",
    "purple":       "#7E57C2",
    "green":        "#2E7D32",
    "amber":        "#F2994A",
    "light":        "#F8FAFC",
    "mid":          "#EEF3F8",
    "teal":         "#2D9CDB",
    "navy":         "#1B4F9C",
    "gold":         "#F2994A",
}

DOMAINS = [
    "Healthcare Access Index",
    "Population Vulnerability Index",
    "Immunization Coverage Index",
    "Disease Burden Index",
]

DOMAIN_META = {
    "Healthcare Access Index":        {"short": "Healthcare Access",  "color": "#C8690A", "icon": ""},
    "Population Vulnerability Index": {"short": "Pop. Vulnerability", "color": "#1A1A2E", "icon": ""},
    "Immunization Coverage Index":    {"short": "Immunization",       "color": "#2B7A3F", "icon": ""},
    "Disease Burden Index":           {"short": "Disease Burden",     "color": "#6D28D9", "icon": ""},
}

DOMAIN_SHORT = {d: DOMAIN_META[d]["short"] for d in DOMAINS}

# KDHS column families → domain mapping
# Keys are partial strings that might appear in raw DHS column headers
KDHS_COLUMN_MAP = {
    "Healthcare Access Index": [
        "antenatal", "anc", "skilled_birth", "delivery", "postnatal", "pnc",
        "insurance", "nhif", "facility_dist", "travel_time", "distance",
        "health_worker", "hrh", "doctor", "nurse", "midwife",
    ],
    "Population Vulnerability Index": [
        "wealth", "poverty", "poorest", "education", "no_school",
        "water", "sanitation", "toilet", "wash", "household_size",
        "dependency", "literacy",
    ],
    "Immunization Coverage Index": [
        "bcg", "dpt", "polio", "measles", "vaccine", "immuniz", "vitamin_a",
        "deworming", "fully_immunized", "vaccination",
    ],
    "Disease Burden Index": [
        "stunting", "wasting", "underweight", "malnutrition", "anaemia",
        "malaria", "fever", "diarrhea", "ari", "tb", "hiv",
    ],
}

KENYA_COORDS = {
    "Mombasa": [-4.04, 39.68], "Kwale": [-4.54, 39.45], "Kilifi": [-3.53, 39.60],
    "Tana River": [-1.79, 39.97], "Lamu": [-2.22, 40.10], "Taita Taveta": [-3.39, 38.44],
    "Garissa": [-0.45, 40.12], "Wajir": [1.75, 40.06], "Mandera": [3.94, 40.65],
    "Marsabit": [2.54, 37.98], "Isiolo": [0.35, 38.49], "Meru": [0.16, 37.96],
    "Tharaka-Nithi": [-0.21, 37.67], "Embu": [-0.53, 37.45], "Kitui": [-1.37, 38.01],
    "Machakos": [-1.52, 37.26], "Makueni": [-2.24, 37.56], "Nyandarua": [-0.65, 36.43],
    "Nyeri": [-0.42, 36.96], "Kirinyaga": [-0.58, 37.28], "Murang'a": [-0.79, 36.96],
    "Kiambu": [-1.17, 36.84], "Nairobi": [-1.29, 36.82], "Kajiado": [-1.88, 36.78],
    "Kericho": [-0.37, 35.28], "Bomet": [-0.52, 35.15], "Nakuru": [-0.29, 36.07],
    "Narok": [-1.08, 35.86], "Baringo": [1.17, 35.97], "Elgeyo Marakwet": [0.75, 35.50],
    "West Pokot": [1.23, 35.03], "Samburu": [1.19, 36.75], "Trans-Nzoia": [1.03, 35.00],
    "Uasin Gishu": [0.54, 35.29], "Nandi": [0.20, 35.00], "Kakamega": [0.28, 34.75],
    "Vihiga": [0.04, 34.60], "Bungoma": [0.53, 34.56], "Busia": [0.44, 34.25],
    "Siaya": [-0.06, 34.18], "Kisumu": [-0.02, 34.76], "Homa Bay": [-0.53, 34.45],
    "Migori": [-1.15, 34.38], "Kisii": [-0.69, 34.76], "Nyamira": [-0.56, 34.96],
    "Laikipia": [0.30, 36.78], "Turkana": [3.46, 35.54],
}

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #222222;
    background: #FFFFFF;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0.35rem !important;
    padding-left: 0.65rem !important;
    padding-right: 0.65rem !important;
    max-width: 100% !important;
}

div[data-testid="stVerticalBlock"] > div { gap: 0.22rem !important; }
.element-container { margin-bottom: 0.12rem !important; }
.stPlotlyChart { margin-bottom: 0 !important; }
hr { margin: 0.35rem 0 !important; border-color: #D9E2EC !important; }

/* Header — white with blue bottom border, matching MoH COVID dashboard */
.moh-header {
    background: #FFFFFF;
    border-bottom: 2px solid #2F80ED;
    padding: 7px 12px 5px 12px;
    margin: 0 -0.65rem 0.25rem -0.65rem;
}
.moh-header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo-row {
    display: flex;
    gap: 7px;
    align-items: center;
    min-width: 145px;
}
.logo-box {
    height: 28px;
    min-width: 42px;
    border: 1px solid #D9E2EC;
    color: #2F80ED;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 800;
    background: #F8FAFC;
}
.titles {
    flex: 1;
    text-align: center;
}
.ministry {
    color: #2F80ED;
    font-size: 1.52rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    line-height: 1.05;
    text-transform: uppercase;
}
.subtitle-line {
    color: #2F80ED;
    font-size: 0.82rem;
    line-height: 1.2;
}
.badge-gold {
    color: #C75B12;
    font-size: 0.74rem;
    font-weight: 700;
    min-width: 145px;
    text-align: right;
}

/* Navigation buttons — compact blue tab style */
.stButton button {
    background: #FFFFFF !important;
    color: #2F80ED !important;
    border: 1px solid #D9E2EC !important;
    border-radius: 0 !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    padding: 0.30rem 0.25rem !important;
    box-shadow: none !important;
}
.stButton button:hover {
    background: #EAF3FF !important;
    border-color: #2F80ED !important;
    color: #1B4F9C !important;
}
.stButton button[kind="primary"] {
    background: #EAF3FF !important;
    border-color: #2F80ED !important;
    color: #1B4F9C !important;
    border-bottom: 3px solid #F2994A !important;
}

/* Nav row container */
div[data-testid="stHorizontalBlock"]:first-of-type {
    background: #FFFFFF !important;
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 0 4px 0 !important;
    border-bottom: 2px solid #D9E2EC;
}
div[data-testid="stHorizontalBlock"]:first-of-type > div {
    padding: 0 !important;
    flex: 1 !important;
}

/* Page titles */
.pg-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #2F80ED;
    margin: 4px 0 0 0;
    padding: 0;
    line-height: 1.22;
}
.pg-sub {
    font-size: 0.68rem;
    color: #4F4F4F;
    margin: 0 0 4px 0;
}

/* KPI tiles */
.kpi {
    background: #FFFFFF;
    border-radius: 0;
    padding: 5px 7px;
    border: 1px solid #D9E2EC;
    border-top: 2px solid var(--kc, #2F80ED);
    text-align: center;
    margin-bottom: 2px;
    box-shadow: none;
}
.kpi .v {
    font-size: 1.08rem;
    font-weight: 800;
    color: #F2994A;
    line-height: 1.05;
}
.kpi .l {
    font-size: 0.55rem;
    color: #2F80ED;
    text-transform: none;
    letter-spacing: 0.01em;
    margin-top: 1px;
}

/* Streamlit metrics */
div[data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    color: #F2994A !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.60rem !important;
    color: #2F80ED !important;
}
div[data-testid="stMetricDelta"] { font-size: 0.62rem !important; }

/* Alert boxes */
.box-info, .box-warn, .box-ok, .box-err {
    background: #FFFFFF;
    border: 1px solid #D9E2EC;
    border-left: 4px solid #2F80ED;
    padding: 7px 10px;
    border-radius: 0;
    font-size: 0.76rem;
    color: #222222;
    margin: 4px 0;
    line-height: 1.35;
}
.box-warn { border-left-color: #F2994A; background: #FFF8F0; }
.box-ok   { border-left-color: #2D9CDB; background: #F4FBFF; }
.box-err  { border-left-color: #D94A38; background: #FFF4F2; }

/* Section headings */
h1, h2, h3, h4, h5, h6 { color: #2F80ED !important; }
h5 { font-size: 0.84rem !important; margin: 0.2rem 0 0.15rem 0 !important; }

/* Forms */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stFileUploader"] label {
    font-size: 0.70rem !important;
    color: #4F4F4F !important;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #2F80ED !important;
    border-radius: 0 !important;
    background: #F8FAFC !important;
}

/* Data tables */
.stDataFrame { border: 1px solid #D9E2EC !important; }
.stDataFrame thead tr th {
    background: #EAF3FF !important;
    color: #1B4F9C !important;
    font-size: 0.70rem !important;
}

/* Progress bar */
.stProgress > div > div > div > div { background-color: #F2994A !important; }
.stProgress > div > div > div { height: 7px !important; background: #EAF3FF !important; }

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px; height: 18px;
    border-radius: 0;
    background: #2F80ED;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    margin-right: 5px;
}

/* Prediction card */
.pred-card {
    border-radius: 0;
    padding: 12px 14px;
    margin: 6px 0;
    background: #EAF3FF;
    color: #222222;
    border-left: 5px solid #F2994A;
}
.pred-card .county-name { font-size: 1rem; font-weight: 700; }
.pred-card .ucs-big { font-size: 2rem; font-weight: 800; line-height: 1; color: #F2994A; }
.pred-card .ucs-sub { font-size: 0.72rem; color: #4F4F4F; }

small, .stCaption {
    font-size: 0.65rem !important;
    color: #4F4F4F !important;
}

/* Hide sidebar */
section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_main_data():
    for p in ["county_ucs_final.csv", "./county_ucs_final.csv", "../county_ucs_final.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
            for county, (lat, lon) in KENYA_COORDS.items():
                if county in df.index:
                    df.loc[county, "lat"] = lat
                    df.loc[county, "lon"] = lon
            return df
    return None

@st.cache_data
def load_shap_data():
    for p in ["shap_values.csv", "./shap_values.csv", "../shap_values.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
    return None

df = load_main_data()
shap_df = load_shap_data()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def kpi(val, label, color=COLORS["navy"]):
    st.markdown(f'<div class="kpi" style="--kc:{color}"><div class="v">{val}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)

def box(text, kind="info"):
    prefix = {"info":"Note","warn":"Note","ok":"","err":"Error"}.get(kind,"")
    leader = f"<b>{prefix}:</b> " if prefix else ""
    st.markdown(f'<div class="box-{kind}">{leader}{text}</div>', unsafe_allow_html=True)

def ucs_color(score):
    if score >= 70: return COLORS["red"]
    if score >= 40: return COLORS["orange"]
    return COLORS["green"]

def ucs_label(score):
    if score >= 70: return "Most Underserved"
    if score >= 40: return "Moderate"
    return "Well Served"

def norm_val(val, series):
    mn, mx = series.min(), series.max()
    return (val - mn) / (mx - mn) * 100 if mx > mn else 0.0

def radar_chart(values_dict, county_name, height=240):
    cats = list(values_dict.keys()); cats.append(cats[0])
    vals = list(values_dict.values()); vals.append(vals[0])
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(200,105,10,0.15)",
        line=dict(color=COLORS["navy"], width=2),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], ticksuffix="%", tickfont_size=8)),
        showlegend=False, height=height, margin=dict(t=10,b=10,l=30,r=30)
    )
    return fig

def build_folium_map(data, height=380):
    m = folium.Map(location=[0.5, 37.5], zoom_start=5.8, tiles="CartoDB positron")
    for county, row in data.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon): continue
        ucs_v = row.get("UCS", 0)
        col = "red" if ucs_v >= 70 else "orange" if ucs_v >= 40 else "green"
        popup = f"<b>{county}</b><br>UCS: {ucs_v:.1f} — {ucs_label(ucs_v)}<hr>"
        for d in DOMAINS:
            if d in row:
                popup += f"{DOMAIN_META[d]['short']}: {row[d]:.3f}<br>"
        folium.CircleMarker(
            location=[lat, lon],
            radius=5 + ucs_v / 18,
            popup=folium.Popup(popup, max_width=220),
            color=col, fill=True, fillColor=col, fillOpacity=0.72, weight=1.5
        ).add_to(m)
    return m

# ─────────────────────────────────────────────────────────────
# KDHS INGESTION ENGINE
# ─────────────────────────────────────────────────────────────

def detect_column_domain(col_name: str):
    """Return which domain a raw KDHS column most likely belongs to."""
    col_lower = col_name.lower()
    for domain, keywords in KDHS_COLUMN_MAP.items():
        if any(kw in col_lower for kw in keywords):
            return domain
    return None

def auto_map_columns(raw_df: pd.DataFrame):
    """Automatically map raw columns to domains."""
    mapping = {d: [] for d in DOMAINS}
    for col in raw_df.columns:
        domain = detect_column_domain(col)
        if domain:
            mapping[domain].append(col)
    return mapping

def compute_domain_score(raw_df: pd.DataFrame, cols: list[str], domain: str) -> pd.Series:
    """Compute a single domain score (0–100) from raw columns using PCA or mean."""
    if not cols:
        return pd.Series(np.nan, index=raw_df.index)

    sub = raw_df[cols].copy()
    # coerce to numeric; fill with column median
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.fillna(sub.median())

    # Polarity fix: if column means suggest "higher = better", flip
    # (e.g. immunization coverage — higher value means LOWER burden → invert)
    good_keywords = ["immuniz","vaccine","bcg","dpt","polio","measles","vitamin","insur","skilled","antenatal"]
    for c in cols:
        if any(kw in c.lower() for kw in good_keywords):
            sub[c] = 100 - sub[c].clip(0, 100) if sub[c].max() <= 100 else -sub[c]

    if len(cols) == 1 or not SKLEARN_OK:
        score = sub[cols[0]] if len(cols) == 1 else sub.mean(axis=1)
    else:
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(sub)
            pca = PCA(n_components=1)
            score = pd.Series(pca.fit_transform(scaled).ravel(), index=raw_df.index)
        except Exception:
            score = sub.mean(axis=1)

    # Normalise to 0–100
    mn, mx = score.min(), score.max()
    if mx > mn:
        score = (score - mn) / (mx - mn) * 100
    else:
        score = pd.Series(50.0, index=raw_df.index)
    return score

def compute_ucs(domain_scores: pd.DataFrame) -> pd.Series:
    """Compute final UCS from four domain scores using CV weighting."""
    cv_weights = {}
    for d in DOMAINS:
        if d in domain_scores.columns:
            s = domain_scores[d].dropna()
            mu = s.mean()
            cv = (s.std() / abs(mu)) if mu != 0 else 0.0
            cv_weights[d] = cv

    total_cv = sum(cv_weights.values())
    weights = {d: cv_weights[d] / total_cv if total_cv > 0 else 0.25 for d in DOMAINS}

    raw_ucs = sum(domain_scores[d] * weights[d] for d in DOMAINS if d in domain_scores.columns)
    mn, mx = raw_ucs.min(), raw_ucs.max()
    return ((raw_ucs - mn) / (mx - mn) * 100) if mx > mn else raw_ucs

def build_prediction_model(reference_df: pd.DataFrame):
    """Train a simple RF model on existing data to predict UCS from domain scores."""
    if not SKLEARN_OK or reference_df is None:
        return None
    avail = [d for d in DOMAINS if d in reference_df.columns]
    if not avail or "UCS" not in reference_df.columns:
        return None
    X = reference_df[avail].fillna(reference_df[avail].median())
    y = reference_df["UCS"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

# ── TOP NAVIGATION BAR (replaces sidebar nav) ──────────────────────────────
PAGES = [
    ("", "Overview"),
    ("", "Map"),
    ("", "PCA Analysis"),
    ("", "County Deep Dive"),
    ("", "ML & SHAP"),
    ("", "KDHS Predictor"),
]

# Initialise page state
if "page" not in st.session_state:
    st.session_state.page = "Overview"

# ── MOH HEADER ───────────────────────────────────────────────────────────────
st.markdown('''
<div class="moh-header">
  <div class="moh-header-top">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Coat_of_arms_of_Kenya.svg/60px-Coat_of_arms_of_Kenya.svg.png"
         class="coat" alt="Kenya Coat of Arms">
    <div class="titles">
      <div class="ministry">Ministry of Health — Republic of Kenya</div>
      <div class="subtitle-line">Kenya Health Equity Monitor &nbsp;·&nbsp; Underserved County Score (UCS) &nbsp;·&nbsp; 47 Counties</div>
    </div>
    <span class="badge-gold">KDHS 2020 / 2022</span>
  </div>
</div>''', unsafe_allow_html=True)

# ── STREAMLIT NATIVE TAB BAR ─────────────────────────────────────────────────
# Use st.columns as clickable tab buttons — works without JS
_tab_labels = [label for icon, label in PAGES]
_tab_cols = st.columns(len(PAGES))
for i, (col, (icon, label)) in enumerate(zip(_tab_cols, PAGES)):
    with col:
        _btn_style = "primary" if st.session_state.page == label else "secondary"
        if st.button(label, key=f"nav_{label}",
                     type=_btn_style, use_container_width=True):
            st.session_state.page = label
            st.rerun()

page = st.session_state.page

# Active page indicator bar
_active_idx = [l for _, l in PAGES].index(page)
_bar_html = '<div style="display:flex;background:#E8F0E8;border-bottom:2px solid #C8D8C8;margin-bottom:2px">' 
for i, (_, label) in enumerate(PAGES):
    _w = f"{100/len(PAGES):.1f}%"
    _bg = "#2F80ED" if i == _active_idx else "transparent"
    _bar_html += f'<div style="flex:1;height:3px;background:{_bg}"></div>'
_bar_html += '</div>'
st.markdown(_bar_html, unsafe_allow_html=True)

# ── COMPACT FILTER STRIP (below header) ─────────────────────────────────────
if df is not None:
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1, 1, 1, 1])
    with fc1:
        ucs_range = st.slider("UCS Range Filter", 0, 100, (0, 100), 5,
                              label_visibility="collapsed",
                              help="Filter counties by UCS score range")
    with fc2:
        st.markdown(f'<div style="font-size:.7rem;color:#6B7280;padding:6px 0"><b style="color:#006633">{len(df)}</b> counties', unsafe_allow_html=True)
    with fc3:
        st.markdown(f'<div style="font-size:.7rem;color:#6B7280;padding:6px 0">Mean UCS <b style="color:#006633">{df["UCS"].mean():.1f}</b></div>', unsafe_allow_html=True)
    with fc4:
        n_crit = len(df[df["UCS"] >= 70])
        st.markdown(f'<div style="font-size:.7rem;color:#6B7280;padding:6px 0">Critical <b style="color:#C0392B">{n_crit}</b></div>', unsafe_allow_html=True)
    with fc5:
        if "Anomaly" in df.columns:
            n_a = (df["Anomaly"] == "Anomaly").sum()
            st.markdown(f'<div style="font-size:.7rem;color:#6B7280;padding:6px 0">Anomalies <b style="color:#C8690A">{n_a}</b></div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin:3px 0'>", unsafe_allow_html=True)
else:
    ucs_range = (0, 100)
    st.markdown("<hr style='margin:3px 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────

if page == "Overview":
    if df is None:
        st.error("county_ucs_final.csv not found. Upload via the KDHS Predictor page or place the file in the app directory.")
        st.stop()

    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]

    st.markdown('<p class="pg-title">Kenya Healthcare Access Inequality Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Underserved County Score (UCS) · KDHS 2020 & 2022 · 47 Counties · Higher Score = More Underserved · Mean UCS = 56.67 · Range 0–100</p>', unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi(len(fdf), "Counties", COLORS["blue"])
    with k2: kpi(f"{fdf['UCS'].mean():.1f}", "Mean UCS", COLORS["orange"])
    with k3: kpi(f"{fdf['UCS'].std():.1f}", "Std Dev", COLORS["purple"])
    worst = fdf["UCS"].idxmax()
    with k4: kpi(f"{fdf['UCS'].max():.0f}", f"Worst · {worst[:8]}", COLORS["red"])
    best = fdf["UCS"].idxmin()
    with k5: kpi(f"{fdf['UCS'].min():.0f}", f"Best · {best[:8]}", COLORS["green"])

    st.markdown("---")
    left, right = st.columns([1.1, 1])

    with left:
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### Top 10 Underserved")
            top10 = fdf["UCS"].sort_values(ascending=False).head(10).reset_index()
            top10.columns = ["County","UCS"]
            fig = px.bar(top10, x="UCS", y="County", orientation="h",
                color="UCS", color_continuous_scale=["#E8A87C","#C0392B"], range_color=[0,100])
            fig.update_traces(texttemplate="%{x:.0f}", textposition="outside", textfont_size=8)
            fig.update_layout(yaxis_autorange="reversed", height=240,
                margin=dict(l=5,r=30,t=5,b=5), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False,
                yaxis_tickfont_size=8,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with b2:
            st.markdown("##### Top 10 Best Served")
            bot10 = fdf["UCS"].sort_values().head(10).reset_index()
            bot10.columns = ["County","UCS"]
            fig2 = px.bar(bot10, x="UCS", y="County", orientation="h",
                color="UCS", color_continuous_scale=["#5BA08A","#1A1A2E"], range_color=[0,100])
            fig2.update_traces(texttemplate="%{x:.0f}", textposition="outside", textfont_size=8)
            fig2.update_layout(yaxis_autorange="reversed", height=240,
                margin=dict(l=5,r=30,t=5,b=5), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False,
                yaxis_tickfont_size=8,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Domain Score Distributions")
        avail_domains = [d for d in DOMAINS if d in fdf.columns]
        if avail_domains:
            dmelt = fdf[avail_domains].copy()
            dmelt.columns = [DOMAIN_META[d]["short"] for d in avail_domains]
            dmelt = dmelt.melt(var_name="Domain", value_name="Score").dropna()
            fig_box = px.box(dmelt, x="Domain", y="Score",
                color="Domain",
                color_discrete_sequence=[DOMAIN_META[d]["color"] for d in avail_domains])
            fig_box.update_layout(height=175, showlegend=False,
                margin=dict(l=5,r=5,t=5,b=5), xaxis_title="", yaxis_title="",
                xaxis_tickfont_size=9,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, use_container_width=True)

    with right:
        st.markdown("##### Domain Architecture (CV Weighted)")
        fig_d = go.Figure(go.Pie(
            labels=[f"{""} {DOMAIN_META[d]['short']}" for d in DOMAINS],
            values=[25,25,25,25], hole=0.58,
            marker_colors=[DOMAIN_META[d]["color"] for d in DOMAINS],
            textinfo="label", textfont_size=9
        ))
        fig_d.update_layout(height=160, margin=dict(l=10,r=10,t=5,b=5), showlegend=False,
            annotations=[dict(text="CV Weighted<br>4 Domains", x=0.5, y=0.5, font_size=8, showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("##### UCS Map")
        mf = st.selectbox("Filter", ["All Counties","Cluster 1 — Underserved (UCS ≥70)","Cluster 2 — Moderately Served (UCS <70)"], key="ov_mf", label_visibility="collapsed")
        mdf = fdf.copy()
        if mf == "Cluster 1 — Underserved (UCS ≥70)":     mdf = mdf[mdf["UCS"] >= 70]
        elif mf == "Cluster 2 — Moderately Served (UCS <70)": mdf = mdf[mdf["UCS"] < 70]

        if FOLIUM_OK and "lat" in mdf.columns:
            m = build_folium_map(mdf, height=310)
            st_html(m._repr_html_(), height=310, scrolling=False)
            st.caption("High (70+) · Moderate (40-70) · Well Served (<40) · Click markers for detail")
        else:
            fig_sc = px.scatter(mdf.reset_index(), x="lon", y="lat", color="UCS",
                size="UCS", hover_name="index", color_continuous_scale=[[0,"#2B7A3F"],[0.5,"#F5F6FA"],[1,"#C0392B"]],
                range_color=[0,100])
            fig_sc.update_layout(height=310, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    n_crit = len(fdf[fdf["UCS"] >= 70])
    box(f"**{n_crit} counties** critically underserved (UCS ≥ 70). Mean UCS **{fdf['UCS'].mean():.1f}** (range 0–100). "
        f"ASAL counties dominate the top 10: Wajir (100.0), Marsabit (97.5), Turkana (96.4), Tana River (93.0), Samburu (89.9). "
        f"Population Vulnerability is the strongest domain driver (r=0.83); Immunization Coverage is operationally independent (r=0.04). "
        f"Use the **KDHS Predictor** tab to score new county data.", "info")

# ─────────────────────────────────────────────────────────────
# PAGE: MAP
# ─────────────────────────────────────────────────────────────

elif page == "Map":
    if df is None:
        st.error("Data not found."); st.stop()

    st.markdown('<p class="pg-title">Interactive Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Geographic distribution of healthcare underservice across all 47 counties</p>', unsafe_allow_html=True)

    c1,c2 = st.columns([2,1])
    with c1:
        mf = st.selectbox("Filter by UCS", ["All","Critical (≥70)","High (50–70)","Moderate (30–50)","Low (<30)"])
    with c2:
        show_labels = st.checkbox("Label anomalies", value=True)

    mdf = df.copy()
    if mf == "Critical (≥70)":    mdf = mdf[mdf["UCS"] >= 70]
    elif mf == "High (50–70)":    mdf = mdf[(mdf["UCS"] >= 50) & (mdf["UCS"] < 70)]
    elif mf == "Moderate (30–50)":mdf = mdf[(mdf["UCS"] >= 30) & (mdf["UCS"] < 50)]
    elif mf == "Low (<30)":       mdf = mdf[mdf["UCS"] < 30]

    st.caption(f"Showing {len(mdf)} / 47 counties")

    if FOLIUM_OK and "lat" in mdf.columns:
        m = folium.Map(location=[0.5, 37.5], zoom_start=6, tiles="CartoDB positron")
        for county, row in mdf.iterrows():
            lat, lon = row.get("lat"), row.get("lon")
            if pd.isna(lat) or pd.isna(lon): continue
            ucs_v = row["UCS"]
            col = "red" if ucs_v >= 70 else "orange" if ucs_v >= 50 else "beige" if ucs_v >= 30 else "green"
            popup = f"<div style='width:200px'><b>{county}</b><br><b>UCS:</b> {ucs_v:.1f}<br><hr>"
            for d in DOMAINS:
                if d in row:
                    popup += f"• {DOMAIN_META[d]['short']}: {row[d]:.3f}<br>"
            if "Anomaly" in row and row["Anomaly"] == "Anomaly":
                popup += "<br><b>Anomaly flagged</b>"
            popup += "</div>"
            is_anom = show_labels and "Anomaly" in row and row["Anomaly"] == "Anomaly"
            if is_anom:
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup, max_width=230),
                    icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
                    tooltip=f"{county}"
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[lat, lon], radius=6 + ucs_v / 16,
                    popup=folium.Popup(popup, max_width=230),
                    color=col, fill=True, fillColor=col, fillOpacity=0.72, weight=1.5,
                    tooltip=f"{county}: UCS {ucs_v:.1f}"
                ).add_to(m)
        st_html(m._repr_html_(), height=520, scrolling=False)
    else:
        fig_sc = px.scatter(mdf.reset_index(), x="lon", y="lat", color="UCS",
            size="UCS", hover_name="index", color_continuous_scale=[[0,"#2B7A3F"],[0.5,"#F5F6FA"],[1,"#C0392B"]],
            range_color=[0,100], title="County Locations (proxy scatter)")
        fig_sc.update_layout(height=480)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("| | UCS Range | Status |")
    st.markdown("|---|---|---|")
    st.markdown("|  | ≥70 | Critical underservice |")
    st.markdown("|  | 50–70 | High underservice |")
    st.markdown("|  | 30–50 | Moderate |")
    st.markdown("|  | <30 | Relatively well served |")

# ─────────────────────────────────────────────────────────────
# PAGE: PCA ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "PCA Analysis":
    if df is None: st.error("Data not found."); st.stop()
    if not SKLEARN_OK: st.warning("scikit-learn required for PCA."); st.stop()

    st.markdown('<p class="pg-title">Principal Component Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Dimensionality reduction to understand feature composition and county clustering</p>', unsafe_allow_html=True)

    avail = [d for d in DOMAINS if d in df.columns]
    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]
    X = fdf[avail].dropna()
    if len(X) < 4: st.warning("Insufficient data for PCA."); st.stop()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca_full = PCA().fit(Xs)
    var_exp = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(var_exp)

    col_v, col_l = st.columns([2,1])
    with col_v:
        st.markdown("##### Variance Explained by Component")
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(var_exp))],
            y=var_exp, name="Individual", marker_color=COLORS["navy"]))
        fig_var.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumvar))],
            y=cumvar, name="Cumulative", yaxis="y2",
            line=dict(color=COLORS["red"], width=2)))
        fig_var.update_layout(height=270, yaxis=dict(title="Variance", tickformat=".0%"),
            yaxis2=dict(title="Cumulative", overlaying="y", side="right", tickformat=".0%"),
            legend=dict(orientation="h",y=1.12,x=0.5,xanchor="center"),
            margin=dict(l=40,r=40,t=35,b=30),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_var, use_container_width=True)

    with col_l:
        st.markdown("##### Component Loadings")
        loadings = pd.DataFrame(
            pca_full.components_.T,
            columns=[f"PC{i+1}" for i in range(len(avail))],
            index=[DOMAIN_META[d]["short"] for d in avail]
        )
        st.dataframe(loadings.round(3), use_container_width=True, height=240)
        st.caption(f"PC1: {var_exp[0]*100:.1f}% · PC1+PC2: {cumvar[1]*100:.1f}%")

    st.markdown("---")
    st.markdown("##### County Distribution in PCA Space")
    pca2 = PCA(n_components=2).fit(Xs)
    coords = pca2.transform(Xs)
    pca_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1],
        "County": X.index, "UCS": fdf.loc[X.index, "UCS"]})
    if "Cluster" in fdf.columns:
        pca_df["Cluster"] = fdf.loc[X.index, "Cluster"].astype(str)
        color_col = "Cluster"
    else:
        color_col = "UCS"

    cp1, cp2 = st.columns([2,1])
    with cp1:
        fig_sc = px.scatter(pca_df, x="PC1", y="PC2", color=color_col,
            hover_name="County", size="UCS",
            color_continuous_scale=[[0,"#2B7A3F"],[0.5,"#F5F6FA"],[1,"#C0392B"]] if color_col == "UCS" else None,
            title="Counties in PCA space")
        fig_sc.update_layout(height=380, margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    with cp2:
        st.markdown("##### PC1 Driver Loadings")
        pc1_load = loadings["PC1"].sort_values()
        for domain, val in pc1_load.items():
            bar = int(abs(val) * 100)
            arrow = "↑" if val > 0 else "↓"
            col_b = COLORS["red"] if val > 0 else COLORS["green"]
            st.markdown(f"""
            <div style="font-size:.78rem;margin:3px 0">
              <span style="font-weight:600">{arrow} {domain}</span><br>
              <div style="background:#e8ecf4;border-radius:4px;height:8px;width:100%;overflow:hidden">
                <div style="background:{col_b};height:100%;width:{bar}%;border-radius:4px"></div>
              </div>
              <span style="color:#888;font-size:.68rem">{val:+.3f}</span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE: COUNTY DEEP DIVE
# ─────────────────────────────────────────────────────────────

elif page == "County Deep Dive":
    if df is None: st.error("Data not found."); st.stop()

    st.markdown('<p class="pg-title">County Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Domain profiles, percentile ranks, SHAP drivers and multi-county comparison</p>', unsafe_allow_html=True)

    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]
    county_list = sorted(df.index.tolist())

    sc, mc = st.columns([3,1])
    with sc: selected = st.selectbox("County", county_list, label_visibility="collapsed")
    with mc: mode = st.selectbox("Mode", ["Single","Compare 2","Compare 3"], label_visibility="collapsed")

    cdata = df.loc[selected]
    ucs_v = cdata["UCS"]
    sts_lbl, sts_col = ucs_label(ucs_v), ucs_color(ucs_v)

    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("UCS Score", f"{ucs_v:.1f}", sts_lbl)
    with m2:
        cl = str(cdata.get("Cluster_label", cdata.get("Cluster","N/A")))
        st.metric("Cluster", cl[:20]+"…" if len(cl)>20 else cl)
    with m3:
        an = cdata.get("Anomaly","Normal")
        st.metric("Anomaly", "Yes" if an=="Anomaly" else "No")
    with m4:
        ap = cdata.get("Anomaly_prob",0)*100
        st.metric("Anomaly Prob", f"{ap:.1f}%")
    with m5:
        rank = int((df["UCS"] >= ucs_v).sum())
        st.metric("Rank (worst)", f"#{rank}/47")

    st.markdown("---")
    avail_domains = [d for d in DOMAINS if d in df.columns]
    rad_c, dom_c, sh_c = st.columns([1.2,1,1])

    with rad_c:
        if mode == "Single":
            st.markdown(f"##### ️ {selected}")
            vals = {DOMAIN_META[d]["short"]: norm_val(cdata[d], df[d]) for d in avail_domains}
            st.plotly_chart(radar_chart(vals, selected, 250), use_container_width=True)
        else:
            st.markdown("##### Comparison")
            comp = [selected]
            rem = [c for c in county_list if c != selected]
            c2 = st.selectbox("2nd County", rem, key="c2")
            comp.append(c2)
            if mode == "Compare 3":
                comp.append(st.selectbox("3rd County", [c for c in rem if c!=c2], key="c3"))
            cats = [DOMAIN_META[d]["short"] for d in avail_domains]; cats.append(cats[0])
            palette = [COLORS["blue"],COLORS["red"],COLORS["green"]]
            fig_r = go.Figure()
            for i, cn in enumerate(comp):
                if cn not in df.index: continue
                vs = [norm_val(df.loc[cn,d], df[d]) for d in avail_domains]; vs.append(vs[0])
                fig_r.add_trace(go.Scatterpolar(r=vs, theta=cats,
                    fill="toself" if i==0 else "none",
                    fillcolor="rgba(200,200,200,0.12)",
                    line=dict(color=palette[i%3],width=2), name=cn,
                    hovertemplate="%{theta}: %{r:.1f}<extra></extra>"))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True,range=[0,100],ticksuffix="%",tickfont_size=8)),
                showlegend=True, legend=dict(orientation="h",y=-0.12,x=0.5,xanchor="center",font_size=9),
                height=250, margin=dict(t=10,b=30,l=30,r=30))
            st.plotly_chart(fig_r, use_container_width=True)

    with dom_c:
        st.markdown("##### Domain Scores")
        for d in avail_domains:
            n = norm_val(cdata[d], df[d])
            st.markdown(f"<span style='font-size:.78rem;font-weight:600'>{""} {DOMAIN_META[d]['short']}</span>", unsafe_allow_html=True)
            st.progress(n/100, text=f"{n:.0f}%")
        st.markdown("<span style='font-size:.72rem;font-weight:600;color:#555'>Percentile vs 47 counties</span>", unsafe_allow_html=True)
        pct_rows = []
        for d in avail_domains:
            p = (df[d] <= cdata[d]).sum() / len(df) * 100
            pct_rows.append({"Domain": DOMAIN_META[d]["short"], "Pct": f"{p:.0f}th"})
        st.dataframe(pd.DataFrame(pct_rows).set_index("Domain"), height=130, use_container_width=True)

    with sh_c:
        st.markdown("##### SHAP Drivers")
        if shap_df is not None:
            rdf = shap_df[shap_df.index==selected] if "County" not in shap_df.columns else shap_df[shap_df["County"]==selected]
            if len(rdf) > 0:
                srow = rdf.iloc[0]
                dcols = [c for c in shap_df.columns if c in DOMAINS]
                sv = pd.Series({DOMAIN_META[c]["short"]: abs(srow[c]) for c in dcols}).sort_values()
                fig_shap = px.bar(sv, orientation="h", color=sv.values,
                    color_continuous_scale=[COLORS["blue"],COLORS["red"]])
                fig_shap.update_traces(texttemplate="%{x:.3f}", textposition="outside", textfont_size=9)
                fig_shap.update_layout(height=180, margin=dict(l=5,r=25,t=5,b=5),
                    xaxis_title="|SHAP|", coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_shap, use_container_width=True)
                top_d = sv.idxmax()
                box(f"**{top_d}** drives **{sv[top_d]/sv.sum()*100:.0f}%** of this county's classification.", "info")
            else:
                box("No SHAP data for this county.", "warn")
        else:
            box("Run notebook to generate shap_values.csv", "warn")

    if mode != "Single":
        st.markdown("---")
        st.markdown("##### Comparison Table")
        rows = []
        for cn in comp:
            if cn in df.index:
                r = {"County": cn, "UCS": f"{df.loc[cn,'UCS']:.1f}"}
                for d in avail_domains: r[DOMAIN_META[d]["short"]] = f"{df.loc[cn,d]:.3f}"
                rows.append(r)
        st.dataframe(pd.DataFrame(rows).set_index("County"), use_container_width=True, height=100)

# ─────────────────────────────────────────────────────────────
# PAGE: ML & SHAP
# ─────────────────────────────────────────────────────────────

elif page == "ML & SHAP":
    if df is None: st.error("Data not found."); st.stop()

    st.markdown('<p class="pg-title">ML & SHAP — Model Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">XGBoost · Feature importance · Anomaly detection · County SHAP drill-down · Policy implications</p>', unsafe_allow_html=True)

    MODEL_METRICS = {
        "Random Forest":       {"accuracy": 87.0, "f1": 0.80, "roc_auc": 0.7944, "holdout_auc": 0.5769, "note": "Best CV AUC (0.79)"},
        "XGBoost":             {"accuracy": 87.0, "f1": 0.80, "roc_auc": None,   "holdout_auc": 0.4808, "note": "Best overall model selected"},
        "Gradient Boosting":   {"accuracy": 87.0, "f1": 0.80, "roc_auc": 0.6764, "holdout_auc": 0.5000, "note": "CV AUC 0.68"},
        "Logistic Regression": {"accuracy": 80.0, "f1": 0.77, "roc_auc": 0.6333, "holdout_auc": 0.4615, "note": "Linear baseline"},
    }

    ms_col, p1c, p2c, p3c, p4c = st.columns([1.5,1,1,1,1])
    with ms_col: sel_model = st.selectbox("Model", list(MODEL_METRICS.keys()))
    mm = MODEL_METRICS[sel_model]
    with p1c: st.metric("Accuracy",  f"{mm['accuracy']:.0f}%")
    with p2c: st.metric("Wtd F1",    f"{mm['f1']:.2f}")
    cv_str = f"{mm['roc_auc']:.3f}" if mm['roc_auc'] else "—"
    with p3c: st.metric("CV AUC",    cv_str)
    with p4c: st.metric("Holdout AUC", f"{mm['holdout_auc']:.3f}")

    box(f"**{sel_model}**: {mm['note']}. Note: small dataset (N=47, 5 anomalies) limits AUC reliability. Random Forest achieved the strongest cross-validated AUC (0.79). SHAP values interpret the XGBoost model outputs.", "info")
    st.markdown("---")


    # ── KEY FINDINGS FROM NOTEBOOK ─────────────────────────────
    st.markdown("##### Key Findings — KDHS 2020 & 2022 Analysis")
    nf1, nf2, nf3, nf4 = st.columns(4)
    with nf1:
        kpi("k=2", "Optimal Clusters", COLORS["blue"])
        st.caption("Silhouette = 0.459 (k=2 best across all 3 metrics)")
    with nf2:
        kpi("5", "Anomalous Counties", COLORS["orange"])
        st.caption("Turkana · Nairobi · Marsabit · Tana River · Kilifi")
    with nf3:
        kpi("0.794", "Best CV AUC", COLORS["teal"])
        st.caption("Random Forest · XGBoost selected as primary model")
    with nf4:
        kpi("r = 0.83", "Pop. Vulnerability", COLORS["red"])
        st.caption("Strongest domain driver (r²=0.692)")

    st.markdown("""
| Domain | CV Weight | r with UCS | r² |
|---|---|---|---|
| Population Vulnerability Index | 0.3997 | 0.832 | 0.692 |
| Healthcare Access Index | 0.3717 | 0.814 | 0.663 |
| Disease Burden Index | 0.2258 | 0.650 | 0.423 |
| Immunization Coverage Index | 0.0028 | 0.039 | 0.002 |
    """)
    box("Immunization Coverage is operationally independent (r=0.039, CV weight=0.003). Vertical programme success does not automatically translate to broader health system strength.", "warn")

    st.markdown("---")
    st.markdown("##### SHAP Explorer")
    if shap_df is None:
        box("shap_values.csv not found. Run the UCS notebook first.", "warn")
    else:
        shap_data = shap_df.copy()
        has_county = "County" in shap_data.columns
        avail_d = [c for c in shap_data.columns if c in DOMAINS]

        vm_col, vs_col, vi_col = st.columns([1.3,1.4,1.3])
        with vm_col:
            view_mode = st.selectbox("View", [
                "All Counties (Average)",
                "Individual County",
                "Top 5 Underserved",
                "Underserved vs Well-Served",
            ])

        shap_fig = None; insight_txt = ""

        if view_mode == "Individual County":
            county_opts = sorted(shap_data["County"].tolist()) if has_county else sorted(df.index.tolist())
            with vs_col: sel_c = st.selectbox("County", county_opts, label_visibility="collapsed")
            rdf = shap_data[shap_data["County"]==sel_c] if has_county else shap_data[shap_data.index==sel_c]
            if len(rdf) > 0:
                sv = pd.Series({DOMAIN_META[c]["short"]: abs(rdf.iloc[0][c]) for c in avail_d}).sort_values()
                ucs_v2 = df.loc[sel_c,"UCS"] if sel_c in df.index else "?"
                with vi_col: st.metric("UCS", f"{ucs_v2:.1f}" if isinstance(ucs_v2,float) else ucs_v2)
                shap_fig = px.bar(sv, orientation="h", color=sv.values,
                    color_continuous_scale=[COLORS["blue"],COLORS["red"]],
                    title=f"SHAP — {sel_c}")
                insight_txt = f"**{sv.idxmax()}** drives **{sv.max()/sv.sum()*100:.0f}%** of {sel_c}'s UCS prediction."

        elif view_mode == "Top 5 Underserved":
            top5 = df["UCS"].sort_values(ascending=False).head(5).index
            rdf = shap_data[shap_data["County"].isin(top5)] if has_county else shap_data[shap_data.index.isin(top5)]
            sv = pd.Series({DOMAIN_META[c]["short"]: abs(rdf[c].mean()) for c in avail_d}).sort_values()
            with vs_col: st.caption(", ".join(top5[:3]) + "…")
            with vi_col: st.metric("Avg UCS", f"{df.loc[top5,'UCS'].mean():.1f}")
            shap_fig = px.bar(sv, orientation="h", color=sv.values,
                color_continuous_scale=[COLORS["blue"],COLORS["red"]],
                title="SHAP — Top 5 Underserved")
            insight_txt = f"**{sv.idxmax()}** is the shared driver across the 5 most underserved counties."

        elif view_mode == "Underserved vs Well-Served":
            high = df[df["UCS"] >= 70].index; low = df[df["UCS"] < 40].index
            sh = shap_data[shap_data["County"].isin(high)][avail_d].mean() if has_county else shap_data[shap_data.index.isin(high)][avail_d].mean()
            sl = shap_data[shap_data["County"].isin(low)][avail_d].mean()  if has_county else shap_data[shap_data.index.isin(low)][avail_d].mean()
            sh.index = [DOMAIN_META.get(c,{}).get("short",c) for c in sh.index]
            sl.index = [DOMAIN_META.get(c,{}).get("short",c) for c in sl.index]
            cmp = pd.DataFrame({"Underserved (≥70)": sh, "Well Served (<40)": sl})
            shap_fig = go.Figure()
            shap_fig.add_trace(go.Bar(name="Underserved", x=cmp.index, y=cmp["Underserved (≥70)"], marker_color=COLORS["red"]))
            shap_fig.add_trace(go.Bar(name="Well Served",  x=cmp.index, y=cmp["Well Served (<40)"],  marker_color=COLORS["green"]))
            shap_fig.update_layout(barmode="group", height=230, margin=dict(l=5,r=5,t=30,b=5),
                xaxis_tickfont_size=9, legend=dict(font_size=9),
                title="SHAP: Underserved vs Well-Served",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            insight_txt = f"**{(sh-sl).idxmax()}** shows the largest SHAP gap — strongest differentiator of inequality."

        else:  # All Counties Average
            sv = pd.Series({DOMAIN_META[c]["short"]: abs(shap_data[c].mean()) for c in avail_d}).sort_values()
            with vi_col: st.metric("Counties", len(shap_data))
            shap_fig = px.bar(sv, orientation="h", color=sv.values,
                color_continuous_scale=[COLORS["blue"],COLORS["red"]],
                title="SHAP — All 47 Counties Average")
            insight_txt = f"**{sv.idxmax()}** is the strongest system-wide predictor (avg SHAP {sv.max():.3f})."

        if shap_fig is not None:
            if not isinstance(shap_fig, go.Figure):
                shap_fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", textfont_size=9)
                shap_fig.update_layout(height=230, margin=dict(l=5,r=25,t=30,b=5),
                    coloraxis_showscale=False, xaxis_title="|SHAP|",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(shap_fig, use_container_width=True)
        if insight_txt: box(insight_txt, "info")

    st.markdown("---")
    ac, pc = st.columns([1.2,1])

    with ac:
        st.markdown("##### Anomaly Detection")
        if "Anomaly" in df.columns:
            adf = df.copy()
            adf["Status"] = adf["Anomaly"].map(lambda x: "Anomaly" if x=="Anomaly" else "Normal")
            avail_x = DOMAINS[0] if DOMAINS[0] in adf.columns else adf.select_dtypes("number").columns[0]
            fig_a = px.scatter(adf, x="UCS", y=avail_x, color="Status",
                color_discrete_map={"Anomaly": COLORS["red"], "Normal": COLORS["blue"]},
                hover_name=adf.index)
            fig_a.update_layout(height=210, margin=dict(l=5,r=5,t=20,b=5),
                legend=dict(font_size=9),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_a, use_container_width=True)
            n_a = (df["Anomaly"]=="Anomaly").sum()
            if n_a > 0: box(f"{n_a} counties flagged — unusual domain profiles vs cluster peers.", "warn")
            else: box("No significant anomalies detected.", "ok")

    with pc:
        st.markdown("##### County Clusters (K-Means k=2)")
        # Show the 2-cluster solution from the notebook — not 3 UCS bands
        fig_p = go.Figure(go.Pie(
            labels=["Cluster 1 — Structurally Underserved (n=8)",
                    "Cluster 2 — Moderately Served (n=39)"],
            values=[8, 39], hole=0.52,
            marker_colors=[COLORS["orange_dark"], COLORS["blue"]],
            textinfo="percent", textfont_size=9
        ))
        fig_p.update_layout(
            height=200, margin=dict(l=5,r=5,t=10,b=5),
            showlegend=True,
            legend=dict(font_size=8, orientation="v", x=0.55, y=0.5),
            annotations=[dict(text="k=2<br>Sil=0.459", x=0.18, y=0.5,
                             font_size=8, showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)
        box("Cluster 1 (8 counties): Garissa, Mandera, Marsabit, Samburu, Tana River, Turkana, Wajir, West Pokot — all ASAL. Cluster 2 (39 counties): all remaining. Silhouette = 0.459, k=2 confirmed optimal.", "info")

# ─────────────────────────────────────────────────────────────
# PAGE: KDHS PREDICTOR  ← NEW
# ─────────────────────────────────────────────────────────────

elif page == "KDHS Predictor":

    st.markdown('<p class="pg-title">KDHS Raw Data Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Upload a raw KDHS CSV → auto-map columns to domains → compute domain scores → predict UCS for each county</p>', unsafe_allow_html=True)

    # ── Step indicator ──────────────────────────────────────
    s1c, s2c, s3c, s4c = st.columns(4)
    with s1c: st.markdown('<span class="step-badge">1</span> **Upload CSV**', unsafe_allow_html=True)
    with s2c: st.markdown('<span class="step-badge">2</span> **Review column mapping**', unsafe_allow_html=True)
    with s3c: st.markdown('<span class="step-badge">3</span> **Adjust & compute scores**', unsafe_allow_html=True)
    with s4c: st.markdown('<span class="step-badge">4</span> **View UCS predictions**', unsafe_allow_html=True)

    st.markdown("---")

    # ── STEP 1: Upload ───────────────────────────────────────
    st.markdown("#### Step 1 — Upload Raw KDHS Data")

    col_up, col_fmt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a county-level KDHS CSV (rows = counties, columns = indicators)",
            type=["csv", "xlsx"],
            help="The file should have one row per county and one column per indicator. "
                 "The first column should be county names (or set as the index)."
        )

    with col_fmt:
        box("""
        **Expected format:**
        - Rows = counties (47 ideally)
        - Columns = health indicators
        - County names in first column or as row index
        - Numeric values (percentages or rates)
        """, "info")

    # Demo data button
    if not uploaded:
        st.markdown("##### Or try with demo data")
        if st.button("Generate synthetic KDHS demo data (47 counties)"):
            np.random.seed(42)
            counties = list(KENYA_COORDS.keys())
            demo_data = {"County": counties}
            # Healthcare Access indicators
            demo_data["antenatal_4plus_visits_pct"]   = np.random.uniform(20, 95, 47)
            demo_data["skilled_birth_attendance_pct"] = np.random.uniform(15, 98, 47)
            demo_data["health_insurance_coverage_pct"]= np.random.uniform(5,  75, 47)
            demo_data["distance_barrier_pct"]         = np.random.uniform(5,  80, 47)
            demo_data["postnatal_care_pct"]            = np.random.uniform(10, 90, 47)
            # Vulnerability
            demo_data["poorest_wealth_quintile_pct"]  = np.random.uniform(5,  70, 47)
            demo_data["no_education_pct"]             = np.random.uniform(2,  60, 47)
            demo_data["unimproved_water_pct"]         = np.random.uniform(5,  85, 47)
            demo_data["open_defecation_pct"]          = np.random.uniform(1,  80, 47)
            # Immunization
            demo_data["fully_immunized_pct"]           = np.random.uniform(30, 98, 47)
            demo_data["bcg_coverage_pct"]              = np.random.uniform(50, 99, 47)
            demo_data["measles_vaccine_pct"]           = np.random.uniform(25, 99, 47)
            demo_data["vitamin_a_supplement_pct"]      = np.random.uniform(15, 92, 47)
            # Disease Burden
            demo_data["stunting_pct"]                  = np.random.uniform(5,  55, 47)
            demo_data["underweight_pct"]               = np.random.uniform(3,  45, 47)
            demo_data["malaria_prevalence_pct"]        = np.random.uniform(0,  60, 47)
            demo_data["anaemia_children_pct"]          = np.random.uniform(10, 80, 47)

            demo_df = pd.DataFrame(demo_data)
            csv_bytes = demo_df.to_csv(index=False).encode()
            st.download_button("Download demo CSV", csv_bytes,
                file_name="kdhs_demo.csv", mime="text/csv")
            st.session_state["demo_df"] = demo_df
            box("Demo data generated! Download and re-upload, or we'll use it directly below.", "ok")

    # ── Process uploaded file ────────────────────────────────
    raw_df = None

    if uploaded:
        try:
            if uploaded.name.endswith(".xlsx"):
                raw_df = pd.read_excel(uploaded)
            else:
                raw_df = pd.read_csv(uploaded)
            box(f"Loaded **{len(raw_df)} rows × {len(raw_df.columns)} columns**", "ok")
        except Exception as e:
            box(f"Error reading file: {e}", "err")

    elif "demo_df" in st.session_state:
        raw_df = st.session_state["demo_df"]
        box("Using generated demo data", "info")

    if raw_df is not None:
        st.markdown("---")

        # Detect county column
        county_col = None
        for c in raw_df.columns:
            if "county" in c.lower() or "region" in c.lower() or "area" in c.lower():
                county_col = c; break
        if county_col is None and raw_df.columns[0].dtype == object:
            county_col = raw_df.columns[0]

        if county_col:
            raw_df = raw_df.set_index(county_col)
            raw_df.index.name = "County"

        st.markdown(f"**Preview** (first 5 rows, {len(raw_df.columns)} columns):")
        st.dataframe(raw_df.head(), use_container_width=True, height=130)

        st.markdown("---")

        # ── STEP 2: Column Mapping ───────────────────────────
        st.markdown("#### Step 2 — Column → Domain Mapping")
        box("Columns are auto-mapped to domains by keyword matching. Adjust any misclassified columns below.", "info")

        auto_mapping = auto_map_columns(raw_df)
        unmapped = [c for c in raw_df.columns if not any(c in v for v in auto_mapping.values())]

        # Show mapping summary
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col_obj, domain in zip([mc1,mc2,mc3,mc4], DOMAINS):
            with col_obj:
                n = len(auto_mapping[domain])
                color = DOMAIN_META[domain]["color"]
                st.markdown(f"""<div class="kpi" style="--kc:{color}">
                    <div class="v">{n}</div>
                    <div class="l">{DOMAIN_META[domain]['icon']} {DOMAIN_META[domain]['short']}</div>
                </div>""", unsafe_allow_html=True)

        if unmapped:
            box(f"**{len(unmapped)} columns not mapped:** {', '.join(unmapped[:8])}{'…' if len(unmapped)>8 else ''}. "
                "Assign them below or they will be excluded.", "warn")

        # Manual override UI
        _user_mapping = {d: [] for d in DOMAINS}
        _mapping_edited = False
        with st.expander(f" Edit column assignments ({len(raw_df.columns)} columns)", expanded=False):
            _mapping_edited = True
            domain_options = ["Exclude"] + DOMAINS
            e1, e2, e3 = st.columns(3)
            cols_list = list(raw_df.columns)
            for i, col in enumerate(cols_list):
                auto_dom = detect_column_domain(col)
                default_idx = domain_options.index(auto_dom) if auto_dom else 0
                target_col = [e1, e2, e3][i % 3]
                with target_col:
                    chosen = st.selectbox(
                        f"`{col}`", domain_options, index=default_idx,
                        key=f"col_map_{col}", label_visibility="visible"
                    )
                    if chosen != "Exclude":
                        _user_mapping[chosen].append(col)

        # Use user edits if expander was opened, else auto mapping
        final_mapping = _user_mapping if any(_user_mapping[d] for d in DOMAINS) else auto_mapping

        st.markdown("---")

        # ── STEP 3: Compute Domain Scores ────────────────────
        st.markdown("#### Step 3 — Compute Domain Scores")

        min_cols_ok = all(len(final_mapping[d]) > 0 for d in DOMAINS)
        if not min_cols_ok:
            missing = [DOMAIN_META[d]["short"] for d in DOMAINS if not final_mapping[d]]
            box(f"Missing columns for: **{', '.join(missing)}**. Assign at least one column per domain to proceed.", "warn")
        else:
            if st.button("Compute Domain Scores & Predict UCS", type="primary"):
                with st.spinner("Computing domain scores using PCA weighting…"):
                    try:
                        domain_scores = pd.DataFrame(index=raw_df.index)
                        pca_report = {}

                        for d in DOMAINS:
                            cols = final_mapping[d]
                            score = compute_domain_score(raw_df, cols, d)
                            domain_scores[d] = score
                            pca_report[d] = {"n_cols": len(cols), "cols": cols}

                        # Compute UCS
                        ucs_pred = compute_ucs(domain_scores)
                        domain_scores["UCS"] = ucs_pred

                        # Anomaly detection
                        if SKLEARN_OK and len(domain_scores) >= 5:
                            iso = IsolationForest(contamination=0.10, random_state=42)
                            anom_labels = iso.fit_predict(domain_scores[DOMAINS].fillna(0))
                            domain_scores["Anomaly"] = ["Anomaly" if x==-1 else "Normal" for x in anom_labels]
                        
                        # Rank counties
                        domain_scores["Rank"] = domain_scores["UCS"].rank(ascending=False).astype(int)

                        # Add cluster label if model exists
                        if SKLEARN_OK:
                            try:
                                from sklearn.cluster import KMeans
                                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                                km.fit(domain_scores[DOMAINS].fillna(0))
                                domain_scores["Cluster"] = ["Structurally Underserved" if l==km.labels_[domain_scores["UCS"].idxmax()] else "Moderately Served" for l in km.labels_]
                            except Exception:
                                pass

                        st.session_state["prediction_results"] = domain_scores
                        box(f"Computed UCS for **{len(domain_scores)} counties** successfully.", "ok")

                    except Exception as e:
                        box(f"Error during computation: {e}", "err")

        # ── STEP 4: Show Results ─────────────────────────────
        if "prediction_results" in st.session_state:
            results = st.session_state["prediction_results"]
            st.markdown("---")
            st.markdown("#### Step 4 — Prediction Results")

            # KPI summary
            k1c, k2c, k3c, k4c, k5c = st.columns(5)
            with k1c: kpi(len(results), "Counties Scored", COLORS["blue"])
            with k2c: kpi(f"{results['UCS'].mean():.1f}", "Mean UCS", COLORS["orange"])
            worst_p = results["UCS"].idxmax()
            with k3c: kpi(f"{results['UCS'].max():.1f}", f"Worst · {worst_p[:8]}", COLORS["red"])
            best_p = results["UCS"].idxmin()
            with k4c: kpi(f"{results['UCS'].min():.1f}", f"Best · {best_p[:8]}", COLORS["green"])
            n_anom = len(results[results.get("Anomaly","Normal")=="Anomaly"]) if "Anomaly" in results.columns else 0
            with k5c: kpi(n_anom, "Anomalies", COLORS["purple"])

            st.markdown("---")
            res_left, res_right = st.columns([1.3, 1])

            with res_left:
                # Top/bottom bar charts
                st.markdown("##### Most Underserved")
                top_r = results["UCS"].sort_values(ascending=False).head(10).reset_index()
                top_r.columns = ["County","UCS"]
                fig_tr = px.bar(top_r, x="UCS", y="County", orientation="h",
                    color="UCS", color_continuous_scale=["#E8A87C","#C0392B"], range_color=[0,100])
                fig_tr.update_traces(texttemplate="%{x:.1f}", textposition="outside", textfont_size=9)
                fig_tr.update_layout(yaxis_autorange="reversed", height=270,
                    margin=dict(l=5,r=30,t=5,b=5), xaxis_title="", yaxis_title="",
                    coloraxis_showscale=False, yaxis_tickfont_size=9,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tr, use_container_width=True)

                # Domain breakdown for worst county
                st.markdown(f"##### ️ Profile: {worst_p}")
                worst_row = results.loc[worst_p]
                vals_w = {DOMAIN_META[d]["short"]: float(worst_row.get(d, 0)) for d in DOMAINS}
                st.plotly_chart(radar_chart(vals_w, worst_p, 220), use_container_width=True)

            with res_right:
                # Full ranked table
                st.markdown("#####  Full Rankings")
                display_cols = ["UCS","Rank"] + \
                    ([c for c in ["Cluster","Anomaly"] if c in results.columns])
                show_df = results[display_cols].sort_values("UCS", ascending=False)
                show_df["UCS"] = show_df["UCS"].round(1)
                show_df["Status"] = show_df["UCS"].apply(ucs_label)
                st.dataframe(show_df, use_container_width=True, height=380)

                # Domain scores heatmap
                st.markdown("##### ️ Domain Score Heatmap")
                heat_df = results[DOMAINS].round(1)
                heat_df.columns = [DOMAIN_META[d]["short"] for d in DOMAINS]
                fig_h = px.imshow(heat_df.sort_values(by=list(heat_df.columns), ascending=False).head(20),
                    color_continuous_scale=[[0,"#2B7A3F"],[0.5,"#F5F6FA"],[1,"#C0392B"]], aspect="auto",
                    title="Top 20 counties — domain scores")
                fig_h.update_layout(height=300, margin=dict(l=5,r=5,t=30,b=5),
                    xaxis_tickfont_size=9, yaxis_tickfont_size=8)
                st.plotly_chart(fig_h, use_container_width=True)

            st.markdown("---")

            # Download results
            d1, d2 = st.columns(2)
            with d1:
                csv_out = results.round(3).to_csv().encode()
                st.download_button(
                    "Download UCS Predictions (CSV)",
                    csv_out, file_name="ucs_predictions.csv", mime="text/csv",
                    type="primary"
                )
            with d2:
                box("Download the CSV and place it as **county_ucs_final.csv** in the app directory to use it in all other pages.", "info")

            # Comparison with reference data
            if df is not None:
                st.markdown("---")
                st.markdown("#####  Comparison with Reference UCS (KDHS 2020/2022)")
                common = list(set(results.index) & set(df.index))
                if len(common) >= 3:
                    comp_df = pd.DataFrame({
                        "Predicted UCS": results.loc[common, "UCS"],
                        "Reference UCS": df.loc[common, "UCS"]
                    })
                    corr = comp_df.corr().iloc[0,1]
                    fig_cmp = px.scatter(comp_df, x="Reference UCS", y="Predicted UCS",
                        hover_name=comp_df.index,
                        title=f"Predicted vs Reference UCS — r = {corr:.2f}")
                    fig_cmp.add_trace(go.Scatter(x=[0,100], y=[0,100],
                        mode="lines", line=dict(dash="dash", color="gray"),
                        name="Perfect agreement", showlegend=True))
                    fig_cmp.update_layout(height=320, margin=dict(l=20,r=20,t=35,b=20))
                    st.plotly_chart(fig_cmp, use_container_width=True)
                    if corr > 0.8:
                        box(f"Strong agreement with reference data (r = {corr:.2f}). The column mapping is working well.", "ok")
                    elif corr > 0.5:
                        box(f"Moderate agreement (r = {corr:.2f}). Review column mappings and check data quality.", "warn")
                    else:
                        box(f"Weak agreement (r = {corr:.2f}). Check that columns are correctly mapped to domains and data is on the right scale.", "err")
                else:
                    box("Not enough matching counties to compare with reference data.", "warn")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Kenya Health Equity Monitor Dashboard v2 · UCS Methodology · KDHS 2020 & 2022 · 47 Counties · "
    "Cynthia Ngugi (138725) · MSc Data Science & Analytics, Strathmore University"
)
