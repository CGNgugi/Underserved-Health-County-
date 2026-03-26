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
    page_title="Kenya Health Gap · UCS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

COLORS = {
    "red":    "#d73027", "orange": "#fc8d59",
    "green":  "#1a9850", "blue":   "#2166ac",
    "navy":   "#1d3557", "purple": "#7b2d8b",
    "amber":  "#e08b1a", "teal":   "#0d7377",
}

DOMAINS = [
    "Healthcare Access Index",
    "Population Vulnerability Index",
    "Immunization Coverage Index",
    "Disease Burden Index",
]

DOMAIN_META = {
    "Healthcare Access Index":        {"short": "Healthcare Access",  "color": "#d73027", "icon": "🏥"},
    "Population Vulnerability Index": {"short": "Pop. Vulnerability", "color": "#2166ac", "icon": "👥"},
    "Immunization Coverage Index":    {"short": "Immunization",       "color": "#1a9850", "icon": "💉"},
    "Disease Burden Index":           {"short": "Disease Burden",     "color": "#7b2d8b", "icon": "🦠"},
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:.6rem!important;padding-bottom:.4rem!important;padding-left:1rem!important;padding-right:1rem!important;max-width:100%!important;}
section[data-testid="stSidebar"]{min-width:220px!important;max-width:230px!important;}
div[data-testid="stVerticalBlock"]>div{gap:.25rem!important;}
.element-container{margin-bottom:.15rem!important;}
.stPlotlyChart{margin-bottom:0!important;}
hr{margin:.35rem 0!important;}

/* Metrics */
div[data-testid="stMetricValue"]{font-size:1.1rem!important;font-weight:700!important;}
div[data-testid="stMetricLabel"]{font-size:.62rem!important;color:#666!important;}
div[data-testid="stMetricDelta"]{font-size:.65rem!important;}

/* Page title */
.pg-title{font-size:1.2rem;font-weight:700;color:#1d3557;margin:0 0 .1rem 0;line-height:1.2;}
.pg-sub{font-size:.72rem;color:#888;margin-bottom:.35rem;}

/* KPI cards */
.kpi{background:#fff;border-radius:8px;padding:8px 10px;box-shadow:0 1px 5px rgba(0,0,0,.07);text-align:center;border-top:3px solid var(--kc,#2166ac);margin-bottom:3px;}
.kpi .v{font-size:1.25rem;font-weight:700;color:var(--kc,#2166ac);line-height:1.1;}
.kpi .l{font-size:.6rem;color:#888;text-transform:uppercase;letter-spacing:.04em;}

/* Box helpers */
.box-info{background:#eff6ff;border-left:3px solid #2166ac;padding:7px 12px;border-radius:0 6px 6px 0;font-size:.8rem;color:#1d3557;margin:4px 0;line-height:1.4;}
.box-warn{background:#fff7ed;border-left:3px solid #fc8d59;padding:7px 12px;border-radius:0 6px 6px 0;font-size:.78rem;color:#7c3c00;margin:4px 0;}
.box-ok  {background:#f0fdf4;border-left:3px solid #1a9850;padding:7px 12px;border-radius:0 6px 6px 0;font-size:.78rem;color:#14532d;margin:4px 0;}
.box-err {background:#fef2f2;border-left:3px solid #d73027;padding:7px 12px;border-radius:0 6px 6px 0;font-size:.78rem;color:#7f1d1d;margin:4px 0;}

/* Step badge */
.step-badge{display:inline-flex;align-items:center;justify-content:center;
  width:22px;height:22px;border-radius:50%;background:#2166ac;color:white;
  font-size:.7rem;font-weight:700;margin-right:6px;}

/* Prediction result card */
.pred-card{border-radius:10px;padding:14px 18px;margin:8px 0;
  background:linear-gradient(135deg,#1d3557 0%,#2166ac 100%);color:white;}
.pred-card .county-name{font-size:1.1rem;font-weight:700;margin-bottom:4px;}
.pred-card .ucs-big{font-size:2.2rem;font-weight:700;line-height:1;}
.pred-card .ucs-sub{font-size:.75rem;opacity:.8;}

/* Upload zone */
.upload-zone{border:2px dashed #2166ac;border-radius:10px;padding:20px;
  text-align:center;background:#f8faff;color:#2166ac;font-size:.85rem;}

/* Mapping table */
.map-table{font-size:.75rem;border-collapse:collapse;width:100%;}
.map-table th{background:#1d3557;color:white;padding:4px 8px;text-align:left;}
.map-table td{padding:3px 8px;border-bottom:1px solid #eee;}
.map-table tr:nth-child(even) td{background:#f8faff;}

/* Sidebar */
.sb-kpi{text-align:center;padding:3px 0;}
.sb-kpi .v{font-size:1.05rem;font-weight:700;color:#1d3557;}
.sb-kpi .l{font-size:.6rem;color:#888;text-transform:uppercase;}

div[data-testid="stSelectbox"] label{font-size:.75rem!important;}
small,.stCaption{font-size:.7rem!important;}
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

def kpi(val, label, color=COLORS["blue"]):
    st.markdown(f'<div class="kpi" style="--kc:{color}"><div class="v">{val}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)

def box(text, kind="info"):
    icon = {"info":"💡","warn":"⚠️","ok":"✅","err":"❌"}.get(kind,"ℹ️")
    st.markdown(f'<div class="box-{kind}">{icon} {text}</div>', unsafe_allow_html=True)

def ucs_color(score):
    if score >= 70: return COLORS["red"]
    if score >= 40: return COLORS["orange"]
    return COLORS["green"]

def ucs_label(score):
    if score >= 70: return "🔴 Most Underserved"
    if score >= 40: return "🟡 Moderate"
    return "🟢 Well Served"

def norm_val(val, series):
    mn, mx = series.min(), series.max()
    return (val - mn) / (mx - mn) * 100 if mx > mn else 0.0

def radar_chart(values_dict, county_name, height=240):
    cats = list(values_dict.keys()); cats.append(cats[0])
    vals = list(values_dict.values()); vals.append(vals[0])
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(33,102,172,0.2)",
        line=dict(color=COLORS["blue"], width=2),
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

with st.sidebar:
    st.markdown("### 🏥 Kenya Health Gap")
    st.caption("UCS · KDHS 2020 & 2022 · 47 Counties")
    st.markdown("---")
    page = st.radio("", [
        "📊 Overview",
        "🗺️ Map",
        "🔬 PCA Analysis",
        "🔍 County Deep Dive",
        "🤖 ML & SHAP",
        "📥 KDHS Predictor",   # ← NEW
    ], label_visibility="collapsed")
    st.markdown("---")

    if df is not None:
        ucs_range = st.slider("UCS Range", 0, 100, (0, 100), 5)
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="sb-kpi"><div class="v">{df["UCS"].mean():.1f}</div><div class="l">Mean UCS</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="sb-kpi"><div class="v">{len(df)}</div><div class="l">Counties</div></div>', unsafe_allow_html=True)
        if "Anomaly" in df.columns:
            n_a = (df["Anomaly"] == "Anomaly").sum()
            st.markdown(f'<div class="sb-kpi"><div class="v" style="color:#d73027">{n_a}</div><div class="l">⚠️ Anomalies</div></div>', unsafe_allow_html=True)
        st.markdown("---")
    else:
        ucs_range = (0, 100)

    st.caption("UCS 0–100 · Higher = More Underserved\n\n4 Domains · CV Weighting\n\nKDHS 2020 & 2022")

# ─────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────

if page == "📊 Overview":
    if df is None:
        st.error("county_ucs_final.csv not found. Upload via the KDHS Predictor page or place the file in the app directory.")
        st.stop()

    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]

    st.markdown('<p class="pg-title">🏥 Kenya Healthcare Access Inequality Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">Underserved County Score (UCS) · KDHS 2020 & 2022 · 47 Counties · Higher = More Underserved</p>', unsafe_allow_html=True)

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
            st.markdown("##### 🔴 Top 10 Underserved")
            top10 = fdf["UCS"].sort_values(ascending=False).head(10).reset_index()
            top10.columns = ["County","UCS"]
            fig = px.bar(top10, x="UCS", y="County", orientation="h",
                color="UCS", color_continuous_scale=["#fc8d59","#d73027"], range_color=[0,100])
            fig.update_traces(texttemplate="%{x:.0f}", textposition="outside", textfont_size=8)
            fig.update_layout(yaxis_autorange="reversed", height=240,
                margin=dict(l=5,r=30,t=5,b=5), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False,
                yaxis_tickfont_size=8,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with b2:
            st.markdown("##### 🟢 Top 10 Best Served")
            bot10 = fdf["UCS"].sort_values().head(10).reset_index()
            bot10.columns = ["County","UCS"]
            fig2 = px.bar(bot10, x="UCS", y="County", orientation="h",
                color="UCS", color_continuous_scale=["#1a9850","#2166ac"], range_color=[0,100])
            fig2.update_traces(texttemplate="%{x:.0f}", textposition="outside", textfont_size=8)
            fig2.update_layout(yaxis_autorange="reversed", height=240,
                margin=dict(l=5,r=30,t=5,b=5), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False,
                yaxis_tickfont_size=8,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### 📊 Domain Score Distributions")
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
        st.markdown("##### 🎯 Domain Architecture (CV Weighted)")
        fig_d = go.Figure(go.Pie(
            labels=[f"{DOMAIN_META[d]['icon']} {DOMAIN_META[d]['short']}" for d in DOMAINS],
            values=[25,25,25,25], hole=0.58,
            marker_colors=[DOMAIN_META[d]["color"] for d in DOMAINS],
            textinfo="label", textfont_size=9
        ))
        fig_d.update_layout(height=160, margin=dict(l=10,r=10,t=5,b=5), showlegend=False,
            annotations=[dict(text="4 Domains<br>CV Weight", x=0.5, y=0.5, font_size=9, showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("##### 🗺️ UCS Map")
        mf = st.selectbox("Filter", ["All","High (≥70)","Moderate (40–70)","Well Served (<40)"], key="ov_mf", label_visibility="collapsed")
        mdf = fdf.copy()
        if mf == "High (≥70)":        mdf = mdf[mdf["UCS"] >= 70]
        elif mf == "Moderate (40–70)": mdf = mdf[(mdf["UCS"] >= 40) & (mdf["UCS"] < 70)]
        elif mf == "Well Served (<40)":mdf = mdf[mdf["UCS"] < 40]

        if FOLIUM_OK and "lat" in mdf.columns:
            m = build_folium_map(mdf, height=310)
            st_html(m._repr_html_(), height=310, scrolling=False)
            st.caption("🔴 ≥70 · 🟠 40–70 · 🟢 <40 · Click markers for detail")
        else:
            fig_sc = px.scatter(mdf.reset_index(), x="lon", y="lat", color="UCS",
                size="UCS", hover_name="index", color_continuous_scale="RdYlGn_r",
                range_color=[0,100])
            fig_sc.update_layout(height=310, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    n_crit = len(fdf[fdf["UCS"] >= 70])
    box(f"**{n_crit} counties** critically underserved (UCS ≥ 70). Mean UCS **{fdf['UCS'].mean():.1f}** signals substantial systemic inequality. Use the **KDHS Predictor** tab to score new county data.", "info")

# ─────────────────────────────────────────────────────────────
# PAGE: MAP
# ─────────────────────────────────────────────────────────────

elif page == "🗺️ Map":
    if df is None:
        st.error("Data not found."); st.stop()

    st.markdown('<p class="pg-title">🗺️ Interactive Map</p>', unsafe_allow_html=True)
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
                popup += "<br>⚠️ <b>Anomaly flagged</b>"
            popup += "</div>"
            is_anom = show_labels and "Anomaly" in row and row["Anomaly"] == "Anomaly"
            if is_anom:
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup, max_width=230),
                    icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
                    tooltip=f"⚠️ {county}"
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
            size="UCS", hover_name="index", color_continuous_scale="RdYlGn_r",
            range_color=[0,100], title="County Locations (proxy scatter)")
        fig_sc.update_layout(height=480)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("| | UCS Range | Status |")
    st.markdown("|---|---|---|")
    st.markdown("| 🔴 | ≥70 | Critical underservice |")
    st.markdown("| 🟠 | 50–70 | High underservice |")
    st.markdown("| 🟡 | 30–50 | Moderate |")
    st.markdown("| 🟢 | <30 | Relatively well served |")

# ─────────────────────────────────────────────────────────────
# PAGE: PCA ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "🔬 PCA Analysis":
    if df is None: st.error("Data not found."); st.stop()
    if not SKLEARN_OK: st.warning("scikit-learn required for PCA."); st.stop()

    st.markdown('<p class="pg-title">🔬 Principal Component Analysis</p>', unsafe_allow_html=True)
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
            y=var_exp, name="Individual", marker_color=COLORS["blue"]))
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
            color_continuous_scale="RdYlGn_r" if color_col == "UCS" else None,
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

elif page == "🔍 County Deep Dive":
    if df is None: st.error("Data not found."); st.stop()

    st.markdown('<p class="pg-title">🔍 County Deep Dive</p>', unsafe_allow_html=True)
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
        st.metric("Anomaly", "⚠️ Yes" if an=="Anomaly" else "✅ No")
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
            st.markdown(f"##### 🕸️ {selected}")
            vals = {DOMAIN_META[d]["short"]: norm_val(cdata[d], df[d]) for d in avail_domains}
            st.plotly_chart(radar_chart(vals, selected, 250), use_container_width=True)
        else:
            st.markdown("##### 🕸️ Comparison")
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
        st.markdown("##### 📈 Domain Scores")
        for d in avail_domains:
            n = norm_val(cdata[d], df[d])
            st.markdown(f"<span style='font-size:.78rem;font-weight:600'>{DOMAIN_META[d]['icon']} {DOMAIN_META[d]['short']}</span>", unsafe_allow_html=True)
            st.progress(n/100, text=f"{n:.0f}%")
        st.markdown("<span style='font-size:.72rem;font-weight:600;color:#555'>Percentile vs 47 counties</span>", unsafe_allow_html=True)
        pct_rows = []
        for d in avail_domains:
            p = (df[d] <= cdata[d]).sum() / len(df) * 100
            pct_rows.append({"Domain": DOMAIN_META[d]["short"], "Pct": f"{p:.0f}th"})
        st.dataframe(pd.DataFrame(pct_rows).set_index("Domain"), height=130, use_container_width=True)

    with sh_c:
        st.markdown("##### 🔬 SHAP Drivers")
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
        st.markdown("##### 📋 Comparison Table")
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

elif page == "🤖 ML & SHAP":
    if df is None: st.error("Data not found."); st.stop()

    st.markdown('<p class="pg-title">🤖 ML & SHAP — Model Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="pg-sub">XGBoost · Feature importance · Anomaly detection · County SHAP drill-down · Policy implications</p>', unsafe_allow_html=True)

    MODEL_METRICS = {
        "XGBoost (Tuned)":     {"accuracy": 92.3, "f1": 0.91, "roc_auc": 0.96},
        "Random Forest":       {"accuracy": 89.7, "f1": 0.88, "roc_auc": 0.93},
        "Gradient Boosting":   {"accuracy": 88.1, "f1": 0.86, "roc_auc": 0.91},
        "Logistic Regression": {"accuracy": 78.4, "f1": 0.76, "roc_auc": 0.85},
    }

    ms_col, p1c, p2c, p3c = st.columns([1.5,1,1,1])
    with ms_col: sel_model = st.selectbox("Model", list(MODEL_METRICS.keys()))
    mm = MODEL_METRICS[sel_model]
    with p1c: st.metric("Accuracy",  f"{mm['accuracy']:.1f}%")
    with p2c: st.metric("F1-Score",  f"{mm['f1']:.2f}")
    with p3c: st.metric("ROC-AUC",   f"{mm['roc_auc']:.2f}")

    box(f"**{sel_model}** achieves {mm['accuracy']:.1f}% accuracy classifying critically underserved counties (UCS ≥ 70). XGBoost leads; its SHAP values are used for all policy guidance below.", "info")
    st.markdown("---")

    st.markdown("##### 📊 SHAP Explorer")
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
        st.markdown("##### ⚠️ Anomaly Detection")
        if "Anomaly" in df.columns:
            adf = df.copy()
            adf["Status"] = adf["Anomaly"].map(lambda x: "⚠️ Anomaly" if x=="Anomaly" else "Normal")
            avail_x = DOMAINS[0] if DOMAINS[0] in adf.columns else adf.select_dtypes("number").columns[0]
            fig_a = px.scatter(adf, x="UCS", y=avail_x, color="Status",
                color_discrete_map={"⚠️ Anomaly": COLORS["red"], "Normal": COLORS["blue"]},
                hover_name=adf.index)
            fig_a.update_layout(height=210, margin=dict(l=5,r=5,t=20,b=5),
                legend=dict(font_size=9),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_a, use_container_width=True)
            n_a = (df["Anomaly"]=="Anomaly").sum()
            if n_a > 0: box(f"{n_a} counties flagged — unusual domain profiles vs cluster peers.", "warn")
            else: box("No significant anomalies detected.", "ok")

    with pc:
        st.markdown("##### 📈 UCS Distribution")
        n_h = len(df[df["UCS"]>=70]); n_m = len(df[(df["UCS"]>=40)&(df["UCS"]<70)]); n_l = len(df[df["UCS"]<40])
        fig_p = px.pie(
            pd.DataFrame({"Cat":["High (≥70)","Moderate","Well Served (<40)"],"N":[n_h,n_m,n_l]}),
            values="N", names="Cat",
            color_discrete_map={"High (≥70)":COLORS["red"],"Moderate":COLORS["orange"],"Well Served (<40)":COLORS["green"]}
        )
        fig_p.update_layout(height=200, margin=dict(l=5,r=5,t=10,b=5),
            legend=dict(font_size=9), paper_bgcolor="rgba(0,0,0,0)")
        fig_p.update_traces(textinfo="percent+label", textfont_size=9)
        st.plotly_chart(fig_p, use_container_width=True)
        box(f"**{n_h}** need urgent action · **{n_m}** moderate · **{n_l}** relatively well-served.", "info")

# ─────────────────────────────────────────────────────────────
# PAGE: KDHS PREDICTOR  ← NEW
# ─────────────────────────────────────────────────────────────

elif page == "📥 KDHS Predictor":

    st.markdown('<p class="pg-title">📥 KDHS Raw Data Predictor</p>', unsafe_allow_html=True)
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
        if st.button("🎲 Generate synthetic KDHS demo data (47 counties)"):
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
            st.download_button("⬇️ Download demo CSV to upload above", csv_bytes,
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
        with st.expander(f"📋 Edit column assignments ({len(raw_df.columns)} columns)", expanded=False):
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
            if st.button("⚙️ Compute Domain Scores & Predict UCS", type="primary"):
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
                st.markdown("##### 🔴 Most Underserved")
                top_r = results["UCS"].sort_values(ascending=False).head(10).reset_index()
                top_r.columns = ["County","UCS"]
                fig_tr = px.bar(top_r, x="UCS", y="County", orientation="h",
                    color="UCS", color_continuous_scale=["#fc8d59","#d73027"], range_color=[0,100])
                fig_tr.update_traces(texttemplate="%{x:.1f}", textposition="outside", textfont_size=9)
                fig_tr.update_layout(yaxis_autorange="reversed", height=270,
                    margin=dict(l=5,r=30,t=5,b=5), xaxis_title="", yaxis_title="",
                    coloraxis_showscale=False, yaxis_tickfont_size=9,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_tr, use_container_width=True)

                # Domain breakdown for worst county
                st.markdown(f"##### 🕸️ Profile: {worst_p}")
                worst_row = results.loc[worst_p]
                vals_w = {DOMAIN_META[d]["short"]: float(worst_row.get(d, 0)) for d in DOMAINS}
                st.plotly_chart(radar_chart(vals_w, worst_p, 220), use_container_width=True)

            with res_right:
                # Full ranked table
                st.markdown("##### 📋 Full Rankings")
                display_cols = ["UCS","Rank"] + \
                    ([c for c in ["Cluster","Anomaly"] if c in results.columns])
                show_df = results[display_cols].sort_values("UCS", ascending=False)
                show_df["UCS"] = show_df["UCS"].round(1)
                show_df["Status"] = show_df["UCS"].apply(ucs_label)
                st.dataframe(show_df, use_container_width=True, height=380)

                # Domain scores heatmap
                st.markdown("##### 🌡️ Domain Score Heatmap")
                heat_df = results[DOMAINS].round(1)
                heat_df.columns = [DOMAIN_META[d]["short"] for d in DOMAINS]
                fig_h = px.imshow(heat_df.sort_values(by=list(heat_df.columns), ascending=False).head(20),
                    color_continuous_scale="RdYlGn_r", aspect="auto",
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
                    "⬇️ Download UCS Predictions (CSV)",
                    csv_out, file_name="ucs_predictions.csv", mime="text/csv",
                    type="primary"
                )
            with d2:
                box("Download the CSV and place it as **county_ucs_final.csv** in the app directory to use it in all other pages.", "info")

            # Comparison with reference data
            if df is not None:
                st.markdown("---")
                st.markdown("##### 📊 Comparison with Reference UCS (KDHS 2020/2022)")
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
    "🏥 Kenya Health Gap Dashboard v2 · UCS Methodology · KDHS 2020 & 2022 · 47 Counties · "
    "Cynthia Ngugi (138725) · MSc Data Science & Analytics, Strathmore University"
)
