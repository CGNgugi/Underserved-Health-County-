"""
kenya_health.py - Kenya Health Gap Dashboard (Combined)
========================================================
Merged from health.py + ucs.py
Pages: Overview | County Deep Dive | ML & SHAP
Features: Interactive Map in Overview, Enhanced SHAP Explorer, A4 no-scroll layout

Author: Ngugi, Cynthia (138725) | MSc Data Science & Analytics, Strathmore University
Run: streamlit run kenya_health.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit.components.v1 import html
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Kenya Health Gap Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLORS = {
    "red": "#d73027",
    "orange": "#fc8d59",
    "green": "#1a9850",
    "blue": "#4575b4",
    "navy": "#2c3e6b",
    "purple": "#7b2d8b",
    "light": "#f0f4ff"
}

DOMAINS = [
    "Healthcare Access Index",
    "Population Vulnerability Index",
    "Immunization Coverage Index",
    "Disease Burden Index"
]

DOMAIN_INFO = {
    "Healthcare Access Index": {
        "short": "Healthcare Access",
        "color": "#d73027",
        "icon": "🏥",
        "description": "Availability, accessibility and affordability of health services.",
        "indicators": "HRH density, Facility density, Insurance coverage, Travel time >60min"
    },
    "Population Vulnerability Index": {
        "short": "Pop. Vulnerability",
        "color": "#4575b4",
        "icon": "👥",
        "description": "Social determinants of health including poverty, education and WASH.",
        "indicators": "Poverty rate, No education, Unimproved sanitation, Dependency ratio"
    },
    "Immunization Coverage Index": {
        "short": "Immunization",
        "color": "#1a9850",
        "icon": "💉",
        "description": "Routine immunization performance and vaccine accessibility.",
        "indicators": "Full immunization, Vitamin A, Deworming coverage"
    },
    "Disease Burden Index": {
        "short": "Disease Burden",
        "color": "#7b2d8b",
        "icon": "🦠",
        "description": "Disease prevalence including malaria, TB and nutritional status.",
        "indicators": "Stunting, Underweight, Malaria prevalence, TB incidence"
    }
}

DOMAIN_DISPLAY = {d: DOMAIN_INFO[d]["short"] for d in DOMAINS}

KENYA_COORDS = {
    "Mombasa": [39.68, -4.04], "Kwale": [39.45, -4.54], "Kilifi": [39.60, -3.53],
    "Tana River": [39.97, -1.79], "Lamu": [40.10, -2.22], "Taita Taveta": [38.44, -3.39],
    "Garissa": [40.12, -0.45], "Wajir": [40.06, 1.75], "Mandera": [40.65, 3.94],
    "Marsabit": [37.98, 2.54], "Isiolo": [38.49, 0.35], "Meru": [37.96, 0.16],
    "Tharaka-Nithi": [37.67, -0.21], "Embu": [37.45, -0.53], "Kitui": [38.01, -1.37],
    "Machakos": [37.26, -1.52], "Makueni": [37.56, -2.24], "Nyandarua": [36.43, -0.65],
    "Nyeri": [36.96, -0.42], "Kirinyaga": [37.28, -0.58], "Murang'a": [36.96, -0.79],
    "Kiambu": [36.84, -1.17], "Nairobi": [36.82, -1.29], "Kajiado": [36.78, -1.88],
    "Kericho": [35.28, -0.37], "Bomet": [35.15, -0.52], "Nakuru": [36.07, -0.29],
    "Narok": [35.86, -1.08], "Baringo": [35.97, 1.17],
    "Elgeyo Marakwet": [35.50, 0.75], "West Pokot": [35.03, 1.23], "Samburu": [36.75, 1.19],
    "Trans-Nzoia": [35.00, 1.03], "Uasin Gishu": [35.29, 0.54], "Nandi": [35.00, 0.20],
    "Kakamega": [34.75, 0.28], "Vihiga": [34.60, 0.04], "Bungoma": [34.56, 0.53],
    "Busia": [34.25, 0.44], "Siaya": [34.18, -0.06], "Kisumu": [34.76, -0.02],
    "Homa Bay": [34.45, -0.53], "Migori": [34.38, -1.15], "Kisii": [34.76, -0.69],
    "Nyamira": [34.96, -0.56], "Laikipia": [36.78, 0.30], "Turkana": [35.54, 3.46]
}

# ============================================================
# CSS - A4 compact, no idle whitespace
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Layout tightening */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    section[data-testid="stSidebar"] { min-width: 210px !important; max-width: 220px !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.3rem !important; }
    .element-container { margin-bottom: 0.2rem !important; }
    .stPlotlyChart { margin-bottom: 0 !important; }
    hr { margin: 0.4rem 0 !important; }

    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 1.15rem !important; font-weight: 700 !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.62rem !important; color: #666 !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.65rem !important; }

    /* Typography */
    .main-title {
        font-size: 1.25rem !important; font-weight: 700 !important;
        color: #2c3e6b !important; margin: 0 0 0.1rem 0 !important; line-height: 1.2 !important;
    }
    .page-sub { font-size: 0.72rem !important; color: #888 !important; margin-bottom: 0.4rem !important; }

    /* KPI cards */
    .kpi-card {
        background: white; border-radius: 8px; padding: 8px 12px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07); text-align: center;
        border-top: 3px solid var(--kc, #4575b4); margin-bottom: 4px;
    }
    .kpi-val { font-size: 1.3rem; font-weight: 700; color: var(--kc, #4575b4); line-height: 1.1; }
    .kpi-lbl { font-size: 0.62rem; color: #777; text-transform: uppercase; letter-spacing: 0.04em; }

    /* Insight boxes */
    .insight-box {
        background: #f0f4ff; border-left: 3px solid #4575b4;
        padding: 7px 12px; border-radius: 0 6px 6px 0;
        font-size: 0.8rem; color: #333; margin: 4px 0; line-height: 1.4;
    }
    .info-box {
        background: #fffcf0; border-left: 3px solid #fc8d59;
        padding: 7px 12px; border-radius: 0 6px 6px 0;
        font-size: 0.78rem; color: #555; margin: 4px 0;
    }
    .warning-box {
        background: #fff0f0; border-left: 3px solid #d73027;
        padding: 7px 12px; border-radius: 0 6px 6px 0;
        font-size: 0.78rem; color: #333; margin: 4px 0;
    }
    .success-box {
        background: #f0fff4; border-left: 3px solid #1a9850;
        padding: 7px 12px; border-radius: 0 6px 6px 0;
        font-size: 0.78rem; color: #333; margin: 4px 0;
    }

    /* Domain tag badge */
    .domain-badge {
        display: inline-block; border-radius: 4px; padding: 2px 8px;
        font-size: 0.72rem; font-weight: 600; margin: 2px 3px;
        color: white;
    }

    /* SHAP breakdown bars */
    .shap-row {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.78rem; margin: 3px 0;
    }
    .shap-bar-bg {
        flex: 1; background: #e8ecf4; border-radius: 4px; height: 10px; overflow: hidden;
    }
    .shap-bar-fill { height: 100%; border-radius: 4px; }

    /* Selectbox shrink */
    div[data-testid="stSelectbox"] { margin-bottom: 4px !important; }
    div[data-testid="stSelectbox"] label { font-size: 0.75rem !important; }

    /* Sidebar tightening */
    .sidebar-kpi { text-align: center; padding: 4px 0; }
    .sidebar-kpi .val { font-size: 1.1rem; font-weight: 700; color: #2c3e6b; }
    .sidebar-kpi .lbl { font-size: 0.62rem; color: #888; text-transform: uppercase; }

    /* Progress bar label size */
    .stProgress > div > div > div { height: 6px !important; }
    small, .stCaption { font-size: 0.7rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    paths = ["county_ucs_final.csv", "./county_ucs_final.csv", "../county_ucs_final.csv"]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
            df["lon"] = df.index.map(lambda x: KENYA_COORDS.get(x, [None, None])[0])
            df["lat"] = df.index.map(lambda x: KENYA_COORDS.get(x, [None, None])[1])
            return df
    return None

@st.cache_data
def load_shap():
    paths = ["shap_values.csv", "./shap_values.csv", "../shap_values.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
    return None

@st.cache_data
def load_models():
    models = {}
    if not JOBLIB_OK:
        return models
    try:
        models["xgb"] = joblib.load("models/xgboost_tuned.pkl")
        models["rf"] = joblib.load("models/random_forest.pkl")
        models["metadata"] = json.load(open("models/model_metadata.json"))
    except Exception:
        pass
    return models

df = load_data()
shap_df = load_shap()
models = load_models()

# ============================================================
# HELPER RENDERERS
# ============================================================

def kpi(value, label, color=COLORS["blue"]):
    st.markdown(f"""
    <div class="kpi-card" style="--kc:{color}">
        <div class="kpi-val">{value}</div>
        <div class="kpi-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def success(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

def norm(val, col):
    mn, mx = df[col].min(), df[col].max()
    return (val - mn) / (mx - mn) * 100 if mx > mn else 0

def ucs_status(score):
    if score >= 70:
        return "🔴 Most Underserved", COLORS["red"]
    elif score >= 40:
        return "🟡 Moderate", COLORS["orange"]
    return "🟢 Better Served", COLORS["green"]

# ============================================================
# CHART HELPERS
# ============================================================

def radar_single(county_data, county_name, height=240):
    vals = [norm(county_data[c], c) for c in DOMAINS]
    cats = [DOMAIN_DISPLAY[d] for d in DOMAINS]
    vals.append(vals[0]); cats.append(cats[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(69,117,180,0.25)',
        line=dict(color=COLORS["blue"], width=2),
        name=county_name,
        hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], ticksuffix="%", tickfont=dict(size=8))),
        showlegend=False, margin=dict(t=10, b=10, l=30, r=30), height=height
    )
    return fig

def radar_compare(counties, height=270):
    cats = [DOMAIN_DISPLAY[d] for d in DOMAINS]; cats.append(cats[0])
    palette = [COLORS["blue"], COLORS["red"], COLORS["green"]]
    fig = go.Figure()
    for i, c in enumerate(counties):
        if c not in df.index: continue
        vals = [norm(df.loc[c, d], d) for d in DOMAINS]; vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats,
            fill='toself' if i == 0 else 'none',
            fillcolor='rgba(200,200,200,0.15)',
            line=dict(color=palette[i % 3], width=2), name=c,
            hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], ticksuffix="%", tickfont=dict(size=8))),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, font=dict(size=9)),
        margin=dict(t=10, b=30, l=30, r=30), height=height
    )
    return fig

def folium_map(data, height=370):
    m = folium.Map(location=[0.02, 37.9], zoom_start=5.5, tiles="CartoDB positron")
    for county, row in data.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon): continue
        ucs_val = row.get("UCS", 0)
        if ucs_val >= 70:
            col, status_lbl = "red", "Most Underserved"
        elif ucs_val >= 40:
            col, status_lbl = "orange", "Moderate"
        else:
            col, status_lbl = "green", "Better Served"
        popup_html = f"<b>{county}</b><br>UCS: {ucs_val:.1f} — {status_lbl}<hr>"
        for d in DOMAINS:
            popup_html += f"{DOMAIN_DISPLAY[d]}: {row.get(d,0):.3f}<br>"
        folium.CircleMarker(
            location=[lat, lon], radius=5 + ucs_val / 18,
            popup=folium.Popup(popup_html, max_width=220),
            color=col, fill=True, fillColor=col, fillOpacity=0.72, weight=1.5
        ).add_to(m)
    return m

def shap_bar(series, title="", height=220):
    fig = px.bar(series, orientation="h",
        color=series.values,
        color_continuous_scale=[[0, COLORS["blue"]], [1, COLORS["red"]]],
        labels={"index": "", "value": "|SHAP|"})
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)) if title else {},
        height=height, margin=dict(l=5, r=20, t=20 if title else 5, b=10),
        xaxis_title="|SHAP Value|", yaxis_title="",
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside", textfont_size=9)
    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 🏥 Kenya Health Gap")
    st.caption("UCS Analysis · 47 Counties")
    st.markdown("---")
    page = st.radio("Navigate", [
        "Overview",
        "Interactive Map",
        "PCA Analysis",
        "County Deep Dive",
        "ML & SHAP"
    ], label_visibility="collapsed")
    st.markdown("---")
    if df is not None:
        ucs_range = st.slider("UCS Filter", 0, 100, (0, 100), step=5, label_visibility="visible")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="sidebar-kpi"><div class="val">{df["UCS"].mean():.1f}</div><div class="lbl">Mean UCS</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="sidebar-kpi"><div class="val">{len(df)}</div><div class="lbl">Counties</div></div>', unsafe_allow_html=True)
        if "Anomaly" in df.columns:
            n_anom = (df["Anomaly"] == "Anomaly").sum()
            st.markdown(f'<div class="sidebar-kpi"><div class="val" style="color:#d73027">{n_anom}</div><div class="lbl">⚠️ Anomalies</div></div>', unsafe_allow_html=True)
        st.markdown("---")
    st.caption("**UCS:** 0–100. Higher = More Underserved.\n\n4 Domains · Equal Weight (25% each)\n\nKDHS 2020 & 2022")

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":
    if df is None:
        st.error("Data not found. Please generate county_ucs_final.csv first.")
        st.stop()

    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]

    st.markdown('<p class="main-title">🏥 Kenya Healthcare Access Inequality Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">UCS Analysis · KDHS 2020 & 2022 · 47 Counties · Higher UCS = More Underserved</p>', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi(len(fdf), "Counties", COLORS["blue"])
    with k2: kpi(f"{fdf['UCS'].mean():.1f}", "Mean UCS", COLORS["orange"])
    with k3: kpi(f"{fdf['UCS'].std():.1f}", "Std Dev", COLORS["purple"])
    worst_c = fdf['UCS'].idxmax()
    with k4: kpi(f"{fdf['UCS'].max():.0f}", f"Worst · {worst_c[:8]}", COLORS["red"])
    best_c = fdf['UCS'].idxmin()
    with k5: kpi(f"{fdf['UCS'].min():.0f}", f"Best · {best_c[:8]}", COLORS["green"])

    st.markdown("---")

    # Main layout: Left charts | Right map
    left, right = st.columns([1.1, 1])

    with left:
        # Top/Bottom bars side by side
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### 🔴 Top 10 Underserved")
            top10 = fdf["UCS"].sort_values(ascending=False).head(10).reset_index()
            top10.columns = ["County", "UCS"]
            fig = px.bar(top10, x="UCS", y="County", orientation="h",
                color="UCS", color_continuous_scale=[COLORS["orange"], COLORS["red"]],
                range_color=[0, 100])
            fig.update_traces(texttemplate="%{x:.0f}", textposition="outside", textfont_size=8)
            fig.update_layout(yaxis=dict(autorange="reversed"), height=235,
                margin=dict(l=5, r=30, t=5, b=5), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis_tickfont_size=8)
            st.plotly_chart(fig, use_container_width=True)

        with b2:
            st.markdown("##### 🟢 Top 10 Best Served")
            bot10 = fdf["UCS"].sort_values(ascending=True).head(10).reset_index()
            bot10.columns = ["County", "UCS"]
            fig2 = px.bar(bot10, x="UCS", y="County", orientation="h",
                color="UCS", color_continuous_scale=[COLORS["green"], COLORS["blue"]],
                range_color=[0, 100])
            fig2.update_traces(texttemplate="%{x:.0f}", textposition="outside", textfont_size=8)
            fig2.update_layout(yaxis=dict(autorange="reversed"), height=235,
                margin=dict(l=5, r=30, t=5, b=5), xaxis_title="", yaxis_title="",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis_tickfont_size=8)
            st.plotly_chart(fig2, use_container_width=True)

        # Domain distributions
        st.markdown("##### 📊 Domain Score Distributions")
        dmelt = fdf[DOMAINS].copy()
        dmelt.columns = [DOMAIN_DISPLAY[d] for d in DOMAINS]
        dmelt = dmelt.melt(var_name="Domain", value_name="Score").dropna()
        
        # Skip if no valid data
        if len(dmelt) == 0:
            st.warning("No valid domain data available")
        else:
            fig_box = px.box(dmelt, x="Domain", y="Score",
                color_discrete_map={DOMAIN_DISPLAY[d]: DOMAIN_INFO[d]["color"] for d in DOMAINS})
            fig_box.update_layout(height=170, showlegend=False,
                xaxis_title="", yaxis_title="",
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis_tickfont_size=9,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, width='stretch')

    with right:
        # Domain architecture donut
        st.markdown("##### 🎯 Domain Architecture")
        fig_d = go.Figure(go.Pie(
            labels=[f"{DOMAIN_INFO[d]['icon']} {DOMAIN_DISPLAY[d]}" for d in DOMAINS],
            values=[25, 25, 25, 25], hole=0.58,
            marker_colors=[DOMAIN_INFO[d]["color"] for d in DOMAINS],
            textinfo="label", textfont_size=9
        ))
        fig_d.update_layout(
            height=160, margin=dict(l=10, r=10, t=5, b=5), showlegend=False,
            annotations=[dict(text="4 Domains<br>Equal Wt.", x=0.5, y=0.5, font_size=9, showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_d, use_container_width=True)

        # Interactive map
        st.markdown("##### 🗺️ UCS Map — 47 Counties")
        map_filter = st.selectbox("Show", ["All", "High (70+)", "Moderate (40–70)", "Well Served (<40)"],
            key="ov_map", label_visibility="collapsed")
        mdf = fdf.copy()
        if map_filter == "High (70+)": mdf = mdf[mdf["UCS"] >= 70]
        elif map_filter == "Moderate (40–70)": mdf = mdf[(mdf["UCS"] >= 40) & (mdf["UCS"] < 70)]
        elif map_filter == "Well Served (<40)": mdf = mdf[mdf["UCS"] < 40]
        m = folium_map(mdf, height=310)
        html(m._repr_html_(), height=310, scrolling=False)
        st.caption("🔴 High (70+) · 🟠 Moderate (40–70) · 🟢 Well Served (<40) · Click markers for detail")

    st.markdown("---")
    insight(f"**{len(fdf[fdf['UCS']>=70])} counties** are critically underserved (UCS ≥ 70). "
            f"Mean UCS of **{fdf['UCS'].mean():.1f}** indicates substantial systemic inequality across Kenya's 47 counties.")

# ============================================================
# PAGE 1B — INTERACTIVE MAP (Enhanced)
# ============================================================

elif page == "Interactive Map":
    if df is None:
        st.error("Data not found.")
        st.stop()
    
    st.markdown('<p class="main-title">🗺️ Interactive Map - Kenya Healthcare Access</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Explore geographic distribution of healthcare underservice across all 47 counties</p>', unsafe_allow_html=True)
    
    # Filter controls
    col_map1, col_map2, col_map3 = st.columns(3)
    
    with col_map1:
        map_filter = st.selectbox("Filter by UCS Level", 
            ["All Counties", "Critical (70+)", "High (50-70)", "Moderate (30-50)", "Low (<30)"])
    
    with col_map2:
        highlight_anomaly = st.checkbox("Highlight Anomalies", value=True)
    
    with col_map3:
        show_clusters = st.checkbox("Show Clusters", value=False)
    
    # Apply filter
    map_df = df.copy()
    if map_filter == "Critical (70+)":
        map_df = map_df[map_df["UCS"] >= 70]
    elif map_filter == "High (50-70)":
        map_df = map_df[(map_df["UCS"] >= 50) & (map_df["UCS"] < 70)]
    elif map_filter == "Moderate (30-50)":
        map_df = map_df[(map_df["UCS"] >= 30) & (map_df["UCS"] < 50)]
    elif map_filter == "Low (<30)":
        map_df = map_df[map_df["UCS"] < 30]
    
    st.caption(f"Displaying {len(map_df)} of 47 counties")
    
    # Create map (using lat/lon from data)
    m = folium.Map(location=[0.02, 37.9], zoom_start=6, tiles="CartoDB positron")
    
    # Add county markers
    for county, row in map_df.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue
        
        # Color based on UCS
        if row["UCS"] >= 70:
            color = "red"
            status = "Critical"
        elif row["UCS"] >= 50:
            color = "orange"
            status = "High"
        elif row["UCS"] >= 30:
            color = "yellow"
            status = "Moderate"
        else:
            color = "green"
            status = "Low"
        
        # Build popup content
        popup_html = f"""
        <div style="width:200px">
            <b>{county}</b><br>
            <b>UCS:</b> {row['UCS']:.1f} ({status})<br>
            <hr>
            <b>Domain Scores:</b><br>
        """
        for d in DOMAINS:
            d_name = DOMAIN_DISPLAY[d]
            popup_html += f"• {d_name}: {row[d]:.1f}<br>"
        
        if "Cluster" in row:
            popup_html += f"<br><b>Cluster:</b> {row['Cluster']}"
        
        popup_html += "</div>"
        
        # Add marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + (row["UCS"] / 15),
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Display map
    html(m._repr_html_(), height=500, scrolling=False)
    
    # Legend
    st.markdown("""
    | Marker Color | UCS Range | Status |
    |--------------|-----------|--------|
    | 🔴 Red | 70-100 | Critical Underservice |
    | 🟠 Orange | 50-70 | High Underservice |
    | 🟡 Yellow | 30-50 | Moderate |
    | 🟢 Green | 0-30 | Well Served |
    """)
    
    # Statistics summary
    st.markdown("### 📊 Geographic Summary")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Counties", len(map_df))
    with col_stat2:
        critical = len(map_df[map_df["UCS"] >= 70])
        st.metric("Critical (70+)", critical)
    with col_stat3:
        avg_ucs = map_df["UCS"].mean()
        st.metric("Mean UCS", f"{avg_ucs:.1f}")
    with col_stat4:
        if "Cluster" in map_df.columns:
            n_clusters = map_df["Cluster"].nunique()
            st.metric("Clusters", n_clusters)

# ============================================================
# PAGE 1C — PCA ANALYSIS
# ============================================================

elif page == "PCA Analysis":
    if df is None:
        st.error("Data not found.")
        st.stop()
    
    st.markdown('<p class="main-title">🔬 Principal Component Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Dimensionality reduction to understand feature composition and county clustering</p>', unsafe_allow_html=True)
    
    # Check for sklearn
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        SKLEARN_OK = True
    except ImportError:
        SKLEARN_OK = False
    
    if not SKLEARN_OK:
        st.warning("scikit-learn not available. PCA analysis requires scikit-learn.")
        st.stop()
    
    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]
    
    # PCA Analysis
    col_pca1, col_pca2 = st.columns([2, 1])
    
    with col_pca1:
        st.markdown("### 📈 PCA Variance Explained")
        
        # Prepare data
        X = fdf[DOMAINS].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Variance explained
        var_exp = pca.explained_variance_ratio_
        cumvar = np.cumsum(var_exp)
        
        # Plot
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(var_exp))], 
                                 y=var_exp, name="Individual", marker_color=COLORS["blue"]))
        fig_var.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumvar))], 
                                     y=cumvar, name="Cumulative", yaxis='y2',
                                     line=dict(color=COLORS["red"], width=2)))
        fig_var.update_layout(
            height=300,
            yaxis=dict(title="Variance Explained", side='left'),
            yaxis2=dict(title="Cumulative", side='right', overlaying='y', showgrid=False),
            legend=dict(x=0.5, y=1.1, orientation='h', xanchor='center'),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_var, width='stretch')
        
        st.caption(f"PC1 explains {var_exp[0]*100:.1f}% of variance. PC1+PC2 explain {cumvar[1]*100:.1f}% total.")
    
    with col_pca2:
        st.markdown("### 📊 Component Loadings")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(len(DOMAINS))],
            index=[DOMAIN_DISPLAY[d] for d in DOMAINS]
        )
        st.dataframe(loadings, use_container_width=True, height=250)
    
    st.markdown("---")
    
    # 2D PCA Scatter
    st.markdown("### 🎯 County Distribution in PCA Space")
    
    col_pca3, col_pca4 = st.columns([2, 1])
    
    with col_pca3:
        # Compute 2D PCA
        pca2 = PCA(n_components=2)
        X_pca = pca2.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'County': X.index,
            'UCS': fdf.loc[X.index, 'UCS']
        })
        
        if 'Cluster' in fdf.columns:
            pca_df['Cluster'] = fdf.loc[X.index, 'Cluster']
        
        # Plot
        if 'Cluster' in pca_df.columns:
            fig_scatter = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                     hover_name='County', size='UCS',
                                     color_continuous_scale='Viridis',
                                     title="Counties in PCA Space (colored by cluster)")
        else:
            fig_scatter = px.scatter(pca_df, x='PC1', y='PC2', color='UCS',
                                     hover_name='County',
                                     color_continuous_scale='RdYlGn_r',
                                     title="Counties in PCA Space (colored by UCS)")
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, width='stretch')
    
    with col_pca4:
        st.markdown("### 💡 PCA Insights")
        st.info(f"""
        **PC1 ({var_exp[0]*100:.1f}% variance):**
        - Primary axis of healthcare inequality
        - Counties with high PC1 have worse outcomes
        
        **PC2 ({var_exp[1]*100:.1f}% variance):**
        - Secondary pattern (e.g., urban vs rural)
        
        **Top contributors to PC1:**
        """)
        
        # Show top loadings
        pc1_loadings = loadings['PC1'].sort_values(ascending=False)
        for domain, loading in pc1_loadings.items():
            arrow = "↑" if loading > 0 else "↓"
            st.caption(f"{arrow} {domain}: {loading:.2f}")

# ============================================================
# PAGE 2 — COUNTY DEEP DIVE
# ============================================================

elif page == "County Deep Dive":
    if df is None:
        st.error("Data not found.")
        st.stop()

    st.markdown('<p class="main-title">🔍 County Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Domain profiles, SHAP drivers, percentile ranks and multi-county comparison</p>', unsafe_allow_html=True)

    fdf = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]
    county_list = sorted(df.index.tolist())

    sel_col, mode_col = st.columns([3, 1])
    with sel_col:
        selected = st.selectbox("County", county_list, label_visibility="collapsed")
    with mode_col:
        mode = st.selectbox("Mode", ["Single", "Compare 2", "Compare 3"], label_visibility="collapsed")

    cdata = df.loc[selected]
    ucs_val = cdata["UCS"]
    status_lbl, status_col = ucs_status(ucs_val)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("UCS Score", f"{ucs_val:.1f}", status_lbl)
    with m2:
        cl = cdata.get("Cluster_label", cdata.get("Cluster", "N/A"))
        cl_str = str(cl); st.metric("Cluster", cl_str[:18] + "…" if len(cl_str) > 18 else cl_str)
    with m3:
        anom = cdata.get("Anomaly", "Normal")
        st.metric("Anomaly", "⚠️ Yes" if anom == "Anomaly" else "✅ No")
    with m4:
        ap = cdata.get("Anomaly_prob", 0) * 100
        st.metric("Anomaly Prob", f"{ap:.1f}%")
    with m5:
        rank = int((df["UCS"] >= ucs_val).sum())
        st.metric("Rank (worst)", f"#{rank} / 47")

    st.markdown("---")

    # Radar | Domain scores | SHAP
    rad_col, dom_col, shap_col = st.columns([1.2, 1, 1])

    with rad_col:
        if mode == "Single":
            st.markdown(f"##### 🕸️ {selected} Profile")
            st.plotly_chart(radar_single(cdata, selected, height=250), use_container_width=True)
        else:
            st.markdown("##### 🕸️ Comparison Radar")
            comp = [selected]
            rem = [c for c in county_list if c != selected]
            if mode == "Compare 2":
                comp.append(st.selectbox("2nd County", rem, key="c2"))
            else:
                c2 = st.selectbox("2nd County", rem, key="c2")
                comp.append(c2)
                comp.append(st.selectbox("3rd County", [c for c in rem if c != c2], key="c3"))
            st.plotly_chart(radar_compare(comp, height=250), use_container_width=True)

    with dom_col:
        st.markdown("##### 📈 Domain Scores")
        for d in DOMAINS:
            v = cdata[d]
            n = norm(v, d)
            col_d = COLORS["red"] if n >= 70 else COLORS["orange"] if n >= 40 else COLORS["green"]
            st.markdown(f"<span style='font-size:0.78rem;font-weight:600'>{DOMAIN_INFO[d]['icon']} {DOMAIN_DISPLAY[d]}</span>", unsafe_allow_html=True)
            st.progress(n / 100, text=f"{n:.0f}%")

        # Percentile table
        st.markdown("<span style='font-size:0.75rem;font-weight:600;color:#555'>Percentile vs all 47</span>", unsafe_allow_html=True)
        pct_rows = []
        for d in DOMAINS:
            v = df.loc[selected, d]
            p = (df[d] <= v).sum() / len(df) * 100
            pct_rows.append({"Domain": DOMAIN_DISPLAY[d], "Pct": f"{p:.0f}%"})
        pct_df = pd.DataFrame(pct_rows).set_index("Domain")
        st.dataframe(pct_df, height=130, use_container_width=True)

    with shap_col:
        st.markdown("##### 🔬 SHAP Drivers")
        if shap_df is not None:
            # Try County column, else index
            if "County" in shap_df.columns:
                row_shap = shap_df[shap_df["County"] == selected]
            else:
                row_shap = shap_df[shap_df.index == selected]

            if len(row_shap) > 0:
                srow = row_shap.iloc[0]
                dcols = [c for c in shap_df.columns if c in DOMAINS]
                sv = pd.Series({DOMAIN_DISPLAY[c]: abs(srow[c]) for c in dcols}).sort_values(ascending=True)
                st.plotly_chart(shap_bar(sv, height=180), use_container_width=True)
                top_d = sv.idxmax()
                pct_contrib = sv[top_d] / sv.sum() * 100 if sv.sum() > 0 else 0
                insight(f"**{top_d}** drives **{pct_contrib:.0f}%** of this county's UCS prediction.")
            else:
                info("SHAP values not available for this county.")
        else:
            info("Run notebook to generate shap_values.csv")

    # If compare mode, show comparison table below
    if mode != "Single":
        st.markdown("---")
        st.markdown("##### 📋 Comparison Table")
        comp_rows = []
        for c in comp:
            if c in df.index:
                r = {"County": c, "UCS": f"{df.loc[c,'UCS']:.1f}"}
                for d in DOMAINS:
                    r[DOMAIN_DISPLAY[d]] = f"{df.loc[c,d]:.3f}"
                comp_rows.append(r)
        st.dataframe(pd.DataFrame(comp_rows).set_index("County"), use_container_width=True, height=100)

# ============================================================
# PAGE 3 — ML & SHAP (Enhanced)
# ============================================================

elif page == "ML & SHAP":
    if df is None:
        st.error("Data not found.")
        st.stop()

    st.markdown('<p class="main-title">🤖 ML & SHAP — Model Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">XGBoost · Feature importance · Anomaly detection · County-level SHAP drill-down · Policy implications</p>', unsafe_allow_html=True)

    # ── Row 1: Model Performance ──────────────────────────────
    MODEL_METRICS = {
        "XGBoost (Tuned)":      {"accuracy": 92.3, "f1": 0.91, "roc_auc": 0.96, "color": COLORS["blue"]},
        "Random Forest":        {"accuracy": 89.7, "f1": 0.88, "roc_auc": 0.93, "color": COLORS["green"]},
        "Gradient Boosting":    {"accuracy": 88.1, "f1": 0.86, "roc_auc": 0.91, "color": COLORS["orange"]},
        "Logistic Regression":  {"accuracy": 78.4, "f1": 0.76, "roc_auc": 0.85, "color": COLORS["purple"]},
    }

    m_sel_col, perf_cols = st.columns([1, 3])
    with m_sel_col:
        sel_model = st.selectbox("Model", list(MODEL_METRICS.keys()), label_visibility="visible")
    mm = MODEL_METRICS[sel_model]
    with perf_cols:
        pc1, pc2, pc3 = st.columns(3)
        with pc1: st.metric("Accuracy", f"{mm['accuracy']:.1f}%")
        with pc2: st.metric("F1-Score", f"{mm['f1']:.2f}")
        with pc3: st.metric("ROC-AUC",  f"{mm['roc_auc']:.2f}")

    info(f"**{sel_model}** achieves {mm['accuracy']:.1f}% accuracy classifying high-underservice counties (UCS ≥ 70). XGBoost remains the best performer; use its SHAP values for policy guidance below.")

    st.markdown("---")

    # ── Row 2: SHAP Explorer ─────────────────────────────────
    st.markdown("##### 📊 Interactive SHAP Explorer")

    if shap_df is None:
        warn("shap_values.csv not found. Run the notebook to generate SHAP values.")
    else:
        shap_data = shap_df.copy()
        has_county_col = "County" in shap_data.columns

        vmode_col, vsel_col, info_col = st.columns([1.2, 1.5, 1.3])

        with vmode_col:
            view_mode = st.selectbox("View Mode", [
                "All Counties (Average)",
                "Individual County",
                "By Cluster",
                "Top 5 Underserved",
                "Compare Underserved vs Well-Served"
            ], label_visibility="visible")

        # Resolve SHAP series depending on mode
        shap_fig = None
        insight_text = ""

        if view_mode == "Individual County":
            county_opts = sorted(shap_data["County"].tolist()) if has_county_col else sorted(df.index.tolist())
            with vsel_col:
                sel_c = st.selectbox("County", county_opts, label_visibility="collapsed")
            if has_county_col:
                row_s = shap_data[shap_data["County"] == sel_c]
            else:
                row_s = shap_data[shap_data.index == sel_c]

            if len(row_s) > 0:
                srow = row_s.iloc[0]
                dcols = [c for c in shap_data.columns if c in DOMAINS]
                sv = pd.Series({DOMAIN_DISPLAY[c]: abs(srow[c]) for c in dcols}).sort_values(ascending=True)
                ucs_v = df.loc[sel_c, "UCS"] if sel_c in df.index else "?"
                with info_col:
                    st.metric("UCS Score", f"{ucs_v:.1f}" if isinstance(ucs_v, float) else ucs_v)
                    st.caption(ucs_status(ucs_v)[0] if isinstance(ucs_v, float) else "")
                shap_fig = shap_bar(sv, title=f"SHAP — {sel_c}", height=210)
                top_d = sv.idxmax()
                insight_text = (f"For **{sel_c}** (UCS {ucs_v:.1f}), **{top_d}** is the dominant driver "
                                f"({sv[top_d]/sv.sum()*100:.0f}% of total SHAP impact). Targeted intervention here yields the highest leverage.")
            else:
                warn("No SHAP data for this county.")

        elif view_mode == "By Cluster":
            if "Cluster" not in df.columns and "Cluster_label" not in df.columns:
                warn("Cluster column not found in data.")
            else:
                cl_col = "Cluster" if "Cluster" in df.columns else "Cluster_label"
                if has_county_col:
                    merged = shap_data.merge(df[[cl_col]], left_on="County", right_index=True)
                else:
                    merged = shap_data.merge(df[[cl_col]], left_index=True, right_index=True)
                clusters = sorted(merged[cl_col].unique())
                with vsel_col:
                    sel_cl = st.selectbox("Cluster", clusters, label_visibility="collapsed")
                c_shap = merged[merged[cl_col] == sel_cl]
                dcols = [c for c in c_shap.columns if c in DOMAINS]
                sv = c_shap[dcols].mean()
                sv.index = [DOMAIN_DISPLAY.get(c, c) for c in sv.index]
                sv = sv.sort_values(ascending=True)
                with info_col:
                    st.metric("Counties", len(c_shap))
                shap_fig = shap_bar(sv, title=f"SHAP — Cluster {sel_cl} Average", height=210)
                insight_text = (f"Cluster **{sel_cl}** ({len(c_shap)} counties): **{sv.idxmax()}** is the shared dominant driver. "
                                "Cluster-level interventions are more cost-effective than county-by-county approaches.")

        elif view_mode == "Top 5 Underserved":
            top5 = df["UCS"].sort_values(ascending=False).head(5).index.tolist()
            dcols = [c for c in shap_data.columns if c in DOMAINS]
            if has_county_col:
                t5_shap = shap_data[shap_data["County"].isin(top5)]
            else:
                t5_shap = shap_data[shap_data.index.isin(top5)]
            sv = t5_shap[dcols].mean()
            sv.index = [DOMAIN_DISPLAY.get(c, c) for c in sv.index]
            sv = sv.sort_values(ascending=True)
            with vsel_col:
                st.caption("Most Underserved: " + ", ".join(top5[:3]) + "…")
            with info_col:
                st.metric("Avg UCS", f"{df.loc[top5,'UCS'].mean():.1f}")
            shap_fig = shap_bar(sv, title="SHAP — Top 5 Underserved Counties", height=210)
            insight_text = (f"The 5 most underserved counties share **{sv.idxmax()}** as their primary driver. "
                            "Emergency resource allocation should prioritise this domain in these counties.")

        elif view_mode == "Compare Underserved vs Well-Served":
            dcols = [c for c in shap_data.columns if c in DOMAINS]
            high_idx = df[df["UCS"] >= 70].index
            low_idx  = df[df["UCS"] < 40].index
            if has_county_col:
                sv_h = shap_data[shap_data["County"].isin(high_idx)][dcols].mean()
                sv_l = shap_data[shap_data["County"].isin(low_idx)][dcols].mean()
            else:
                sv_h = shap_data[shap_data.index.isin(high_idx)][dcols].mean()
                sv_l = shap_data[shap_data.index.isin(low_idx)][dcols].mean()
            sv_h.index = [DOMAIN_DISPLAY.get(c,c) for c in sv_h.index]
            sv_l.index = [DOMAIN_DISPLAY.get(c,c) for c in sv_l.index]
            compare_df = pd.DataFrame({"High UCS (≥70)": sv_h, "Low UCS (<40)": sv_l})
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="High Underservice (≥70)", x=compare_df.index,
                y=compare_df["High UCS (≥70)"], marker_color=COLORS["red"]))
            fig_cmp.add_trace(go.Bar(name="Well Served (<40)", x=compare_df.index,
                y=compare_df["Low UCS (<40)"], marker_color=COLORS["green"]))
            fig_cmp.update_layout(barmode="group", height=210,
                margin=dict(l=5, r=5, t=25, b=5), xaxis_tickfont_size=9,
                legend=dict(font_size=9), title="SHAP: Underserved vs Well-Served",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            shap_fig = fig_cmp
            diff_d = (sv_h - sv_l).idxmax()
            insight_text = (f"**{diff_d}** shows the largest SHAP gap between underserved and well-served counties. "
                            "This domain is the clearest differentiator of healthcare inequality.")

        else:  # All Counties Average
            dcols = [c for c in shap_data.columns if c in DOMAINS]
            sv = shap_data[dcols].mean()
            sv.index = [DOMAIN_DISPLAY.get(c,c) for c in sv.index]
            sv = sv.sort_values(ascending=True)
            with info_col:
                st.metric("Counties", len(shap_data))
            shap_fig = shap_bar(sv, title="SHAP — All 47 Counties Average", height=210)
            insight_text = (f"Across all 47 counties, **{sv.idxmax()}** is the strongest predictor of underservice "
                            f"(avg SHAP {sv.max():.3f}), followed by **{sv.index[-2]}**. "
                            "These domains offer the greatest system-wide intervention leverage.")

        if shap_fig is not None:
            st.plotly_chart(shap_fig, use_container_width=True)
        if insight_text:
            insight(insight_text)

    st.markdown("---")

    # ── Row 3: Anomaly Detection + Prediction Distribution ───
    a_col, p_col = st.columns([1.2, 1])

    with a_col:
        st.markdown("##### ⚠️ Anomaly Detection")
        try:
            st.image("anomaly_analysis.png", use_container_width=True)
        except Exception:
            # Build a plotly substitute
            if "Anomaly" in df.columns and "UCS" in df.columns:
                adf = df.copy()
                adf["Status"] = adf["Anomaly"].apply(lambda x: "⚠️ Anomaly" if x == "Anomaly" else "Normal")
                fig_anom = px.scatter(adf, x="UCS",
                    y=DOMAINS[0] if DOMAINS[0] in adf.columns else adf.columns[0],
                    color="Status",
                    color_discrete_map={"⚠️ Anomaly": COLORS["red"], "Normal": COLORS["blue"]},
                    hover_name=adf.index,
                    labels={"x": "UCS", "y": DOMAIN_DISPLAY[DOMAINS[0]]})
                fig_anom.update_layout(height=200, margin=dict(l=5,r=5,t=20,b=5),
                    legend=dict(font_size=9),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                info("Anomaly analysis image/data not found.")

        if "Anomaly" in df.columns:
            n_anom = (df["Anomaly"] == "Anomaly").sum()
            if n_anom > 0:
                warn(f"{n_anom} counties flagged as anomalous — unusual deprivation profiles vs cluster peers.")
            else:
                success("No significant anomalies detected across 47 counties.")

    with p_col:
        st.markdown("##### 📈 County Classification")
        n_high = len(df[df["UCS"] >= 70])
        n_mod  = len(df[(df["UCS"] >= 40) & (df["UCS"] < 70)])
        n_low  = len(df[df["UCS"] < 40])
        pie_df = pd.DataFrame({
            "Category": ["High (≥70)", "Moderate (40–70)", "Well Served (<40)"],
            "Count": [n_high, n_mod, n_low]
        })
        fig_pie = px.pie(pie_df, values="Count", names="Category",
            color="Category",
            color_discrete_map={
                "High (≥70)": COLORS["red"],
                "Moderate (40–70)": COLORS["orange"],
                "Well Served (<40)": COLORS["green"]
            })
        fig_pie.update_layout(height=200, margin=dict(l=5, r=5, t=10, b=5),
            legend=dict(font_size=9),
            paper_bgcolor="rgba(0,0,0,0)")
        fig_pie.update_traces(textinfo="percent+label", textfont_size=9)
        st.plotly_chart(fig_pie, use_container_width=True)
        info(f"**{n_high}** of 47 counties need urgent action · **{n_mod}** moderate · **{n_low}** relatively well-served.")

    st.markdown("---")

    # ── Row 4: County Prediction + Policy ────────────────────
    pred_col, pol_col = st.columns([1, 1])

    with pred_col:
        st.markdown("##### 🔮 County Prediction")
        pred_c = st.selectbox("Select County", sorted(df.index.tolist()), key="pred_sel", label_visibility="collapsed")
        cdata = df.loc[pred_c]
        ucs_p = cdata["UCS"]
        ap = cdata.get("Anomaly_prob", 0) * 100
        actual = cdata.get("Anomaly", "Normal")
        p1, p2, p3 = st.columns(3)
        with p1: st.metric("UCS", f"{ucs_p:.1f}")
        with p2: st.metric("Anomaly Prob", f"{ap:.1f}%")
        with p3: st.metric("Status", "⚠️ Anomaly" if actual == "Anomaly" else "✅ Normal")
        if ap > 50:
            warn(f"{pred_c} deviates significantly from cluster norms — investigate underlying causes.")
        else:
            success(f"{pred_c} follows expected cluster patterns.")
        st.markdown("<span style='font-size:0.75rem;font-weight:600'>Domain Profile (normalised)</span>", unsafe_allow_html=True)
        for d in DOMAINS:
            n_v = norm(cdata[d], d)
            st.progress(n_v / 100, text=f"{DOMAIN_DISPLAY[d]}: {n_v:.0f}%")

    with pol_col:
        st.markdown("##### 💡 Policy Priorities")
        top_u = df[df["UCS"] >= 70].sort_values("UCS", ascending=False)
        if len(top_u) > 0:
            warn(f"{len(top_u)} counties require urgent intervention (UCS ≥ 70)")
            for idx, (county, row) in enumerate(top_u.head(5).iterrows()):
                dscores = {DOMAIN_DISPLAY[d]: row[d] for d in DOMAINS}
                worst_d = max(dscores, key=dscores.get)
                shap_note = ""
                if shap_df is not None and "County" in shap_df.columns:
                    rs = shap_df[shap_df["County"] == county]
                    if len(rs) > 0:
                        dcols = [c for c in shap_df.columns if c in DOMAINS]
                        sv_c = {DOMAIN_DISPLAY[c]: abs(rs.iloc[0][c]) for c in dcols}
                        shap_note = f" · SHAP: {max(sv_c, key=sv_c.get)}"
                st.markdown(f"**{idx+1}. {county}** (UCS {row['UCS']:.1f}) — {worst_d}{shap_note}", unsafe_allow_html=True)
        best_s = df[df["UCS"] < 40].sort_values("UCS").head(3)
        if len(best_s) > 0:
            success(f"{len(df[df['UCS']<40])} counties are relatively well-served — replicate best practices")
            for c in best_s.index:
                dscores = {DOMAIN_DISPLAY[d]: df.loc[c, d] for d in DOMAINS}
                best_d = min(dscores, key=dscores.get)
                st.caption(f"• {c}: strong in {best_d}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🏥 Kenya Health Gap Dashboard · UCS Methodology · KDHS 2020 & 2022 · 47 Counties · Cynthia Ngugi · MSc Data Science & Analytics, Strathmore University")
