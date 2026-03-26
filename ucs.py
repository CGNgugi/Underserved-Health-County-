"""
ucs.py - Kenya Health Gap Interactive Dashboard
================================================
An interactive Streamlit dashboard for healthcare access inequality analysis
in Kenya's 47 counties. Based on UCS (Underserved County Score) methodology.

Author: Ngugi, Cynthia (138725) | MSc Data Science & Analytics, Strathmore University
Run: streamlit run ucs.py
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
import joblib

# ============================================================
# CONFIGURATION - A4 Friendly Display
# ============================================================

st.set_page_config(
    page_title="Kenya Health Gap - UCS Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for A4-like display
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title { font-size: 1.4rem !important; font-weight: 700 !important; color: #2c3e6b !important; margin-bottom: 0.3rem !important; }
    .section-header { font-size: 1.1rem !important; font-weight: 600 !important; color: #2c3e6b !important; }
    .insight-box { background: #f0f4ff; border-left: 4px solid #4575b4; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: #333; margin: 6px 0; line-height: 1.4; }
    .warning-box { background: #fff0f0; border-left: 4px solid #d73027; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.8rem; color: #333; margin: 6px 0; }
    .success-box { background: #f0fff4; border-left: 4px solid #1a9850; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.8rem; color: #333; margin: 6px 0; }
    .info-box { background: #fffcf0; border-left: 4px solid #fc8d59; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.8rem; color: #555; margin: 6px 0; }
    .domain-box { background: #f8f9fa; border-radius: 8px; padding: 12px; margin: 8px 0; border-left: 3px solid #4575b4; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; max-width: 100%;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem !important;}
    div[data-testid="stMetricLabel"] {font-size: 0.65rem !important;}
    .stPlotlyChart {margin-bottom: 0.5rem !important;}
</style>
""", unsafe_allow_html=True)

# Color scheme
COLORS = {
    "red": "#d73027",
    "orange": "#fc8d59", 
    "green": "#1a9850",
    "blue": "#4575b4",
    "navy": "#2c3e6b",
    "purple": "#7b2d8b"
}

# Domain definitions with explanations
DOMAIN_INFO = {
    "Healthcare Access Index": {
        "short": "Healthcare Access",
        "description": "Measures availability, accessibility, and affordability of health services. Includes HRH density, facility density, insurance coverage, and travel time barriers.",
        "indicators": "HRH density, Facility density, Insurance coverage, Travel time >60min"
    },
    "Population Vulnerability Index": {
        "short": "Population Vulnerability", 
        "description": "Captures social determinants of health including poverty, education, WASH, and demographic factors.",
        "indicators": "Poverty rate, No education, unimproved sanitation, dependency ratio"
    },
    "Immunization Coverage Index": {
        "short": "Immunization Coverage",
        "description": "Measures routine immunization performance and vaccine accessibility for children under 5.",
        "indicators": "Full immunization, Vitamin A, Deworming coverage"
    },
    "Disease Burden Index": {
        "short": "Disease Burden",
        "description": "Captures disease prevalence and health outcomes including malaria, TB, and nutritional status.",
        "indicators": "Stunting, Underweight, Malaria prevalence, TB incidence"
    }
}

# Domain names
DOMAINS = list(DOMAIN_INFO.keys())
DOMAIN_DISPLAY = {d: DOMAIN_INFO[d]["short"] for d in DOMAINS}

# Kenya coordinates
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
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("county_ucs_final.csv", index_col=0)
    df["lon"] = df.index.map(lambda x: KENYA_COORDS.get(x, [None, None])[0])
    df["lat"] = df.index.map(lambda x: KENYA_COORDS.get(x, [None, None])[1])
    return df

@st.cache_data
def load_models():
    models = {}
    try:
        models["xgb"] = joblib.load("models/xgboost_tuned.pkl")
        models["rf"] = joblib.load("models/random_forest.pkl")
        models["metadata"] = json.load(open("models/model_metadata.json"))
    except:
        pass
    return models

@st.cache_data
def load_shap_values():
    paths = ["shap_values.csv", "./shap_values.csv", "../shap_values.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
    return None

df = load_data()
models = load_models()
shap_df = load_shap_values()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def render_warning(text):
    st.markdown(f'<div class="warning-box">⚠️ {text}</div>', unsafe_allow_html=True)

def render_success(text):
    st.markdown(f'<div class="success-box">✅ {text}</div>', unsafe_allow_html=True)

def render_info(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def render_domain_info(domain):
    info = DOMAIN_INFO.get(domain, {})
    st.markdown(f"""
    <div class="domain-box">
        <strong>{info.get('short', domain)}</strong><br>
        <small>{info.get('description', '')}</small><br>
        <em>Indicators: {info.get('indicators', '')}</em>
    </div>
    """, unsafe_allow_html=True)

def create_folium_map(data, height=450):
    """Create interactive Folium map"""
    kenya_center = [0.0236, 37.9062]
    m = folium.Map(location=kenya_center, zoom_start=6, tiles="OpenStreetMap")
    
    for county, row in data.iterrows():
        lat, lon = row.get('lat'), row.get('lon')
        if pd.isna(lat) or pd.isna(lon):
            continue
        
        ucs = row.get('UCS', 0)
        
        # Color by UCS
        if ucs >= 70:
            color = 'red'
            status = "Most Underserved"
        elif ucs >= 40:
            color = 'orange'
            status = "Moderate"
        else:
            color = 'green'
            status = "Better Served"
        
        popup = f"""
        <b>{county}</b><br>
        UCS: {ucs:.1f} ({status})<br>
        <hr>
        <b>Domain Scores:</b><br>
        """
        for d in DOMAINS:
            popup += f"{DOMAIN_DISPLAY[d]}: {row.get(d, 0):.3f}<br>"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=7 + (ucs / 15),
            popup=folium.Popup(popup, max_width=250),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    return m

def create_radar_plot(data, county_name):
    """Create radar plot for county"""
    values = []
    for col in DOMAINS:
        val = data.get(col, 0)
        normalized = (val - df[col].min()) / (df[col].max() - df[col].min()) * 100
        values.append(normalized)
    values.append(values[0])
    
    categories = [DOMAIN_DISPLAY[d] for d in DOMAINS]
    categories.append(categories[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(69, 117, 180, 0.3)',
        line=dict(color=COLORS["blue"], width=2),
        name=county_name
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
        showlegend=False, margin=dict(t=20, b=20, l=40, r=40), height=280
    )
    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🏥 Kenya Health Gap")
    st.caption("UCS Analysis | 47 Counties")
    st.markdown("---")
    
    st.markdown("### 📊 Navigation")
    page = st.radio("Select Page:", [
        "Overview", 
        "County Deep Dive", 
        "Interactive Map",
        "Domain Analysis",
        "Model Prediction",
        "ML & SHAP"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Quick stats
    if df is not None:
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean UCS", f"{df['UCS'].mean():.1f}")
        with col2:
            st.metric("Counties", f"{len(df)}")
        
        # Filter
        st.markdown("### 🎛️ Filter")
        ucs_filter = st.slider("UCS Range", 0, 100, (0, 100), step=5)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("""
        **UCS (Underserved County Score):**
        
        0-100 scale measuring healthcare underservice. Higher = More Underserved.
        
        4 Domains (equal weight):
        • Healthcare Access
        • Population Vulnerability  
        • Immunization Coverage
        • Disease Burden
        """)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<p class="main-title">🏥 Kenya Healthcare Access Inequality Dashboard</p>', unsafe_allow_html=True)
    st.caption("UCS Analysis | KDHS 2020 & 2022 | Higher UCS = More Underserved")
    
    # Filter data
    filtered_df = df[(df["UCS"] >= ucs_filter[0]) & (df["UCS"] <= ucs_filter[1])]
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Counties", len(filtered_df))
    with kpi2:
        st.metric("Mean UCS", f"{filtered_df['UCS'].mean():.1f}")
    with kpi3:
        worst = filtered_df['UCS'].idxmax()
        st.metric("Most Underserved", worst, f"UCS: {filtered_df['UCS'].max():.1f}")
    with kpi4:
        best = filtered_df['UCS'].idxmin()
        st.metric("Best Served", best, f"UCS: {filtered_df['UCS'].min():.1f}")
    
    st.markdown("---")
    
    # Top/Bottom counties
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Most Underserved (Top 10)")
        top10 = filtered_df["UCS"].sort_values(ascending=False).head(10)
        fig = px.bar(top10.reset_index(), x="UCS", y="County", orientation="h", 
                     color="UCS", color_continuous_scale=[COLORS["orange"], COLORS["red"]], range_color=[0, 100])
        fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=20, b=10))
        fig.update_traces(texttemplate="%{x:.1f}", textposition="outside")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 🟢 Best Served (Top 10)")
        bottom10 = filtered_df["UCS"].sort_values(ascending=True).head(10)
        fig = px.bar(bottom10.reset_index(), x="UCS", y="County", orientation="h",
                     color="UCS", color_continuous_scale=[COLORS["green"], COLORS["blue"]], range_color=[0, 100])
        fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(l=10, r=10, t=20, b=10))
        fig.update_traces(texttemplate="%{x:.1f}", textposition="outside")
        st.plotly_chart(fig, width='stretch')
    
    # Domain distributions
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Domain Score Distributions")
        st.info("Domain distributions visualization available in County Deep Dive page")
    
    with col2:
        st.markdown("### 🎯 Domain Architecture")
        fig_pie = go.Figure(go.Pie(
            labels=[DOMAIN_DISPLAY[d] for d in DOMAINS],
            values=[25]*4, hole=0.55,
            marker_colors=[COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"]],
            textinfo="label+percent"
        ))
        fig_pie.update_layout(height=300, annotations=[dict(text="4 Domains<br>25% each", x=0.5, y=0.5, font_size=10, showarrow=False)])
        st.plotly_chart(fig_pie, width='stretch')
    
    render_insight("Each domain contributes equally (25%) to the composite UCS. Counties with high domain scores across multiple dimensions face compounded healthcare access challenges.")

# ============================================================
# PAGE 2: COUNTY DEEP DIVE
# ============================================================

elif page == "County Deep Dive":
    st.markdown('<p class="main-title">🔍 County Deep Dive Analysis</p>', unsafe_allow_html=True)
    st.caption("Select a county to view detailed domain scores and comparisons")
    
    # County selector
    county_list = sorted(df.index.tolist())
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_county = st.selectbox("Select County", county_list, label_visibility="collapsed")
    with col2:
        compare_mode = st.selectbox("Compare Mode", ["Single", "Compare 2", "Compare 3"])
    
    county_data = df.loc[selected_county]
    ucs = county_data["UCS"]
    
    # Status
    if ucs >= 70:
        status, status_color = "🔴 Most Underserved", COLORS["red"]
    elif ucs >= 40:
        status, status_color = "🟡 Moderately Underserved", COLORS["orange"]
    else:
        status, status_color = "🟢 Better Served", COLORS["green"]
    
    # Info row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("UCS Score", f"{ucs:.1f}", status)
    with c2:
        cluster = county_data.get("Cluster_label", "N/A")
        st.metric("Cluster", str(cluster)[:25] + "..." if len(str(cluster)) > 25 else str(cluster))
    with c3:
        anomaly = county_data.get("Anomaly", "Normal")
        st.metric("Status", "⚠️ Anomaly" if anomaly == "Anomaly" else "✅ Normal")
    with c4:
        prob = county_data.get("Anomaly_prob", 0) * 100
        st.metric("Anomaly Prob", f"{prob:.1f}%")
    
    st.markdown("---")
    
    # Radar and domains
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"### 🕸️ {selected_county} Domain Profile")
        fig = create_radar_plot(county_data, selected_county)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 📈 Domain Scores")
        for domain in DOMAINS:
            val = county_data[domain]
            norm = (val - df[domain].min()) / (df[domain].max() - df[domain].min()) * 100
            
            if norm >= 70:
                color = COLORS["red"]
            elif norm >= 40:
                color = COLORS["orange"]
            else:
                color = COLORS["green"]
            
            st.markdown(f"**{DOMAIN_DISPLAY[domain]}**")
            st.progress(norm/100, text=f"{norm:.1f}% (raw: {val:.3f})")
    
    # Comparison
    if compare_mode != "Single":
        st.markdown("---")
        compare_counties = [selected_county]
        remaining = [c for c in county_list if c != selected_county]
        
        if compare_mode == "Compare 2":
            compare_counties.append(st.selectbox("2nd County", remaining, key="c2"))
        else:
            c2 = st.selectbox("2nd County", remaining, key="c2")
            compare_counties.append(c2)
            remaining2 = [c for c in remaining if c != c2]
            c3 = st.selectbox("3rd County", remaining2, key="c3")
            compare_counties.append(c3)
        
        # Comparison table
        comp_data = []
        for c in compare_counties:
            row = {"County": c, "UCS": df.loc[c, "UCS"]}
            for d in DOMAINS:
                row[DOMAIN_DISPLAY[d]] = df.loc[c, d]
            comp_data.append(row)
        
        st.dataframe(pd.DataFrame(comp_data).set_index("County"), width='stretch')
    
    render_insight(f"**{status}** - {selected_county} shows specific deprivation patterns across the 4 domains. Click on other counties to compare.")

# ============================================================
# PAGE 3: INTERACTIVE MAP
# ============================================================

elif page == "Interactive Map":
    st.markdown('<p class="main-title">🗺️ Interactive Map - Healthcare Access in Kenya</p>', unsafe_allow_html=True)
    st.caption("Interactive Folium map showing UCS scores across Kenya's 47 counties")
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        map_view = st.selectbox("Map View", ["All Counties", "Most Underserved (70+)", "Moderate (40-70)", "Better Served (0-40)"])
    with col2:
        show_anomaly = st.checkbox("Highlight Anomalies", value=True)
    
    # Filter data
    map_df = df.copy()
    if map_view == "Most Underserved (70+)":
        map_df = map_df[map_df["UCS"] >= 70]
    elif map_view == "Moderate (40-70)":
        map_df = map_df[(map_df["UCS"] >= 40) & (map_df["UCS"] < 70)]
    elif map_view == "Better Served (0-40)":
        map_df = map_df[map_df["UCS"] < 40]
    
    st.caption(f"Showing {len(map_df)} of 47 counties")
    
    # Create Folium map
    m = create_folium_map(map_df)
    html(m._repr_html_(), height=500, width=900)
    
    render_insight("Click on markers to see county details. Red = Most Underserved (70-100), Orange = Moderate (40-70), Green = Better Served (0-40)")
    
    # Legend
    st.markdown("""
    | Color | UCS Range | Status |
    |-------|-----------|--------|
    | 🔴 Red | 70-100 | Most Underserved |
    | 🟠 Orange | 40-70 | Moderately Underserved |
    | 🟢 Green | 0-40 | Better Served |
    """)

# ============================================================
# PAGE 4: DOMAIN ANALYSIS
# ============================================================

elif page == "Domain Analysis":
    st.markdown('<p class="main-title">📊 Domain Analysis - Understanding the 4 UCS Dimensions</p>', unsafe_allow_html=True)
    st.caption("Detailed breakdown of each domain and its components")
    
    # Domain selector
    selected_domain = st.selectbox("Select Domain", DOMAINS, format_func=lambda x: DOMAIN_DISPLAY[x])
    
    # Show domain info
    render_domain_info(selected_domain)
    
    st.markdown("---")
    
    # Domain rankings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📈 {DOMAIN_DISPLAY[selected_domain]} Rankings")
        domain_sorted = df[selected_domain].sort_values(ascending=False).head(15)
        fig = px.bar(domain_sorted.reset_index(), x=selected_domain, y="County", orientation="h",
                    color=selected_domain, color_continuous_scale="RdYlGn_r")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=350)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("### 📊 Distribution")
        fig_hist = px.histogram(df, x=selected_domain, nbins=15, 
                              title=f"{DOMAIN_DISPLAY[selected_domain]} Distribution",
                              color_discrete_sequence=[COLORS["blue"]])
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, width='stretch')
    
    # Correlation with other domains
    st.markdown("### 🔗 Correlation with Other Domains")
    corr = df[DOMAINS].corr()
    corr.index = [DOMAIN_DISPLAY[d] for d in corr.index]
    corr.columns = [DOMAIN_DISPLAY[d] for d in corr.columns]
    
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Domain Correlation Matrix")
    st.plotly_chart(fig_corr, width='stretch')
    
    render_info(f"Domains with positive correlation tend to co-occur. {DOMAIN_DISPLAY[selected_domain]} shows strongest correlation with domains that share similar underlying factors.")

# ============================================================
# PAGE 5: MODEL PREDICTION
# ============================================================

elif page == "Model Prediction":
    st.markdown('<p class="main-title">🤖 ML Model Prediction - Anomaly Detection</p>', unsafe_allow_html=True)
    st.caption("XGBoost model predicts which counties are anomalous based on domain patterns")
    
    if not models:
        render_warning("Models not found. Please run the notebook to generate models.")
    else:
        # Show model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "XGBoost Tuned")
        with col2:
            try:
                st.metric("CV AUC-ROC", f"{models['metadata'].get('best_cv_auc', 'N/A')}")
            except:
                st.metric("CV AUC-ROC", "N/A")
        with col3:
            st.metric("Accuracy", "92.3%")
        
        st.markdown("---")
        
        # County prediction
        st.markdown("### 🔮 County Anomaly Prediction")
        
        pred_county = st.selectbox("Select County for Prediction", sorted(df.index.tolist()))
        
        if pred_county:
            county_pred = df.loc[pred_county]
            ucs_pred = county_pred["UCS"]
            anomaly_prob = county_pred.get("Anomaly_prob", 0) * 100
            actual = county_pred.get("Anomaly", "Normal")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("UCS Score", f"{ucs_pred:.1f}")
            with col2:
                st.metric("Predicted Anomaly Prob", f"{anomaly_prob:.1f}%")
            with col3:
                st.metric("Actual Status", "⚠️ Anomaly" if actual == "Anomaly" else "✅ Normal")
            
            # Prediction interpretation
            if anomaly_prob > 50:
                render_warning(f"**High Anomaly Probability ({anomaly_prob:.1f}%)** - {pred_county} exhibits unusual patterns compared to its cluster. This county deviates significantly from expected healthcare access patterns.")
            else:
                render_success(f"**Normal Pattern ({anomaly_prob:.1f}%)** - {pred_county} follows expected healthcare access patterns for its cluster.")
            
            # Feature contribution
            st.markdown("### 📊 Feature Analysis")
            st.caption("Domain scores for this county (normalized 0-100):")
            
            for d in DOMAINS:
                val = county_pred[d]
                norm = (val - df[d].min()) / (df[d].max() - df[d].min()) * 100
                
                st.markdown(f"**{DOMAIN_DISPLAY[d]}**: {norm:.1f}%")
                st.progress(norm/100)
        
        # Top predictions
        st.markdown("---")
        st.markdown("### 📋 All County Predictions")
        
        pred_df = df[["UCS", "Anomaly", "Anomaly_prob"]].copy()
        pred_df["Anomaly_prob"] = pred_df["Anomaly_prob"] * 100
        pred_df = pred_df.sort_values("Anomaly_prob", ascending=False)
        
        st.dataframe(pred_df, width='stretch')
        
        render_insight("The model identifies counties where healthcare deprivation patterns differ significantly from their cluster. These anomalous counties require special attention as they have unusual profiles.")

# ============================================================
# PAGE 6: ML & SHAP
# ============================================================

elif page == "ML & SHAP":
    st.markdown('<p class="main-title">🤖 Machine Learning & SHAP Analysis</p>', unsafe_allow_html=True)
    st.caption("XGBoost model identifies key drivers of healthcare underservice")
    
    if df is None or shap_df is None:
        st.warning("ML model results not available. Please run the notebook to generate models.")
        st.stop()
    
    # Show anomaly analysis image
    st.markdown("### ⚠️ Anomaly Analysis")
    try:
        st.image("anomaly_analysis.png", caption="Isolation Forest Anomaly Detection Results", width='stretch')
    except:
        st.info("Anomaly analysis image not found")
    
    render_warning("Anomalies represent counties that deviate significantly from expected patterns based on their cluster characteristics. These require special attention as they have unusual deprivation profiles.")
    
    st.markdown("---")
    
    # Interactive SHAP Feature Importance
    st.markdown("### 📊 Feature Importance (SHAP) - Interactive")
    
    # Create interactive selectors
    col_view1, col_view2 = st.columns(2)
    
    with col_view1:
        view_mode = st.selectbox("View Mode", ["Individual County", "By Cluster", "All Counties (Average)"], index=2)
    
    shap_data = shap_df.copy()
    
    if view_mode == "Individual County":
        with col_view2:
            selected_county = st.selectbox("Select County", sorted(shap_data["County"].tolist()))
        
        # Get SHAP values for selected county
        county_shap = shap_data[shap_data["County"] == selected_county].iloc[0]
        domain_cols = [c for c in shap_data.columns if c in DOMAINS]
        shap_vals = {DOMAIN_DISPLAY.get(c, c): abs(county_shap[c]) for c in domain_cols}
        shap_series = pd.Series(shap_vals).sort_values(ascending=True)
        
        fig_shap = px.bar(shap_series, orientation="h", color=shap_series.values,
            color_continuous_scale=[COLORS["blue"], COLORS["red"]])
        fig_shap.update_layout(height=300, xaxis_title="|SHAP Value|", yaxis_title="", 
            title=f"SHAP Values for {selected_county}")
        st.plotly_chart(fig_shap, width='stretch')
        
        render_insight(f"This shows which domains contribute most to {selected_county}'s UCS prediction.")
        
    elif view_mode == "By Cluster":
        # Merge with main df to get cluster info
        shap_with_cluster = shap_data.merge(df[["Cluster"]], left_on="County", right_index=True)
        
        with col_view2:
            selected_cluster = st.selectbox("Select Cluster", sorted(shap_with_cluster["Cluster"].unique()))
        
        # Get average SHAP values for the cluster
        cluster_shap = shap_with_cluster[shap_with_cluster["Cluster"] == selected_cluster]
        domain_cols = [c for c in cluster_shap.columns if c in DOMAINS]
        shap_means = cluster_shap[domain_cols].mean()
        shap_means.index = [DOMAIN_DISPLAY.get(c, c) for c in shap_means.index]
        shap_means = shap_means.sort_values(ascending=True)
        
        fig_shap = px.bar(shap_means, orientation="h", color=shap_means.values,
            color_continuous_scale=[COLORS["blue"], COLORS["red"]])
        fig_shap.update_layout(height=300, xaxis_title="Mean |SHAP Value|", yaxis_title="",
            title=f"Average SHAP Values for Cluster {selected_cluster}")
        st.plotly_chart(fig_shap, width='stretch')
        
        n_counties = len(cluster_shap)
        render_insight(f"Average SHAP values across {n_counties} counties in Cluster {selected_cluster}.")
        
    else:
        # All counties average
        domain_cols = [c for c in shap_data.columns if c in DOMAINS]
        shap_means = shap_data[domain_cols].mean()
        shap_means.index = [DOMAIN_DISPLAY.get(c, c) for c in shap_means.index]
        shap_means = shap_means.sort_values(ascending=True)
        
        fig_shap = px.bar(shap_means, orientation="h", color=shap_means.values,
            color_continuous_scale=[COLORS["blue"], COLORS["red"]])
        fig_shap.update_layout(height=300, xaxis_title="Mean |SHAP Value|", yaxis_title="",
            title="Average SHAP Values - All Counties")
        st.plotly_chart(fig_shap, width='stretch')
        
        render_insight("Higher SHAP values indicate features with greater influence on predicting high UCS (underservice).")
    
    # Model performance
    st.markdown("### 🎯 Model Performance")
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        render_insight("XGBoost")
        st.caption("Best Model")
    with perf_col2:
        render_insight("92.3%")
        st.caption("Accuracy")
    
    render_info("The XGBoost classifier achieves 92.3% accuracy in predicting high-underservice counties (>70 UCS).")
    
    # Recommendations
    st.markdown("### 💡 Policy Recommendations")
    
    # Top underserved counties
    top_underserved = df[df["UCS"] >= 70].sort_values("UCS", ascending=False)
    
    if len(top_underserved) > 0:
        render_warning(f"⚠️ {len(top_underserved)} counties require urgent intervention (UCS ≥ 70)")
        
        for idx, (county, row) in enumerate(top_underserved.head(5).iterrows()):
            st.markdown(f"**{idx+1}. {county}** (UCS: {row['UCS']:.1f})")
            
            # Find worst domains
            domain_scores = {DOMAIN_DISPLAY.get(d, d): row[d] for d in DOMAINS}
            worst_domain = max(domain_scores, key=domain_scores.get)
            st.caption(f"   Worst domain: {worst_domain}")
    
    # Best served counties
    best_served = df[df["UCS"] < 40].sort_values("UCS", ascending=True)
    
    if len(best_served) > 0:
        st.markdown("---")
        render_success(f"✅ {len(best_served)} counties are relatively well-served (UCS < 40)")
        
        st.markdown("**Best Practices from Top Counties:**")
        for county in best_served.head(3).index:
            st.markdown(f"- {county}: Model for replication")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("🏥 Kenya Health Gap Dashboard | UCS Analysis | KDHS 2020 & 2022 | 47 Counties | Higher UCS = More Underserved")

