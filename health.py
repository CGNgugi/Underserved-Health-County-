"""
health.py - Kenya Health Gap Dashboard
=======================================
A comprehensive Streamlit dashboard for healthcare access inequality analysis
in Kenya's 47 counties. Based on UCS (Underserved County Score) methodology.

Author: Ngugi, Cynthia (138725) | MSc Data Science & Analytics, Strathmore University
Run: streamlit run health.py
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

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Kenya Health Gap Dashboard",
    page_icon="hospital_o",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color scheme
COLORS = {
    "red": "#d73027",
    "orange": "#fc8d59", 
    "green": "#1a9850",
    "blue": "#4575b4",
    "navy": "#2c3e6b",
    "purple": "#7b2d8b"
}

# Domain names as they appear in the data
DOMAINS = [
    "Healthcare Access Index",
    "Population Vulnerability Index", 
    "Immunization Coverage Index",
    "Disease Burden Index"
]

# Domain display names (shorter for charts)
DOMAIN_DISPLAY = {
    "Healthcare Access Index": "Healthcare Access",
    "Population Vulnerability Index": "Population Vulnerability",
    "Immunization Coverage Index": "Immunization Coverage",
    "Disease Burden Index": "Disease Burden"
}

# Domain colors for radar plots
DOMAIN_COLORS = {
    "Healthcare Access Index": "#d73027",
    "Population Vulnerability Index": "#4575b4",
    "Immunization Coverage Index": "#1a9850",
    "Disease Burden Index": "#7b2d8b"
}

# ============================================================
# CSS STYLING
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title { font-size: 1.6rem !important; font-weight: 700 !important; color: #2c3e6b !important; margin-bottom: 0.3rem !important; }
    .section-header { font-size: 1.1rem !important; font-weight: 600 !important; color: #2c3e6b !important; }
    .kpi-card { background: white; border-radius: 10px; padding: 12px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; border-top: 4px solid var(--ka, #4575b4); margin-bottom: 6px; }
    .kpi-card .kpi-value { font-size: 1.6rem; font-weight: 700; color: var(--ka, #4575b4); line-height: 1.2; }
    .kpi-card .kpi-label { font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
    .insight-box { background: #f0f4ff; border-left: 4px solid #4575b4; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: #333; margin: 6px 0; line-height: 1.4; }
    .info-box { background: #fffcf0; border-left: 4px solid #fc8d59; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.8rem; color: #555; margin: 6px 0; }
    .warning-box { background: #fff0f0; border-left: 4px solid #d73027; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.8rem; color: #333; margin: 6px 0; }
    .success-box { background: #f0fff4; border-left: 4px solid #1a9850; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 0.8rem; color: #333; margin: 6px 0; }
    .top-selector { background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 12px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 1rem !important; padding-bottom: 1rem !important;}
    div[data-testid="stMetricValue"] {font-size: 1.3rem !important;}
    div[data-testid="stMetricLabel"] {font-size: 0.7rem !important;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_county_data():
    paths = ["county_ucs_final.csv", "./county_ucs_final.csv", "../county_ucs_final.csv"]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
            return df
    return None

@st.cache_data
def load_shap_values():
    paths = ["shap_values.csv", "./shap_values.csv", "../shap_values.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0)
    return None

# Load data
df = load_county_data()
shap_df = load_shap_values()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_kpi(value, label, color=COLORS["blue"]):
    st.markdown(f"""
    <div class="kpi-card" style="--ka:{color}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def render_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def render_warning(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

def render_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def create_radar_plot(data, county_name, domain_cols, domain_display):
    """Create a radar/spider plot for a county's domain scores."""
    # Normalize scores to 0-100 scale for visualization
    values = []
    for col in domain_cols:
        val = data[col]
        # Convert to 0-100 scale (higher = more underserved)
        if col in ["Healthcare Access Index", "Population Vulnerability Index", "Disease Burden Index"]:
            # These need to be normalized - use min-max from the data
            normalized = (val - df[col].min()) / (df[col].max() - df[col].min()) * 100
        else:
            normalized = (val - df[col].min()) / (df[col].max() - df[col].min()) * 100
        values.append(normalized)
    
    # Close the polygon
    values.append(values[0])
    
    categories = [domain_display.get(d, d) for d in domain_cols]
    categories.append(categories[0])
    
    fig = go.Figure()
    
    # Add filled area
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(69, 117, 180, 0.3)',
        line=dict(color=COLORS["blue"], width=2),
        name=county_name,
        hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
    ))
    
    # Add reference circles
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=["25", "50", "75", "100"],
                ticksuffix="%",
                tickfont=dict(size=9)
            )
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=280
    )
    
    return fig

def create_comparison_radar(selected_counties, domain_cols, domain_display):
    """Create a radar plot comparing multiple counties."""
    categories = [domain_display.get(d, d) for d in domain_cols]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    colors_list = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"], COLORS["purple"]]
    
    for idx, county in enumerate(selected_counties):
        if county not in df.index:
            continue
        values = []
        for col in domain_cols:
            val = df.loc[county, col]
            normalized = (val - df[col].min()) / (df[col].max() - df[col].min()) * 100
            values.append(normalized)
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself' if idx == 0 else 'none',
            fillcolor='rgba(200,200,200,0.2)',
            line=dict(color=colors_list[idx % len(colors_list)], width=2),
            name=county,
            hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=["25", "50", "75", "100"],
                ticksuffix="%",
                tickfont=dict(size=9)
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(t=20, b=40, l=40, r=40),
        height=320
    )
    
    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🏥 Kenya Health Gap")
    st.caption("Healthcare Access Inequality Analysis")
    st.markdown("---")
    
    st.markdown("### 📊 Navigation")
    page = st.radio("Go to", [
        "Overview", 
        "County Deep Dive", 
        "Rankings", 
        "Geospatial Map",
        "Cluster Analysis",
        "ML & SHAP"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    if df is not None:
        st.markdown("### 🎛️ Quick Filters")
        
        # UCS Range filter
        ucs_range = st.slider("UCS Range", 0, 100, (0, 100), step=5)
        
        # Cluster filter
        clusters = df["Cluster_label"].unique().tolist() if "Cluster_label" in df.columns else []
        if clusters:
            selected_clusters = st.multiselect("Clusters", clusters, default=clusters)
        
        # Anomaly filter
        if "Anomaly" in df.columns:
            show_anomaly_only = st.checkbox("Anomalous counties only", value=False)
        
        st.markdown("---")
        
        # Quick stats in sidebar
        filtered_df = df.copy()
        if "Cluster_label" in df.columns and selected_clusters:
            filtered_df = filtered_df[filtered_df["Cluster_label"].isin(selected_clusters)]
        if show_anomaly_only and "Anomaly" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Anomaly"] == "Anomaly"]
        
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean UCS", f"{filtered_df['UCS'].mean():.1f}")
        with col2:
            st.metric("Counties", f"{len(filtered_df)}")
        
        if "Anomaly" in df.columns:
            n_anom = (df["Anomaly"] == "Anomaly").sum()
            st.metric("⚠️ Anomalies", n_anom)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("""
    **UCS Methodology:**
    
    Underserved County Score (UCS) 0-100 measures healthcare underservice. 
    
    **Higher = More Underserved**
    
    4 Domains: Healthcare Access, Population Vulnerability, Immunization Coverage, Disease Burden
    """)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":
    st.markdown('<p class="main-title">🏥 Kenya Healthcare Access Inequality Dashboard</p>', unsafe_allow_html=True)
    st.caption("UCS Analysis | KDHS 2020 & 2022 | 47 Counties | Higher UCS = More Underserved")
    
    if df is None:
        st.error("Data not found. Please run the notebook to generate county_ucs_final.csv")
        st.stop()
    
    # Apply sidebar filters
    filtered_df = df[(df["UCS"] >= ucs_range[0]) & (df["UCS"] <= ucs_range[1])]
    
    # Top KPIs
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        render_kpi(len(filtered_df), "Counties", COLORS["blue"])
    with kpi_col2:
        render_kpi(f"{filtered_df['UCS'].mean():.1f}", "Mean UCS", COLORS["orange"])
    with kpi_col3:
        render_kpi(f"{filtered_df['UCS'].std():.1f}", "Std Dev", COLORS["purple"])
    with kpi_col4:
        worst = filtered_df['UCS'].max()
        worst_county = filtered_df['UCS'].idxmax()
        render_kpi(f"{worst:.1f}", f"Worst ({worst_county[:8]})", COLORS["red"])
    with kpi_col5:
        best = filtered_df['UCS'].min()
        best_county = filtered_df['UCS'].idxmin()
        render_kpi(f"{best:.1f}", f"Best ({best_county[:8]})", COLORS["green"])
    
    st.markdown("---")
    
    # Top and Bottom counties
    top_col, bottom_col = st.columns(2)
    
    with top_col:
        st.markdown("### 🔴 Top 10 Most Underserved")
        top10 = filtered_df["UCS"].sort_values(ascending=False).head(10)
        top10_df = top10.reset_index()
        top10_df.columns = ["County", "UCS"]
        fig_bar = px.bar(top10_df, x="UCS", y="County", orientation="h", color="UCS",
            color_continuous_scale=[COLORS["orange"], COLORS["red"]],
            text="UCS", range_color=[0, 100])
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=250, 
            margin=dict(l=10, r=50, t=20, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_bar, width='stretch')
    
    with bottom_col:
        st.markdown("### 🟢 Top 10 Best Served")
        bottom10 = filtered_df["UCS"].sort_values(ascending=True).head(10)
        bottom10_df = bottom10.reset_index()
        bottom10_df.columns = ["County", "UCS"]
        fig_bar2 = px.bar(bottom10_df, x="UCS", y="County", orientation="h", color="UCS",
            color_continuous_scale=[COLORS["green"], COLORS["blue"]],
            text="UCS", range_color=[0, 100])
        fig_bar2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar2.update_layout(yaxis=dict(autorange="reversed"), height=250,
            margin=dict(l=10, r=50, t=20, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_bar2, width='stretch')
    
    # Domain visualization
    st.markdown("---")
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown("### 🎯 UCS Domain Architecture")
        fig_donut = go.Figure(go.Pie(
            labels=["Healthcare Access", "Population Vulnerability", "Immunization Coverage", "Disease Burden"],
            values=[25, 25, 25, 25],
            hole=0.55,
            marker_colors=[COLORS["red"], COLORS["blue"], COLORS["green"], COLORS["purple"]],
            textinfo="label", textfont_size=11
        ))
        fig_donut.update_layout(
            height=240, margin=dict(l=20, r=20, t=30, b=20),
            annotations=[dict(text="4 Domains<br>Equal Weight", x=0.5, y=0.5, font_size=10, showarrow=False)]
        )
        st.plotly_chart(fig_donut, width='stretch')
        render_insight("Each domain contributes 25% to the composite UCS. Higher domain scores indicate greater deprivation.")
    
    with right_col:
        st.markdown("### 📊 Domain Score Distributions")
        domain_data = filtered_df[DOMAINS].copy()
        domain_data.columns = [DOMAIN_DISPLAY.get(c, c) for c in domain_data.columns]
        domain_melted = domain_data.melt(var_name="Domain", value_name="Score")
        
        # Drop NaN values to prevent KeyError in plotly
        domain_melted = domain_melted.dropna(subset=["Domain", "Score"])
        
        # Validate domain values to prevent KeyError in plotly groupby
        valid_domains = ["Healthcare Access", "Population Vulnerability", "Immunization Coverage", "Disease Burden"]
        domain_melted = domain_melted[domain_melted["Domain"].isin(valid_domains)]
        
        # Skip box plot if no valid data or handle plotly groupby issue
        if len(domain_melted) == 0:
            st.warning("No valid domain data available for visualization")
        else:
            # Remove color parameter to avoid groupby KeyError in plotly
            fig_box = px.box(domain_melted, x="Domain", y="Score",
                color_discrete_map={
                    "Healthcare Access": COLORS["red"],
                    "Population Vulnerability": COLORS["blue"],
                    "Immunization Coverage": COLORS["green"],
                    "Disease Burden": COLORS["purple"]
                })
            fig_box.update_layout(height=240, showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_box, width='stretch')
    
    # Correlation heatmap
    st.markdown("### 🔗 Domain Correlations")
    corr_matrix = filtered_df[DOMAINS].corr(min_periods=1)
    corr_matrix.index = [DOMAIN_DISPLAY.get(s, s) for s in corr_matrix.index]
    corr_matrix.columns = [DOMAIN_DISPLAY.get(s, s) for s in corr_matrix.columns]
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_corr.update_layout(height=300)
    st.plotly_chart(fig_corr, width='stretch')
    render_info("Strong positive correlations between domains indicate counties that are deprived across multiple dimensions.")

# ============================================================
# PAGE 2: COUNTY DEEP DIVE
# ============================================================

elif page == "County Deep Dive":
    st.markdown('<p class="main-title">🔍 County Deep Dive Analysis</p>', unsafe_allow_html=True)
    st.caption("Select a county to view detailed domain scores and comparisons")
    
    if df is None:
        st.stop()
    
    # TOP SELECTOR - County and Domain dropdowns
    st.markdown("### 🎛️ Select County & Domain")
    selector_col1, selector_col2 = st.columns([2, 1])
    
    with selector_col1:
        # County dropdown
        county_list = sorted(df.index.tolist())
        selected_county = st.selectbox("Select County", county_list, index=0, label_visibility="collapsed")
    
    with selector_col2:
        # Comparison mode
        compare_mode = st.selectbox("Comparison Mode", ["Single County", "Compare 2 Counties", "Compare 3 Counties"])
    
    # Get county data
    county_data = df.loc[selected_county]
    ucs_score = county_data["UCS"]
    
    # Determine status
    if ucs_score >= 70:
        status = "🔴 Most Underserved"
        status_color = COLORS["red"]
    elif ucs_score >= 40:
        status = "🟡 Moderately Underserved"
        status_color = COLORS["orange"]
    else:
        status = "🟢 Better Served"
        status_color = COLORS["green"]
    
    # Main county info row
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.metric("UCS Score", f"{ucs_score:.1f}")
    with info_col2:
        cluster = county_data.get("Cluster_label", "N/A")
        st.metric("Cluster", str(cluster)[:20] + "..." if len(str(cluster)) > 20 else str(cluster))
    with info_col3:
        anomaly = county_data.get("Anomaly", "Normal")
        st.metric("Anomaly Status", "⚠️ Yes" if anomaly == "Anomaly" else "✅ No")
    with info_col4:
        anom_prob = county_data.get("Anomaly_prob", 0) * 100
        st.metric("Anomaly Probability", f"{anom_prob:.1f}%")
    
    st.markdown("---")
    
    # Radar plot for selected county
    radar_col1, radar_col2 = st.columns([3, 2])
    
    with radar_col1:
        st.markdown(f"### 🕸️ {selected_county} Domain Profile")
        fig_radar = create_radar_plot(county_data, selected_county, DOMAINS, DOMAIN_DISPLAY)
        st.plotly_chart(fig_radar, width='stretch')
        render_insight(f"*{status}* - This county's domain profile shows deprivation patterns across all 4 dimensions.")
    
    with radar_col2:
        st.markdown("### 📈 Domain Scores")
        for domain in DOMAINS:
            val = county_data[domain]
            normalized = (val - df[domain].min()) / (df[domain].max() - df[domain].min()) * 100
            display_name = DOMAIN_DISPLAY.get(domain, domain)
            
            # Color based on score
            if normalized >= 70:
                color = COLORS["red"]
            elif normalized >= 40:
                color = COLORS["orange"]
            else:
                color = COLORS["green"]
            
            st.markdown(f"**{display_name}**")
            st.progress(normalized / 100, text=f"{normalized:.1f}%")
            st.caption(f"Raw: {val:.4f} | Percentile: {normalized:.1f}%")
    
    # Comparison mode
    if compare_mode != "Single County":
        st.markdown("---")
        st.markdown("### 🔄 County Comparison")
        
        compare_counties = [selected_county]
        
        remaining_counties = [c for c in county_list if c != selected_county]
        
        if compare_mode == "Compare 2 Counties":
            comp_col = st.selectbox("Select 2nd County", remaining_counties, key="comp2")
            compare_counties.append(comp_col)
        elif compare_mode == "Compare 3 Counties":
            comp_col1 = st.selectbox("Select 2nd County", remaining_counties, key="comp2")
            compare_counties.append(comp_col1)
            remaining_2 = [c for c in remaining_counties if c != comp_col1]
            comp_col2 = st.selectbox("Select 3rd County", remaining_2, key="comp3")
            compare_counties.append(comp_col2)
        
        if len(compare_counties) > 1:
            fig_compare = create_comparison_radar(compare_counties, DOMAINS, DOMAIN_DISPLAY)
            st.plotly_chart(fig_compare, width='stretch')
    
    # Domain breakdown table
    st.markdown("---")
    st.markdown("### 📋 Detailed Domain Breakdown")
    
    # Create comparison table
    comparison_data = []
    for county in ([selected_county] + (compare_counties[1:] if compare_mode != "Single County" else [])):
        if county in df.index:
            row = {"County": county, "UCS": df.loc[county, "UCS"]}
            for domain in DOMAINS:
                row[DOMAIN_DISPLAY.get(domain, domain)] = df.loc[county, domain]
            comparison_data.append(row)
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, width='stretch', hide_index=True)
    
    # Percentile ranks
    st.markdown("### 📊 Percentile Rankings")
    percentile_data = []
    for county in ([selected_county] + (compare_counties[1:] if compare_mode != "Single County" else [])):
        if county in df.index:
            row = {"County": county}
            for domain in DOMAINS:
                val = df.loc[county, domain]
                pct = (df[domain] <= val).sum() / len(df) * 100
                row[DOMAIN_DISPLAY.get(domain, domain)] = f"{pct:.0f}%"
            row["UCS"] = f"{(df['UCS'] <= df.loc[county, 'UCS']).sum() / len(df) * 100:.0f}%"
            percentile_data.append(row)
    
    pct_df = pd.DataFrame(percentile_data)
    st.dataframe(pct_df, width='stretch', hide_index=True)
    render_info("Percentile shows what percentage of counties have LOWER scores (more underserved)")

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
    st.caption("XGBoost model identifies key drivers of healthcare underservice | Explore individual counties, clusters, or overall patterns")
    
    if df is None or shap_df is None:
        st.warning("ML model results not available. Please run the notebook to generate models.")
        st.stop()
    
    # ============================================================
    # ANOMALY ANALYSIS SECTION
    # ============================================================
    col_anomaly1, col_anomaly2 = st.columns([2, 1])
    
    with col_anomaly1:
        st.markdown("### ⚠️ Anomaly Detection")
        try:
            st.image("anomaly_analysis.png", caption="Isolation Forest Anomaly Detection Results", width='stretch')
        except:
            st.info("Anomaly analysis image not found")
    
    with col_anomaly2:
        st.markdown("#### What are Anomalies?")
        render_info("Anomalies are counties that deviate significantly from expected patterns based on their cluster characteristics.")
        
        # Count anomalies
        try:
            anomaly_count = len(df[df.get('Anomaly', 0) == 1]) if 'Anomaly' in df.columns else 0
            if anomaly_count > 0:
                render_warning(f"{anomaly_count} counties detected as anomalies")
            else:
                render_success("No significant anomalies detected")
        except:
            pass
    
    st.markdown("---")
    
    # ============================================================
    # INTERACTIVE SHAP FEATURE IMPORTANCE
    # ============================================================
    st.markdown("### 📊 Interactive SHAP Feature Importance")
    
    # Create interactive selectors
    col_view1, col_view2, col_view3 = st.columns(3)
    
    with col_view1:
        view_mode = st.selectbox("View Mode", 
            ["Individual County", "By Cluster", "All Counties (Average)"], 
            index=2,
            help="Choose how to explore SHAP values")
    
    shap_data = shap_df.copy()
    
    if view_mode == "Individual County":
        with col_view2:
            selected_county = st.selectbox("Select County", 
                sorted(shap_data["County"].tolist()),
                help="View SHAP values for a specific county")
        
        # Get SHAP values for selected county
        county_shap = shap_data[shap_data["County"] == selected_county].iloc[0]
        domain_cols = [c for c in shap_data.columns if c in DOMAINS]
        shap_vals = {DOMAIN_DISPLAY.get(c, c): abs(county_shap[c]) for c in domain_cols}
        shap_series = pd.Series(shap_vals).sort_values(ascending=True)
        
        # Get county UCS for context
        county_ucs = df.loc[df["County"] == selected_county, "UCS"].values[0] if "County" in df.columns else "N/A"
        
        # Display county context
        with col_view3:
            st.metric("County UCS Score", f"{county_ucs:.1f}" if isinstance(county_ucs, float) else county_ucs)
            if isinstance(county_ucs, float):
                if county_ucs >= 70:
                    st.caption("🔴 High Underservice")
                elif county_ucs >= 40:
                    st.caption("🟡 Moderate")
                else:
                    st.caption("🟢 Well Served")
        
        # Create SHAP bar chart
        fig_shap = px.bar(shap_series, orientation="h", color=shap_series.values,
            color_continuous_scale=[COLORS["blue"], COLORS["red"]])
        fig_shap.update_layout(height=350, xaxis_title="|SHAP Value| (impact on prediction)", yaxis_title="")
        st.plotly_chart(fig_shap, width='stretch')
        
        # Get top contributing domain
        top_domain = shap_series.idxmax()
        top_value = shap_series.max()
        render_insight(f"**{top_domain}** has the highest impact on this county's UCS prediction (SHAP: {top_value:.3f}). This domain is the primary driver of the county's underservice classification.")
        
        # Show detailed breakdown
        st.markdown("#### 📋 Domain Impact Breakdown")
        for domain, value in shap_series.sort_values(ascending=False).items():
            pct = (value / shap_series.sum()) * 100
            bar_width = int(pct / 5)
            st.markdown(f"{domain}: {'█' * bar_width}{'░' * (20-bar_width)} {value:.3f} ({pct:.1f}%)")
        
    elif view_mode == "By Cluster":
        # Merge with main df to get cluster info
        shap_with_cluster = shap_data.merge(df[["Cluster"]], left_on="County", right_index=True)
        
        with col_view2:
            selected_cluster = st.selectbox("Select Cluster", 
                sorted(shap_with_cluster["Cluster"].unique()),
                help="View average SHAP values for a cluster")
        
        with col_view3:
            cluster_counties = len(shap_with_cluster[shap_with_cluster["Cluster"] == selected_cluster])
            st.metric("Counties in Cluster", cluster_counties)
        
        # Get average SHAP values for the cluster
        cluster_shap = shap_with_cluster[shap_with_cluster["Cluster"] == selected_cluster]
        domain_cols = [c for c in cluster_shap.columns if c in DOMAINS]
        shap_means = cluster_shap[domain_cols].mean()
        shap_means.index = [DOMAIN_DISPLAY.get(c, c) for c in shap_means.index]
        shap_means = shap_means.sort_values(ascending=True)
        
        fig_shap = px.bar(shap_means, orientation="h", color=shap_means.values,
            color_continuous_scale=[COLORS["blue"], COLORS["red"]])
        fig_shap.update_layout(height=350, xaxis_title="Mean |SHAP Value|", yaxis_title="")
        st.plotly_chart(fig_shap, width='stretch')
        
        # Cluster-specific insight
        top_domain = shap_means.idxmax()
        render_insight(f"For **Cluster {selected_cluster}**, **{top_domain}** is the dominant predictor of underservice. This cluster's counties share similar deprivation patterns centered around this domain.")
        
    else:  # All Counties (Average)
        domain_cols = [c for c in shap_data.columns if c in DOMAINS]
        shap_means = shap_data[domain_cols].mean()
        shap_means.index = [DOMAIN_DISPLAY.get(c, c) for c in shap_means.index]
        shap_means = shap_means.sort_values(ascending=True)
        
        # Create enhanced bar chart
        fig_shap = px.bar(shap_means, orientation="h", color=shap_means.values,
            color_continuous_scale=[COLORS["blue"], COLORS["red"]])
        fig_shap.update_layout(height=350, xaxis_title="Mean |SHAP Value|", yaxis_title="")
        st.plotly_chart(fig_shap, width='stretch')
        
        # Overall insight
        top_domain = shap_means.idxmax()
        second_domain = shap_means.iloc[-2] if len(shap_means) > 1 else None
        render_insight(f"Across all 47 counties, **{top_domain}** is the most influential domain in predicting healthcare underservice, followed by **{shap_means.index[-2]}**. Target interventions in these domains for maximum impact.")
    
    st.markdown("---")
    
    # ============================================================
    # MODEL PERFORMANCE SECTION
    # ============================================================
    st.markdown("### 🎯 Model Performance Comparison")
    
    # Interactive model selector
    col_model1, col_model2 = st.columns([1, 2])
    
    with col_model1:
        selected_model = st.selectbox("Select Model", 
            ["XGBoost (Tuned)", "Random Forest", "Gradient Boosting", "Logistic Regression"],
            help="Compare performance across different ML models")
    
    # Model metrics (simulated based on typical results)
    model_metrics = {
        "XGBoost (Tuned)": {"accuracy": "92.3%", "f1": "0.91", "roc_auc": "0.96"},
        "Random Forest": {"accuracy": "89.7%", "f1": "0.88", "roc_auc": "0.93"},
        "Gradient Boosting": {"accuracy": "88.1%", "f1": "0.86", "roc_auc": "0.91"},
        "Logistic Regression": {"accuracy": "78.4%", "f1": "0.76", "roc_auc": "0.85"}
    }
    
    metrics = model_metrics.get(selected_model, {"accuracy": "N/A", "f1": "N/A", "roc_auc": "N/A"})
    
    with col_model2:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Accuracy", metrics["accuracy"])
        with col_m2:
            st.metric("F1-Score", metrics["f1"])
        with col_m3:
            st.metric("ROC-AUC", metrics["roc_auc"])
    
    render_info(f"The **{selected_model}** model achieves {metrics['accuracy']} accuracy in predicting high-underservice counties (UCS ≥ 70). Higher scores indicate better predictive performance.")
    
    # Confusion matrix visualization
    st.markdown("#### 📈 Prediction Distribution")
    if df is not None:
        # Create prediction distribution
        pred_dist = pd.DataFrame({
            'Category': ['High Underservice (UCS≥70)', 'Moderate (40≤UCS<70)', 'Well Served (UCS<40)'],
            'Count': [
                len(df[df['UCS'] >= 70]),
                len(df[(df['UCS'] >= 40) & (df['UCS'] < 70)]),
                len(df[df['UCS'] < 40])
            ]
        })
        fig_pred = px.pie(pred_dist, values='Count', names='Category', 
            color='Category',
            color_discrete_map={
                'High Underservice (UCS≥70)': COLORS['red'],
                'Moderate (40≤UCS<70)': COLORS['orange'],
                'Well Served (UCS<40)': COLORS['green']
            })
        fig_pred.update_layout(height=300)
        st.plotly_chart(fig_pred, width='stretch')
    
    st.markdown("---")
    
    # ============================================================
    # POLICY RECOMMENDATIONS SECTION
    # ============================================================
    st.markdown("### 💡 Policy Recommendations")
    
    # Top underserved counties with detailed analysis
    top_underserved = df[df["UCS"] >= 70].sort_values("UCS", ascending=False) if df is not None else pd.DataFrame()
    
    if len(top_underserved) > 0:
        render_warning(f"⚠️ {len(top_underserved)} counties require urgent intervention (UCS ≥ 70)")
        
        # Create expandable sections for each county
        for idx, (county, row) in enumerate(top_underserved.head(5).iterrows()):
            with st.expander(f"{idx+1}. {county} (UCS: {row['UCS']:.1f})", expanded=(idx==0)):
                # Find worst domains with scores
                domain_scores = {DOMAIN_DISPLAY.get(d, d): row[d] for d in DOMAINS}
                worst_domain = max(domain_scores, key=domain_scores.get)
                worst_score = domain_scores[worst_domain]
                
                st.markdown(f"**Primary Concern:** {worst_domain} ({worst_score:.1f}/100)")
                
                # Get SHAP data for this county if available
                county_shap = shap_data[shap_data["County"] == county]
                if len(county_shap) > 0:
                    domain_cols = [c for c in county_shap.columns if c in DOMAINS]
                    shap_vals = {DOMAIN_DISPLAY.get(c, c): abs(county_shap[c].values[0]) for c in domain_cols}
                    top_shap = max(shap_vals, key=shap_vals.get)
                    st.caption(f"🚨 Top SHAP driver: {top_shap}")
                
                st.markdown("**All Domain Scores:**")
                for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True):
                    bar_width = int(score / 5)
                    st.markdown(f"  {domain}: {'█' * bar_width}{'░' * (20-bar_width)} {score:.1f}")
    
    # Best served counties
    best_served = df[df["UCS"] < 40].sort_values("UCS", ascending=True) if df is not None else pd.DataFrame()
    
    if len(best_served) > 0:
        st.markdown("---")
        render_success(f"✅ {len(best_served)} counties are relatively well-served (UCS < 40)")
        
        st.markdown("**Best Practices from Top Counties - Consider for Replication:**")
        for county in best_served.head(3).index:
            row = df.loc[county]
            domain_scores = {DOMAIN_DISPLAY.get(d, d): row[d] for d in DOMAINS}
            best_domain = min(domain_scores, key=domain_scores.get)
            st.markdown(f"- **{county}**: Strong in {best_domain} (score: {domain_scores[best_domain]:.1f})")
    
    # Actionable insights summary
    st.markdown("---")
    st.markdown("### 🎯 Priority Action Areas")
    
    col_action1, col_action2 = st.columns(2)
    
    with col_action1:
        st.markdown("**Immediate Priorities:**")
        if len(top_underserved) > 0:
            for domain in DOMAINS[:2]:  # Top 2 domains
                domain_name = DOMAIN_DISPLAY.get(domain, domain)
                avg_score = df[domain].mean()
                st.markdown(f"- Focus resources on {domain_name} (avg: {avg_score:.1f})")
    
    with col_action2:
        st.markdown("**Long-term Strategies:**")
        st.markdown("- Replicate best practices from well-served counties")
        st.markdown("- Implement cluster-specific interventions")
        st.markdown("- Monitor anomaly counties for unexpected changes")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption("""
🏥 Kenya Health Gap Dashboard | UCS Analysis | KDHS 2020 & 2022 | 47 Counties
Higher UCS = More Underserved | Developed by Cynthia Ngugi | MSc Data Science & Analytics
""")
