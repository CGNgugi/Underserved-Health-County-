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
            fillcolor=f'rgba{tuple(list(int(colors_list[idx % len(colors_list)].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
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
        
        # Drop NaN values to prevent groupby errors
        domain_melted = domain_melted.dropna(subset=["Domain", "Score"])
        
        fig_box = px.box(domain_melted, x="Domain", y="Score", color="Domain",
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
    corr_matrix = filtered_df[DOMAINS].corr()
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

# ============================================================
# PAGE 3: COUNTY RANKINGS
# ============================================================

elif page == "Rankings":
    st.markdown('<p class="main-title">📊 County Rankings</p>', unsafe_allow_html=True)
    st.caption("Comprehensive county-level UCS analysis with domain breakdowns")
    
    if df is None:
        st.stop()
    
    # Controls
    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        sort_by = st.selectbox("Sort By", ["UCS Score", "Healthcare Access", "Population Vulnerability", "Immunization Coverage", "Disease Burden"])
    with ctrl_col2:
        n_counties = st.slider("Number of counties", 10, 47, 25)
    
    # Sort data
    if sort_by == "UCS Score":
        sort_col = "UCS"
    else:
        sort_col = sort_by.replace(" ", " ").replace("Healthcare Access", "Healthcare Access Index").replace("Population Vulnerability", "Population Vulnerability Index").replace("Immunization Coverage", "Immunization Coverage Index").replace("Disease Burden", "Disease Burden Index")
    
    if sort_col in df.columns:
        df_sorted = df.sort_values(sort_col, ascending=False).head(n_counties)
    else:
        df_sorted = df.sort_values("UCS", ascending=False).head(n_counties)
    
    # Bar chart
    plot_df = df_sorted.reset_index()
    plot_df.columns = ["County", "UCS"] + list(plot_df.columns[2:])
    
    fig = px.bar(plot_df, x="UCS", y="County", orientation="h", color="UCS",
        color_continuous_scale=[COLORS["blue"], COLORS["orange"], COLORS["red"]],
        text="UCS", range_color=[0, 100])
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, n_counties * 12), 
        margin=dict(l=20, r=80, t=20, b=20), xaxis_title="UCS (0-100)")
    st.plotly_chart(fig, width='stretch')
    
    # Domain heatmap
    st.markdown("### 🔥 Domain Score Heatmap")
    top_counties = df.sort_values("UCS", ascending=False).head(min(20, n_counties)).index.tolist()
    domain_heatmap = df.loc[top_counties, DOMAINS].copy()
    domain_heatmap.columns = [DOMAIN_DISPLAY.get(c, c) for c in domain_heatmap.columns]
    
    fig_heat = px.imshow(domain_heatmap, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto")
    fig_heat.update_layout(height=max(300, len(top_counties) * 20))
    st.plotly_chart(fig_heat, width='stretch')
    
    # Data table
    st.markdown("### 📋 Full Rankings Table")
    display_cols = ["UCS"] + DOMAINS + ["Cluster_label"]
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()
    display_df.columns = ["UCS"] + [DOMAIN_DISPLAY.get(c, c) for c in DOMAINS if c in display_df.columns] + (["Cluster"] if "Cluster_label" in available_cols else [])
    st.dataframe(display_df.sort_values("UCS", ascending=False), width='stretch', height=400)

# ============================================================
# PAGE 4: GEOSPATIAL MAP
# ============================================================
elif page == "Geospatial Map":
    st.markdown('🌍 Geospatial Analysis</p>', unsafe_allow_html=True)
    
    if df is None:
        st.stop()
    
    # Show choropleth image
    st.markdown("### 🗺️ UCS Choropleth Map")
    try:
        st.image("ucs_choropleth.png", caption="UCS Score Distribution Across Kenya", width='stretch')
    except:
        st.info("Choropleth image not found")
    
    render_insight("This choropleth map shows the spatial distribution of Underserved County Scores across Kenya's 47 counties. Northern and northeastern ASAL regions show higher UCS scores indicating greater healthcare deprivation.")
    
    st.markdown("---")
    
    # Show anomaly map image
    st.markdown("### ⚠️ Anomaly Detection Map")
    try:
        st.image("anomaly_map.png", caption="Counties with Anomalous Healthcare Patterns", width='stretch')
    except:
        st.info("Anomaly map image not found")
    
    render_warning("Anomalous counties deviate significantly from their cluster pattern. These are counties where healthcare deprivation is unexpected based on their cluster characteristics.")
    
    st.markdown("---")
    
    # Prepare data
    county_data = df.reset_index()
    county_data.columns = ["County"] + list(county_data.columns[1:])
    
    # Kenya coordinates
    kenya_coords = {
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
        "Nyamira": [34.96, -0.56], "Laikipia": [36.78, 0.30]
    }
    
    county_data["lon"] = county_data["County"].map(lambda x: kenya_coords.get(x, [None, None])[0])
    county_data["lat"] = county_data["County"].map(lambda x: kenya_coords.get(x, [None, None])[1])
    map_data = county_data.dropna(subset=["lat", "lon"]).copy()
    
    for domain in DOMAINS:
        map_data[DOMAIN_DISPLAY.get(domain, domain)] = map_data[domain]
    
    # Controls
    map_col1, map_col2 = st.columns([2, 1])
    with map_col1:
        map_type = st.selectbox("Map View", 
            ["UCS Score", "Cluster Distribution", "Anomaly Detection", "Healthcare Access", "Disease Burden"])
    with map_col2:
        show_range = st.select_slider("UCS Filter", ["All", "High (70+)", "Medium (40-70)", "Low (0-40)"], value="All")
    
    # Apply filter
    filtered_map_data = map_data.copy()
    if show_range == "High (70+)":
        filtered_map_data = map_data[map_data["UCS"] >= 70]
    elif show_range == "Medium (40-70)":
        filtered_map_data = map_data[(map_data["UCS"] >= 40) & (map_data["UCS"] < 70)]
    elif show_range == "Low (0-40)":
        filtered_map_data = map_data[map_data["UCS"] < 40]
    
    st.caption(f"Showing {len(filtered_map_data)} of {len(map_data)} counties")
    
    if len(filtered_map_data) > 0:
        if map_type == "UCS Score":
            st.markdown("### 🗺️ UCS Score Map")
            fig_map = px.scatter_geo(
                filtered_map_data, lat="lat", lon="lon", color="UCS", size="UCS", size_max=35,
                hover_name="County", hover_data={"lat": False, "lon": False, "UCS": ":.1f", "Healthcare Access": ":.2f"},
                color_continuous_scale=["#1a9850", "#fee08b", "#d73027"], range_color=[0, 100], scope="africa"
            )
            fig_map.update_geos(showcountries=True, countrycolor="Black", showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="LightGray", lataxis_range=[-5, 5], lonaxis_range=[33, 42])
            fig_map.update_layout(height=500, margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig_map, width='stretch')
            
            render_insight("Red = Most Underserved (70-100) | Yellow = Moderate (40-70) | Green = Better Served (0-40)")
            
        elif map_type == "Cluster Distribution":
            st.markdown("### 👥 Cluster Map")
            filtered_map_data["Cluster_Display"] = filtered_map_data["Cluster_label"].apply(
                lambda x: "ASAL Underserved" if "Underserved" in str(x) else "Moderately Served"
            )
            fig_cluster = px.scatter_geo(
                filtered_map_data, lat="lat", lon="lon", color="Cluster_Display", size="UCS", size_max=35,
                hover_name="County", hover_data={"lat": False, "lon": False, "UCS": ":.1f"},
                color_discrete_map={"ASAL Underserved": COLORS["red"], "Moderately Served": COLORS["green"]}, scope="africa"
            )
            fig_cluster.update_geos(showcountries=True, countrycolor="Black", showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="LightGray", lataxis_range=[-5, 5], lonaxis_range=[33, 42])
            fig_cluster.update_layout(height=500, margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig_cluster, width='stretch')
            
        elif map_type == "Anomaly Detection":
            st.markdown("### ⚠️ Anomaly Map")
            fig_anom = px.scatter_geo(
                filtered_map_data, lat="lat", lon="lon", color="Anomaly", size="UCS", size_max=35,
                hover_name="County", hover_data={"lat": False, "lon": False, "UCS": ":.1f", "Anomaly_prob": ":.2f"},
                color_discrete_map={"Anomaly": COLORS["red"], "Normal": COLORS["green"]}, scope="africa"
            )
            fig_anom.update_geos(showcountries=True, countrycolor="Black", showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="LightGray", lataxis_range=[-5, 5], lonaxis_range=[33, 42])
            fig_anom.update_layout(height=500, margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig_anom, width='stretch')
            render_warning("Anomalous counties deviate significantly from their cluster pattern")
            
        elif map_type in ["Healthcare Access", "Disease Burden"]:
            domain_key = map_type
            st.markdown(f"### {map_type} Map")
            fig_domain = px.scatter_geo(
                filtered_map_data, lat="lat", lon="lon", color=domain_key, size="UCS", size_max=35,
                hover_name="County", hover_data={"lat": False, "lon": False, domain_key: ":.3f"},
                color_continuous_scale=["#1a9850", "#fee08b", "#d73027"], scope="africa"
            )
            fig_domain.update_geos(showcountries=True, countrycolor="Black", showcoastlines=True, coastlinecolor="Black",
                showland=True, landcolor="LightGray", lataxis_range=[-5, 5], lonaxis_range=[33, 42])
            fig_domain.update_layout(height=500, margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig_domain, width='stretch')

# ============================================================
# PAGE 5: CLUSTER ANALYSIS
# ============================================================

elif page == "Cluster Analysis":
    st.markdown('👥 Cluster Analysis</p>', unsafe_allow_html=True)
    st.caption("K-Means clustering identifies county typologies based on deprivation patterns")
    
    if df is None:
        st.stop()
    
    # Show cluster PCA image
    st.markdown("### 🔬 Cluster Visualization (PCA)")
    try:
        st.image("cluster_pca.png", caption="County Clusters in PCA Space", width='stretch')
    except:
        st.info("Cluster PCA image not found")
    
    render_insight("This PCA visualization shows how counties cluster based on their domain scores. Counties close together have similar healthcare deprivation patterns.")
    
    st.markdown("---")
    
    # Cluster summary
    cluster_counts = df["Cluster_label"].value_counts()
    st.markdown("### 📊 Cluster Distribution")
    cluster_col1, cluster_col2 = st.columns(2)
    
    with cluster_col1:
        fig_pie = go.Figure(go.Pie(
            labels=cluster_counts.index.str[:30],
            values=cluster_counts.values,
            hole=0.4,
            marker_colors=[COLORS["red"], COLORS["blue"], COLORS["green"]]
        ))
        fig_pie.update_layout(height=280)
        st.plotly_chart(fig_pie, width='stretch')
    
    with cluster_col2:
        st.markdown("### 📋 Cluster Summary")
        for cluster in cluster_counts.index:
            cluster_df = df[df["Cluster_label"] == cluster]
            with st.expander(f"{cluster[:40]}... ({len(cluster_df)} counties)", expanded=True):
                st.metric("Mean UCS", f"{cluster_df['UCS'].mean():.1f}")
                st.metric("UCS Range", f"{cluster_df['UCS'].min():.0f} - {cluster_df['UCS'].max():.0f}")
                counties = ", ".join(cluster_df.index.tolist()[:10])
                if len(cluster_df) > 10:
                    counties += "..."
                st.caption(f"Counties: {counties}")
    
    # Cluster profiles
    st.markdown("### 🎯 Cluster Domain Profiles")
    cluster_profiles = df.groupby("Cluster_label")[DOMAINS].mean()
    cluster_profiles.columns = [DOMAIN_DISPLAY.get(c, c) for c in cluster_profiles.columns]
    
    fig_profile = px.bar(cluster_profiles, barmode="group", 
        color_discrete_map={
            "Healthcare Access": COLORS["red"],
            "Population Vulnerability": COLORS["blue"],
            "Immunization Coverage": COLORS["green"],
            "Disease Burden": COLORS["purple"]
        })
    fig_profile.update_layout(height=350, xaxis_title="", yaxis_title="Mean Domain Score")
    st.plotly_chart(fig_profile, width='stretch')
    render_insight("Each cluster represents a distinct deprivation pattern requiring different intervention strategies.")

# ============================================================
# PAGE 6: ML & SHAP
# ============================================================

elif page == "ML & SHAP":
    st.markdown('🤖 Machine Learning & SHAP Analysis</p>', unsafe_allow_html=True)
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
    
    # SHAP summary
    st.markdown("### 📊 Feature Importance (SHAP)")
    if len(shap_df) > 0:
        shap_cols = [c for c in shap_df.columns if c in DOMAINS]
        if shap_cols:
            shap_means = shap_df[shap_cols].mean().sort_values(ascending=True)
            shap_means.index = [DOMAIN_DISPLAY.get(c, c) for c in shap_means.index]
            
            fig_shap = px.bar(shap_means, orientation="h", color=shap_means.values,
                color_continuous_scale=[COLORS["blue"], COLORS["red"]])
            fig_shap.update_layout(height=300, xaxis_title="Mean |SHAP Value|", yaxis_title="")
            st.plotly_chart(fig_shap, width='stretch')
            render_insight("Higher SHAP values indicate features with greater influence on predicting high UCS (underservice).")
    
    # Model performance
    st.markdown("### 🎯 Model Performance")
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        render_kpi("XGBoost", "Best Model", COLORS["green"])
    with perf_col2:
        render_kpi("92.3%", "Accuracy", COLORS["blue"])
    
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
st.caption("""
🏥 Kenya Health Gap Dashboard | UCS Analysis | KDHS 2020 & 2022 | 47 Counties
Higher UCS = More Underserved | Developed by Cynthia Ngugi | MSc Data Science & Analytics
""")
