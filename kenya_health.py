"""
kenya_health.py - Kenya Health Gap Dashboard
=============================================
Cynthia Ngugi | Reg. 138725 | MSc Data Science | Strathmore University | March 2026

All figures are sourced directly from the submitted dissertation:- UCS rankings, all 47 counties- Domain inter-correlation matrix- Domain importance (r with UCS)- PCA transparency report- K-Means validation metrics- Cluster mean domain profiles- Classification model performance- XGBoost feature importance- SHAP force plot, Wajir county- Isolation Forest anomaly scores

Run: streamlit run kenya_health.py
"""

import os
import warnings
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
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

st.set_page_config(
    page_title="Kenya Health Equity Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

C = {
    "navy":  "#1E3888",
    "blue":  "#2F80ED",
    "gold":  "#C69D1A",
    "red":   "#C8321E",
    "green": "#1A6B2A",
    "orange":"#E08B1A",
    "muted": "#4F4F4F",
    "light": "#EAF0FF",
    "cl1":   "#C8321E",
    "cl2":   "#2F80ED",
    "anom":  "#E08B1A",
}

# ── EXACT DATA FROM TABLE 4.3 (dissertation pp.44-45) ────────────────────────
UCS_DATA = {
    "Wajir":           {"UCS":100.00,"Cluster":"Cluster 1","Anomaly":True, "IF_score":-0.047},
    "Turkana":         {"UCS": 97.53,"Cluster":"Cluster 1","Anomaly":True, "IF_score":-0.006},
    "Tana River":      {"UCS": 95.38,"Cluster":"Cluster 1","Anomaly":False,"IF_score":None},
    "Marsabit":        {"UCS": 94.73,"Cluster":"Cluster 1","Anomaly":True, "IF_score":-0.021},
    "Samburu":         {"UCS": 90.68,"Cluster":"Cluster 1","Anomaly":False,"IF_score":None},
    "Kilifi":          {"UCS": 89.23,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Mandera":         {"UCS": 86.44,"Cluster":"Cluster 1","Anomaly":True, "IF_score":-0.110},
    "Homa Bay":        {"UCS": 78.78,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "West Pokot":      {"UCS": 77.88,"Cluster":"Cluster 1","Anomaly":False,"IF_score":None},
    "Kitui":           {"UCS": 75.07,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Meru":            {"UCS": 72.52,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Vihiga":          {"UCS": 67.26,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Lamu":            {"UCS": 65.95,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Isiolo":          {"UCS": 64.06,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Tharaka-Nithi":   {"UCS": 63.88,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Migori":          {"UCS": 63.57,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Baringo":         {"UCS": 63.50,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Bungoma":         {"UCS": 62.55,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Siaya":           {"UCS": 55.47,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Garissa":         {"UCS": 54.46,"Cluster":"Cluster 1","Anomaly":True, "IF_score":-0.096},
    "Busia":           {"UCS": 53.87,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Nyamira":         {"UCS": 51.45,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kakamega":        {"UCS": 49.68,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Narok":           {"UCS": 49.33,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Murang'a":        {"UCS": 48.98,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kwale":           {"UCS": 47.73,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Nakuru":          {"UCS": 46.54,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Trans-Nzoia":     {"UCS": 46.31,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Makueni":         {"UCS": 45.24,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Bomet":           {"UCS": 45.08,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Taita Taveta":    {"UCS": 44.41,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Nyandarua":       {"UCS": 44.39,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Elgeyo Marakwet": {"UCS": 43.02,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kisii":           {"UCS": 41.97,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Nandi":           {"UCS": 41.65,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kisumu":          {"UCS": 37.62,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Machakos":        {"UCS": 36.77,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Laikipia":        {"UCS": 36.18,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Uasin Gishu":     {"UCS": 33.18,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kiambu":          {"UCS": 32.29,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kericho":         {"UCS": 31.77,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kirinyaga":       {"UCS": 28.89,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Embu":            {"UCS": 27.46,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Nyeri":           {"UCS": 25.93,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Kajiado":         {"UCS": 24.75,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Mombasa":         {"UCS": 19.94,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
    "Nairobi":         {"UCS":  0.00,"Cluster":"Cluster 2","Anomaly":False,"IF_score":None},
}

COORDS = {
    "Mombasa":[-4.04,39.68],"Kwale":[-4.54,39.45],"Kilifi":[-3.53,39.60],
    "Tana River":[-1.79,39.97],"Lamu":[-2.22,40.10],"Taita Taveta":[-3.39,38.44],
    "Garissa":[-0.45,40.12],"Wajir":[1.75,40.06],"Mandera":[3.94,40.65],
    "Marsabit":[2.54,37.98],"Isiolo":[0.35,38.49],"Meru":[0.16,37.96],
    "Tharaka-Nithi":[-0.21,37.67],"Embu":[-0.53,37.45],"Kitui":[-1.37,38.01],
    "Machakos":[-1.52,37.26],"Makueni":[-2.24,37.56],"Nyandarua":[-0.65,36.43],
    "Nyeri":[-0.42,36.96],"Kirinyaga":[-0.58,37.28],"Murang'a":[-0.79,36.96],
    "Kiambu":[-1.17,36.84],"Nairobi":[-1.29,36.82],"Kajiado":[-1.88,36.78],
    "Kericho":[-0.37,35.28],"Bomet":[-0.52,35.15],"Nakuru":[-0.29,36.07],
    "Narok":[-1.08,35.86],"Baringo":[1.17,35.97],"Elgeyo Marakwet":[0.75,35.50],
    "West Pokot":[1.23,35.03],"Samburu":[1.19,36.75],"Trans-Nzoia":[1.03,35.00],
    "Uasin Gishu":[0.54,35.29],"Nandi":[0.20,35.00],"Kakamega":[0.28,34.75],
    "Vihiga":[0.04,34.60],"Bungoma":[0.53,34.56],"Busia":[0.44,34.25],
    "Siaya":[-0.06,34.18],"Kisumu":[-0.02,34.76],"Homa Bay":[-0.53,34.45],
    "Migori":[-1.15,34.38],"Kisii":[-0.69,34.76],"Nyamira":[-0.56,34.96],
    "Laikipia":[0.30,36.78],"Turkana":[3.46,35.54],
}

@st.cache_data
def build_df():
    rows = []
    for county, d in UCS_DATA.items():
        lat, lon = COORDS.get(county, [None, None])
        rows.append({"County":county,"UCS":d["UCS"],"Cluster":d["Cluster"],
                     "Anomaly":"Anomaly" if d["Anomaly"] else "Normal",
                     "IF_score":d["IF_score"],"lat":lat,"lon":lon})
    df = pd.DataFrame(rows).set_index("County")
    df["Rank"] = df["UCS"].rank(ascending=False).astype(int)
    return df

df = build_df()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif;color:#222222;background:#FFFFFF;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0!important;padding-bottom:.4rem!important;padding-left:.7rem!important;padding-right:.7rem!important;max-width:100%!important;}
div[data-testid="stVerticalBlock"]>div{gap:.2rem!important;}
.element-container{margin-bottom:.1rem!important;}
.stPlotlyChart{margin-bottom:0!important;}
hr{margin:.3rem 0!important;border-color:#E0E7FF!important;}

/* ── HEADER ─────────────────────────────────────────────── */
.moh-header{
  background:#FFFFFF;border-bottom:2px solid #2F80ED;
  padding:7px 14px 5px 14px;margin:0 -.7rem .2rem -.7rem;
  display:flex;align-items:center;justify-content:space-between;
}
.moh-logo{display:flex;gap:6px;align-items:center;min-width:130px;}
.moh-badge{
  height:26px;min-width:40px;border:1px solid #BFD0F5;
  color:#2F80ED;display:inline-flex;align-items:center;justify-content:center;
  font-size:.62rem;font-weight:800;background:#EEF4FF;padding:0 6px;
}
.moh-title{flex:1;text-align:center;}
.moh-title .ministry{color:#1B4DB5;font-size:1.28rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;line-height:1.1;}
.moh-title .subtitle{color:#4A76D0;font-size:.72rem;}
.moh-meta{color:#C69D1A;font-size:.7rem;font-weight:700;min-width:130px;text-align:right;}

/* ── NAV TABS ───────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"]:first-of-type{
  background:#1B3F8C!important;gap:0!important;
  padding:0 10px!important;margin:0 -.7rem 0 -.7rem!important;
  border-bottom:3px solid #C69D1A;
}
div[data-testid="stHorizontalBlock"]:first-of-type>div{padding:0!important;flex:1!important;}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton button{
  border-radius:0!important;border:none!important;
  border-bottom:3px solid transparent!important;margin-bottom:-3px!important;
  background:transparent!important;color:#A8C0EE!important;
  font-size:.65rem!important;font-weight:600!important;
  padding:9px 2px!important;letter-spacing:.05em!important;
  text-transform:uppercase!important;box-shadow:none!important;width:100%!important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton button:hover{
  color:#FFFFFF!important;background:rgba(255,255,255,.08)!important;
  border-bottom-color:rgba(198,157,26,.5)!important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton button[kind="primary"]{
  color:#FFFFFF!important;background:rgba(255,255,255,.14)!important;
  border-bottom:3px solid #C69D1A!important;
}

/* ── KPI TILES ──────────────────────────────────────────── */
.kpi{
  background:#FFFFFF;border-radius:3px;padding:8px 10px;
  border:1px solid #D0DCF5;border-top:3px solid var(--kc,#2F80ED);
  text-align:center;margin-bottom:3px;
  box-shadow:0 1px 3px rgba(43,80,180,.06);
}
.kpi .v{font-size:1.15rem;font-weight:800;color:#E07B12;line-height:1.05;}
.kpi .l{font-size:.56rem;color:#3A5FC0;letter-spacing:.01em;margin-top:2px;}

/* ── METRICS ────────────────────────────────────────────── */
div[data-testid="stMetricValue"]{font-size:1.05rem!important;font-weight:800!important;color:#E07B12!important;}
div[data-testid="stMetricLabel"]{font-size:.58rem!important;color:#3A5FC0!important;}

/* ── ALERT BOXES ────────────────────────────────────────── */
.box-info{
  background:#F0F5FF;border:1px solid #C5D5F5;border-left:4px solid #2F80ED;
  padding:8px 12px;border-radius:3px;font-size:.79rem;color:#1A2E5C;margin:5px 0;line-height:1.5;
}
.box-warn{
  background:#FFFBEC;border:1px solid #E8D890;border-left:4px solid #C69D1A;
  padding:8px 12px;border-radius:3px;font-size:.79rem;color:#4A3800;margin:5px 0;line-height:1.5;
}
.box-ok{
  background:#F0FBF4;border:1px solid #A8D8B8;border-left:4px solid #1A8040;
  padding:8px 12px;border-radius:3px;font-size:.79rem;color:#0E3A20;margin:5px 0;line-height:1.5;
}
.box-err{
  background:#FFF2F2;border:1px solid #F5B8B8;border-left:4px solid #C8321E;
  padding:8px 12px;border-radius:3px;font-size:.79rem;color:#4A0E0E;margin:5px 0;line-height:1.5;
}

/* ── HEADINGS ───────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6{color:#1B3F8C!important;}
h5{font-size:.84rem!important;margin:.2rem 0 .12rem 0!important;font-weight:700!important;}

/* ── TABLES ─────────────────────────────────────────────── */
.stDataFrame{border:1px solid #D0DCF5!important;}
.stDataFrame thead tr th{background:#EEF3FF!important;color:#1B3F8C!important;font-size:.68rem!important;}
.stDataFrame tbody tr td{font-size:.72rem!important;color:#222222!important;}

/* ── SELECTBOX ──────────────────────────────────────────── */
div[data-testid="stSelectbox"] label{font-size:.68rem!important;color:#444!important;}
div[data-testid="stSelectbox"]{margin-bottom:3px!important;}

/* ── PROGRESS ───────────────────────────────────────────── */
.stProgress>div>div>div>div{background-color:#E07B12!important;}
.stProgress>div>div>div{height:6px!important;background:#EEF3FF!important;}

/* ── BUTTONS ────────────────────────────────────────────── */
.stButton button{
  background:#1B3F8C!important;color:white!important;
  border:none!important;border-radius:3px!important;
  font-size:.78rem!important;font-weight:600!important;
}
.stButton button:hover{background:#2F80ED!important;}

/* ── FILE UPLOADER ──────────────────────────────────────── */
[data-testid="stFileUploader"]{
  border:1px dashed #2F80ED!important;border-radius:4px!important;background:#F8FAFF!important;
}

/* ── SIDEBAR HIDDEN ─────────────────────────────────────── */
section[data-testid="stSidebar"]{display:none!important;}

small,.stCaption{font-size:.65rem!important;color:#555!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="moh-header">
  <div class="moh-logo">
    <span class="moh-badge">WHO</span>
    <span class="moh-badge">CDC</span>
    <span class="moh-badge">MOH</span>
  </div>
  <div class="moh-title">
    <div class="ministry">Ministry of Health - Republic of Kenya</div>
    <div class="subtitle">Kenya Health Equity Monitor &nbsp;·&nbsp; Underserved County Score (UCS)</div>
  </div>
  <div class="moh-meta">KDHS 2020 / 2022</div>
</div>
""", unsafe_allow_html=True)

PAGES = ["Overview","Map","UCS Rankings","Clustering","Anomaly Detection",
         "Classification & SHAP","KDHS Predictor"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

_cols = st.columns(len(PAGES))
for col, label in zip(_cols, PAGES):
    with col:
        if st.button(label, key=f"nav_{label}",
                     type="primary" if st.session_state.page==label else "secondary",
                     use_container_width=True):
            st.session_state.page = label
            st.rerun()

page = st.session_state.page
_ai  = PAGES.index(page)
bar  = "".join([f'<div style="flex:1;height:3px;background:{"#C69D1A" if i==_ai else "transparent"}"></div>'
                for i in range(len(PAGES))])
st.markdown(f'<div style="display:flex;margin:0 -.7rem 2px -.7rem">{bar}</div>',
            unsafe_allow_html=True)

def kpi(val, label, color=C["blue"]):
    st.markdown(f'<div class="kpi" style="--kc:{color}"><div class="v">{val}</div>'
                f'<div class="l">{label}</div></div>', unsafe_allow_html=True)

def box(text, kind="info"):
    st.markdown(f'<div class="box-{kind}">{text}</div>', unsafe_allow_html=True)

def cl_color(c): return C["cl1"] if c=="Cluster 1" else C["cl2"]

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("##### Kenya Healthcare Access Inequality Dashboard")
    st.caption("UCS 0–100 · Higher = More Underserved · KDHS 2020 & 2022 · 47 Counties")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: kpi("47","Counties Scored",C["blue"])
    with k2: kpi(f"{df['UCS'].mean():.1f}","Mean UCS",C["orange"])
    with k3: kpi("100.00","Max UCS (Wajir)",C["cl1"])
    with k4: kpi("0.00","Min UCS (Nairobi)",C["cl2"])
    with k5: kpi("8","Cluster 1 - ASAL",C["cl1"])
    with k6: kpi("5","Anomalous Counties",C["anom"])

    st.markdown("<hr>",unsafe_allow_html=True)
    left, right = st.columns([1.1,1])

    with left:
        b1,b2 = st.columns(2)
        with b1:
            st.markdown("##### Top 10 Most Underserved")
            t10 = df.sort_values("UCS",ascending=False).head(10).reset_index()
            fig = go.Figure(go.Bar(x=t10["UCS"],y=t10["County"],orientation="h",
                marker_color=[cl_color(c) for c in t10["Cluster"]],
                text=t10["UCS"].map(lambda x:f"{x:.2f}"),
                textposition="outside",textfont_size=8))
            fig.update_layout(yaxis_autorange="reversed",height=260,
                margin=dict(l=5,r=40,t=5,b=5),xaxis_range=[0,112],
                yaxis_tickfont_size=8,
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        with b2:
            st.markdown("##### Top 10 Best Served")
            b10 = df.sort_values("UCS").head(10).reset_index()
            fig2 = go.Figure(go.Bar(x=b10["UCS"],y=b10["County"],orientation="h",
                marker_color=C["cl2"],
                text=b10["UCS"].map(lambda x:f"{x:.2f}"),
                textposition="outside",textfont_size=8))
            fig2.update_layout(yaxis_autorange="reversed",height=260,
                margin=dict(l=5,r=40,t=5,b=5),
                yaxis_tickfont_size=8,
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2,use_container_width=True)

        st.markdown("##### Domain Importance - r with UCS")
        di = pd.DataFrame({
            "Domain":   ["Healthcare Access (HAI)","Disease Burden (DBI)",
                         "Population Vulnerability (PVI)","Immunisation Coverage (ICI)"],
            "r":        [0.808,0.758,0.753,0.068],
            "CV_wt":    [0.3717,0.2258,0.3997,0.0028],
        })
        fig_d = go.Figure(go.Bar(
            x=di["r"],y=di["Domain"],orientation="h",
            marker_color=[C["cl1"],C["orange"],C["blue"],C["muted"]],
            text=[f"r={v:.3f}  CV wt={w:.4f}" for v,w in zip(di["r"],di["CV_wt"])],
            textposition="outside",textfont_size=8))
        fig_d.update_layout(height=160,margin=dict(l=5,r=5,t=5,b=5),
            xaxis_range=[0,0.97],xaxis_title="Pearson r",yaxis_title="",
            yaxis_tickfont_size=8,
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_d,use_container_width=True)
        st.caption("ICI r=0.068 - operationally independent of all other domains. "
                   "ICI-HAI r=0.073, ICI-PVI r=0.043, ICI-DBI r=-0.044. "
                   "ICI explains <1% of UCS variance (r2=0.005).")

    with right:
        st.markdown("##### K-Means - k=2 Optimal")
        fig_c = go.Figure(go.Pie(
            labels=["Cluster 1 - Structurally Underserved (n=8)",
                    "Cluster 2 - Moderately Served (n=39)"],
            values=[8,39],hole=0.52,
            marker_colors=[C["cl1"],C["cl2"]],
            textinfo="percent",textfont_size=9))
        fig_c.update_layout(height=170,margin=dict(l=5,r=5,t=5,b=5),
            showlegend=True,
            legend=dict(font_size=8,x=0.5,y=-0.15,orientation="h",xanchor="center"),
            annotations=[dict(text="k=2<br>Sil=0.459",x=0.5,y=0.5,font_size=8,showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c,use_container_width=True)
        st.caption("Silhouette=0.4595, Davies-Bouldin=0.612, Calinski-Harabasz=41.3 - "
                   "all favour k=2. Silhouette drops 0.46→0.16 at k=3.")

        st.markdown("##### UCS Map - 47 Counties")
        mf = st.selectbox("Filter",["All Counties",
            "Cluster 1 - Structurally Underserved","Cluster 2 - Moderately Served"],
            key="ov_f",label_visibility="collapsed")
        mdf = df.copy()
        if "Cluster 1" in mf: mdf = mdf[mdf["Cluster"]=="Cluster 1"]
        elif "Cluster 2" in mf: mdf = mdf[mdf["Cluster"]=="Cluster 2"]

        if FOLIUM_OK:
            m = folium.Map(location=[0.5,37.5],zoom_start=5.6,tiles="CartoDB positron")
            for county, row in mdf.iterrows():
                if row["lat"] is None: continue
                col = "red" if row["Cluster"]=="Cluster 1" else "blue"
                popup = (f"<b>{county}</b><br>UCS: {row['UCS']:.2f}<br>"
                         f"Rank: {row['Rank']} / 47<br>Cluster: {row['Cluster']}<br>"
                         f"Anomaly: {'Yes - IF ' + str(row['IF_score']) if row['Anomaly']=='Anomaly' else 'No'}")
                is_a = row["Anomaly"]=="Anomaly"
                folium.CircleMarker(
                    location=[row["lat"],row["lon"]],
                    radius=5+row["UCS"]/20,
                    popup=folium.Popup(popup,max_width=220),
                    color="orange" if is_a else col,
                    fill=True,fillColor=col,fillOpacity=0.75,
                    weight=3 if is_a else 1.5,
                    tooltip=f"{'[ANOMALY] ' if is_a else ''}{county}: {row['UCS']:.2f}",
                ).add_to(m)
            st_html(m._repr_html_(),height=290,scrolling=False)
            st.caption("Red=Cluster 1 (ASAL, n=8) · Blue=Cluster 2 (n=39) · "
                       "Orange border=Isolation Forest anomaly")
        else:
            mdf_p = mdf.reset_index()
            mdf_p["ClusterLabel"] = mdf_p["Cluster"]
            fig_sc = px.scatter(mdf_p,x="lon",y="lat",color="ClusterLabel",
                color_discrete_map={"Cluster 1":C["cl1"],"Cluster 2":C["cl2"]},
                size="UCS",hover_name="County",height=290)
            fig_sc.update_layout(margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_sc,use_container_width=True)

    st.markdown("<hr>",unsafe_allow_html=True)
    box("Five anomaly-flagged counties (Wajir, Turkana, Marsabit, Mandera, Garissa) are all "
        "Cluster 1 ASAL counties. Garissa (rank 20, UCS 54.46) is the most instructive: "
        "moderate composite score yet the 2nd most anomalous county (IF score −0.096), "
        "flagged for disproportionately high HAI relative to PVI and DBI.", "info")

# ── MAP ───────────────────────────────────────────────────────────────────────
elif page == "Map":
    st.markdown("##### Interactive Map - 47 Counties")
    st.caption("Red = Cluster 1 Structurally Underserved · Blue = Cluster 2 Moderately Served · "
               "Orange border = Isolation Forest anomaly")
    c1,c2 = st.columns([2,1])
    with c1:
        mf = st.selectbox("Filter",["All Counties",
            "Cluster 1 - Structurally Underserved (n=8)",
            "Cluster 2 - Moderately Served (n=39)",
            "Anomalous Counties Only (n=5)"])
    mdf = df.copy()
    if "Cluster 1" in mf: mdf = mdf[mdf["Cluster"]=="Cluster 1"]
    elif "Cluster 2" in mf: mdf = mdf[mdf["Cluster"]=="Cluster 2"]
    elif "Anomalous" in mf: mdf = mdf[mdf["Anomaly"]=="Anomaly"]
    st.caption(f"Showing {len(mdf)} / 47 counties")

    if FOLIUM_OK:
        m = folium.Map(location=[0.5,37.5],zoom_start=6,tiles="CartoDB positron")
        for county, row in mdf.iterrows():
            if row["lat"] is None: continue
            col = "red" if row["Cluster"]=="Cluster 1" else "blue"
            popup = (f"<div style='width:200px'><b>{county}</b><br>"
                     f"UCS: <b>{row['UCS']:.2f}</b><br>Rank: {row['Rank']}/47<br>"
                     f"Cluster: {row['Cluster']}<br>"
                     f"{'Anomaly: Yes - IF score '+str(row['IF_score']) if row['Anomaly']=='Anomaly' else 'Anomaly: No'}</div>")
            is_a = row["Anomaly"]=="Anomaly"
            folium.CircleMarker(
                location=[row["lat"],row["lon"]],radius=6+row["UCS"]/16,
                popup=folium.Popup(popup,max_width=230),
                color="orange" if is_a else col,fill=True,fillColor=col,
                fillOpacity=0.75,weight=3 if is_a else 1.5,
                tooltip=f"{'[ANOMALY] ' if is_a else ''}{county}: UCS {row['UCS']:.2f}",
            ).add_to(m)
        st_html(m._repr_html_(),height=500,scrolling=False)

    ca,cb = st.columns(2)
    with ca:
        st.markdown("""
| Marker | Cluster | n | UCS Range |
|---|---|---|---|
| Red | Cluster 1 - Structurally Underserved | 8 | 54.46–100.00 |
| Blue | Cluster 2 - Moderately Served | 39 | 0.00–89.23 |
| Orange border | Isolation Forest anomaly | 5 | - |
""")
    with cb:
        box("Cluster 1: Wajir · Turkana · Tana River · Marsabit · "
            "Samburu · Mandera · West Pokot · Garissa. All ASAL. "
            "HDBSCAN robustness check confirms this partition.", "info")

# ── UCS RANKINGS ──────────────────────────────────────────────────────────────
elif page == "UCS Rankings":
    st.markdown("##### UCS Rankings - All 47 Counties")
    st.caption("Higher UCS = more underserved. Red = Cluster 1 ASAL. Blue = Cluster 2. Triangle = anomaly.")

    sdf = df.sort_values("UCS",ascending=False).reset_index()
    sdf["Label"] = sdf.apply(lambda r: f"{r['UCS']:.2f}"+(" ▲" if r["Anomaly"]=="Anomaly" else ""),axis=1)
    fig = go.Figure(go.Bar(
        x=sdf["UCS"],y=sdf["County"],orientation="h",
        marker_color=[cl_color(c) for c in sdf["Cluster"]],
        text=sdf["Label"],textposition="outside",textfont_size=7.5))
    fig.update_layout(height=1100,margin=dict(l=5,r=60,t=5,b=5),
        xaxis_range=[0,115],xaxis_title="UCS Score",yaxis_title="",
        yaxis_autorange="reversed",yaxis_tickfont_size=8,
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")

    cl, cr = st.columns([2,1])
    with cl: st.plotly_chart(fig,use_container_width=True)
    with cr:
        display = sdf[["County","UCS","Rank","Cluster","Anomaly","IF_score"]].copy()
        display["UCS"] = display["UCS"].map(lambda x:f"{x:.2f}")
        st.dataframe(display.set_index("Rank"),use_container_width=True,height=900)

# ── CLUSTERING ────────────────────────────────────────────────────────────────
elif page == "Clustering":
    st.markdown("##### K-Means Clustering - k=2 Optimal")
    st.caption("All three internal validation metrics converge on k=2. "
               "Silhouette drops 0.4595→0.1554 at k=3 - a decisive break.")

    c1,c2 = st.columns(2)
    with c1:
        # Exactvalues
        k_vals=[2,3,4,5,6,7,8,9]
        sil=[0.4595,0.1554,0.1669,0.1845,0.2031,0.1592,0.1559,0.1541]
        db= [0.612, 1.201, 1.188, 1.143, 1.097, 1.134, 1.152, 1.163]
        ch= [41.3,  28.6,  26.1,  24.7,  23.4,  22.1,  21.3,  20.8]

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=k_vals,y=sil,name="Silhouette (↑ better)",
            mode="lines+markers",line=dict(color=C["blue"],width=2.5),marker=dict(size=7)))
        fig_v.add_vline(x=2,line_dash="dash",line_color=C["cl1"],
            annotation_text="k=2 optimal",annotation_font_size=9)
        fig_v.update_layout(height=240,margin=dict(l=40,r=20,t=20,b=30),
            xaxis_title="k",yaxis_title="Silhouette Score",
            xaxis=dict(tickvals=k_vals,tickfont_size=9),yaxis_range=[0,0.55],
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_v,use_container_width=True)

        metric_df = pd.DataFrame({"k":k_vals,"Silhouette":sil,
            "Davies-Bouldin":db,"Calinski-Harabasz":ch}).set_index("k")
        st.dataframe(metric_df,use_container_width=True,height=240)
        st.caption("k=2: Silhouette=0.4595 (max) · Davies-Bouldin=0.612 (min) · "
                   "Calinski-Harabasz=41.3 (max).")

    with c2:
        box("<b>Cluster 1 - Structurally Underserved (n=8)</b><br>"
            "Wajir · Turkana · Tana River · Marsabit · Samburu · Mandera · West Pokot · Garissa<br>"
            "All ASAL counties. UCS range: 54.46–100.00.<br>"
            "Profile: High HAI · High PVI · High DBI · Variable ICI.","err")
        box("<b>Cluster 2 - Moderately Served (n=39)</b><br>"
            "All remaining counties. Low–Moderate HAI, PVI, DBI · Moderate ICI.<br>"
            "Internal range: Kilifi (89.23) to Nairobi (0.00).","ok")
        box("<b>HDBSCAN Robustness Check</b><br>"
            "Applied to the same standardised sub-domain feature space "
            "(min_cluster_size=5, min_samples=3). The 8 counties assigned to Cluster 1 "
            "by K-Means were grouped identically by HDBSCAN. "
            "The ASAL partition reflects genuine high-density structure - "
            "not a K-Means geometry artefact.","info")

        c1_df = df[df["Cluster"]=="Cluster 1"].sort_values("UCS",ascending=False).reset_index()
        fig_c1 = go.Figure(go.Bar(
            x=c1_df["County"],y=c1_df["UCS"],
            marker_color=[C["anom"] if a=="Anomaly" else C["cl1"] for a in c1_df["Anomaly"]],
            text=c1_df["UCS"].map(lambda x:f"{x:.2f}"),
            textposition="outside",textfont_size=8))
        fig_c1.update_layout(height=190,margin=dict(l=5,r=5,t=5,b=40),
            xaxis_tickfont_size=8,yaxis_range=[0,115],xaxis_title="",yaxis_title="UCS",
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c1,use_container_width=True)
        st.caption("Orange = anomaly flagged by Isolation Forest")

# ── ANOMALY DETECTION ─────────────────────────────────────────────────────────
elif page == "Anomaly Detection":
    st.markdown("##### Isolation Forest Anomaly Detection")
    st.caption("Five counties flagged. Negative IF score = more anomalous. Garissa (rank 20) is the key case.")


    anom_df = pd.DataFrame({
        "County":  ["Wajir","Turkana","Marsabit","Mandera","Garissa"],
        "UCS":     [100.00,  97.53,    94.73,     86.44,    54.46],
        "Rank":    [1,       2,         4,          7,        20],
        "Cluster": ["C1","C1","C1","C1","C1"],
        "IF_score":[-0.047, -0.006,   -0.021,    -0.110,   -0.096],
        "Key observation":[
            "HAI dominant. SHAP +2.34. Net SHAP = +5.54, highest of any county.",
            "High HAI and DBI. Moderate PVI relative to severe access deficit.",
            "Extreme PVI - poverty and WASH are the primary lever.",
            "Most extreme IF score (−0.110). Distinct domain imbalance from Wajir/Turkana.",
            "Rank 20 - moderate composite, 2nd most anomalous. High HAI vs PVI and DBI.",
        ],
    })

    ca,cb = st.columns([1.2,1])
    with ca:
        st.dataframe(anom_df[["County","UCS","Rank","IF_score"]].set_index("County"),
                     use_container_width=True,height=195)
        st.caption("Mandera (−0.110) most extreme. Garissa (rank 20) most instructive.")

        normal = df[df["Anomaly"]=="Normal"].copy()
        anom_p = df[df["Anomaly"]=="Anomaly"].copy()
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=normal["UCS"],y=[0]*len(normal),
            mode="markers",name="Normal",
            marker=dict(color=C["cl2"],size=6,opacity=0.5),
            text=normal.index,hovertemplate="%{text}: UCS=%{x:.2f}<extra></extra>"))
        fig_s.add_trace(go.Scatter(x=anom_p["UCS"],y=anom_p["IF_score"].fillna(0),
            mode="markers+text",name="Anomaly",
            marker=dict(color=C["anom"],size=12,symbol="diamond",
                        line=dict(color=C["cl1"],width=2)),
            text=anom_p.index,textposition="top center",textfont_size=8,
            hovertemplate="%{text}: UCS=%{x:.2f} IF=%{y:.3f}<extra></extra>"))
        fig_s.update_layout(height=220,margin=dict(l=40,r=10,t=10,b=30),
            xaxis_title="UCS Score",yaxis_title="IF Score",
            legend=dict(font_size=8),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s,use_container_width=True)

    with cb:
        for _,row in anom_df.iterrows():
            box(f"<b>{row['County']}</b> - UCS {row['UCS']:.2f} | Rank {row['Rank']} | "
                f"IF score {row['IF_score']}<br>{row['Key observation']}","warn")
        box("<b>Why Garissa matters:</b><br>"
            "UCS 54.46 (rank 20) - any top-10 targeting cutoff excludes it. "
            "Yet Isolation Forest assigns it the 2nd most extreme score (−0.096) "
            "because its domain combination - disproportionately high HAI vs PVI and DBI - "
            "is structurally atypical even within Cluster 2. "
            "Anomaly detection catches what ranking alone cannot.","err")

# ── CLASSIFICATION & SHAP ─────────────────────────────────────────────────────
elif page == "Classification & SHAP":
    st.markdown("##### Supervised Classification and SHAP Interpretation")
    st.caption("XGBoost CV AUC-ROC = 0.84 meets pre-specified criterion ≥ 0.80. ")

    r1l,r1r = st.columns(2)

    with r1l:
        st.markdown("##### Model Performance")
        mdf2 = pd.DataFrame({
            "Model":    ["XGBoost","Gradient Boosting","Random Forest","Logistic Regression"],
            "CV AUC":   [0.84, 0.82, 0.81, 0.75],
            "F1":       [0.80, 0.78, 0.77, 0.70],
        })
        fig_m = go.Figure(go.Bar(
            x=mdf2["CV AUC"],y=mdf2["Model"],orientation="h",
            marker_color=[C["navy"],C["blue"],C["blue"],C["muted"]],
            text=mdf2["CV AUC"].map(lambda x:f"{x:.2f}"),
            textposition="outside",textfont_size=9))
        fig_m.add_vline(x=0.80,line_dash="dash",line_color=C["cl1"],
            annotation_text="criterion ≥ 0.80",annotation_font_size=8)
        fig_m.update_layout(height=190,margin=dict(l=5,r=40,t=5,b=5),
            xaxis_range=[0.60,0.95],xaxis_title="CV AUC-ROC",yaxis_title="",
            yaxis_tickfont_size=9,
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_m,use_container_width=True)
        box("XGBoost 0.84 meets the criterion. Gap vs Logistic Regression (0.84 vs 0.75) "
            "confirms non-linear domain–cluster relationships, justifying ensemble methods "
            ".","ok")

    with r1r:
        st.markdown("##### XGBoost Feature Importance")
        fi = pd.DataFrame({
            "Domain": ["Healthcare Access (HAI)","Disease Burden (DBI)",
                       "Population Vulnerability (PVI)","Immunisation Coverage (ICI)"],
            "Gain":   [42.3, 28.7, 25.1, 3.9],
        })
        fig_f = go.Figure(go.Bar(
            x=fi["Gain"],y=fi["Domain"],orientation="h",
            marker_color=[C["cl1"],C["orange"],C["blue"],C["muted"]],
            text=fi["Gain"].map(lambda x:f"{x:.1f}%"),
            textposition="outside",textfont_size=9))
        fig_f.update_layout(height=190,margin=dict(l=5,r=50,t=5,b=5),
            xaxis_range=[0,55],xaxis_title="Gain (%)",yaxis_title="",
            yaxis_tickfont_size=9,
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_f,use_container_width=True)
        box("HAI dominates (42.3%). ICI = 3.9% - consistent with r²=0.005. "
            "Both methods confirm the same hierarchy: HAI > DBI > PVI >> ICI.","info")

    st.markdown("---")
    st.markdown("##### SHAP Force Plot: Wajir County (UCS = 100.00) -")
    shap_df_w = pd.DataFrame({
        "Domain":    ["Healthcare Access (HAI)","Disease Burden (DBI)",
                      "Population Vulnerability (PVI)","Immunisation Coverage (ICI)"],
        "SHAP":      [2.34, 1.87, 1.45, -0.12],
        "Direction": ["↑ Cluster 1","↑ Cluster 1","↑ Cluster 1","↓ Cluster 2"],
        "Priority":  ["Highest","High","Moderate","Maintenance"],
    })

    sl,sr = st.columns([1.2,1])
    with sl:
        fig_sh = go.Figure(go.Bar(
            x=shap_df_w["SHAP"],y=shap_df_w["Domain"],orientation="h",
            marker_color=[C["cl1"] if v>0 else C["cl2"] for v in shap_df_w["SHAP"]],
            text=[f"{v:+.2f}" for v in shap_df_w["SHAP"]],
            textposition="outside",textfont_size=10))
        fig_sh.add_vline(x=0,line_color="#333",line_width=1)
        fig_sh.update_layout(height=195,margin=dict(l=5,r=55,t=5,b=5),
            xaxis_range=[-0.4,3.0],
            xaxis_title="SHAP Value (positive → Cluster 1)",
            yaxis_title="",yaxis_tickfont_size=9,
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sh,use_container_width=True)
        st.caption("Net SHAP = +5.54 - highest of any county.")
    with sr:
        st.markdown("##### Intervention Priority")
        for _,row in shap_df_w.iterrows():
            box(f"<b>{row['Priority']}:</b> {row['Domain']}<br>"
                f"SHAP = {row['SHAP']:+.2f}  {row['Direction']}",
                "warn" if row["SHAP"]>0 else "ok")
        box("ICI SHAP = −0.12: Wajir's immunisation performance partially offsets "
            "structural deprivation without neutralising it. "
            "Net SHAP +5.54 remains the highest of any county.","info")

    st.markdown("---")
    st.markdown("##### Key Insight: Independence of Immunisation Coverage")
    box("<b>ICI r with UCS = 0.068 · r² = 0.005 · CV weight = 0.0028 · XGBoost importance = 3.9%</b><br>"
        "Domain inter-correlations: ICI–HAI r=0.073, ICI–PVI r=0.043, ICI–DBI r=−0.044. "
        "All |r| < 0.08. A county can achieve high immunisation coverage whilst remaining severely "
        "underserved on healthcare access, vulnerability, and disease burden. "
        "Tana River (rank 3, UCS 95.38): its adequate immunisation coverage is entirely invisible as "
        "an indicator of its compound deprivation under single-indicator monitoring. ")

# ── KDHS PREDICTOR ────────────────────────────────────────────────────────────
elif page == "KDHS Predictor":
    st.markdown("##### KDHS Raw Data Predictor")
    st.caption("Upload a county-level KDHS CSV → auto-map to domains → compute domain scores → "
               "predict UCS using CV weighting (dissertation methodology,).")

    s1,s2,s3,s4 = st.columns(4)
    with s1: st.markdown('<span style="font-size:.78rem;font-weight:600;color:#1E3888">1 - Upload CSV</span>',unsafe_allow_html=True)
    with s2: st.markdown('<span style="font-size:.78rem;font-weight:600;color:#1E3888">2 - Review mapping</span>',unsafe_allow_html=True)
    with s3: st.markdown('<span style="font-size:.78rem;font-weight:600;color:#1E3888">3 - Compute scores</span>',unsafe_allow_html=True)
    with s4: st.markdown('<span style="font-size:.78rem;font-weight:600;color:#1E3888">4 - View predictions</span>',unsafe_allow_html=True)
    st.markdown("---")

    KDHS_KWS = {
        "Healthcare Access Index":        ["antenatal","anc","skilled_birth","delivery","postnatal",
                                           "pnc","insurance","nhif","facility","travel_time","distance",
                                           "hrh","doctor","nurse","midwife"],
        "Population Vulnerability Index": ["wealth","poverty","poorest","education","no_school",
                                           "water","sanitation","toilet","wash","dependency","literacy"],
        "Immunisation Coverage Index":    ["bcg","dpt","polio","measles","vaccine","immuniz",
                                           "vitamin_a","deworming","fully_immunized","vaccination"],
        "Disease Burden Index":           ["stunting","wasting","underweight","malnutrition",
                                           "anaemia","malaria","fever","diarrhea","ari","tb","hiv"],
    }

    def detect_domain(col):
        for d,kws in KDHS_KWS.items():
            if any(kw in col.lower() for kw in kws): return d
        return None

    cu,cf = st.columns([2,1])
    with cu:
        uploaded = st.file_uploader("Upload county-level KDHS CSV",type=["csv","xlsx"])
    with cf:
        box("<b>Format:</b> rows=counties, columns=indicators, county name in first column.","info")

    if not uploaded:
        if st.button("Generate demo data (47 counties)",type="secondary"):
            np.random.seed(42)
            counties = list(UCS_DATA.keys())
            demo = {"County":counties,
                    "antenatal_4plus_pct":    np.random.uniform(20,95,47),
                    "skilled_birth_pct":      np.random.uniform(15,98,47),
                    "health_insurance_pct":   np.random.uniform(5,75,47),
                    "distance_barrier_pct":   np.random.uniform(5,80,47),
                    "poorest_quintile_pct":   np.random.uniform(5,70,47),
                    "no_education_pct":       np.random.uniform(2,60,47),
                    "unimproved_water_pct":   np.random.uniform(5,85,47),
                    "fully_immunized_pct":    np.random.uniform(30,98,47),
                    "bcg_coverage_pct":       np.random.uniform(50,99,47),
                    "stunting_pct":           np.random.uniform(5,55,47),
                    "malaria_prevalence_pct": np.random.uniform(0,60,47),
                    "anaemia_children_pct":   np.random.uniform(10,80,47),
                    }
            st.session_state["demo_df"] = pd.DataFrame(demo)
            st.download_button("Download demo CSV",
                pd.DataFrame(demo).to_csv(index=False).encode(),
                "kdhs_demo.csv","text/csv")
            box("Demo data ready.","ok")

    raw_df = None
    if uploaded:
        try:
            raw_df = (pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx")
                      else pd.read_csv(uploaded))
            box(f"Loaded {len(raw_df)} rows × {len(raw_df.columns)} columns.","ok")
        except Exception as e:
            box(f"Error: {e}","err")
    elif "demo_df" in st.session_state:
        raw_df = st.session_state["demo_df"]
        box("Using demo data.","info")

    if raw_df is not None:
        cc = next((c for c in raw_df.columns if any(w in c.lower()
                   for w in ["county","region","area"])),raw_df.columns[0])
        raw_df = raw_df.set_index(cc)
        raw_df.index.name = "County"
        st.dataframe(raw_df.head(),use_container_width=True,height=120)

        auto_map = {d:[] for d in KDHS_KWS}
        for col in raw_df.columns:
            d = detect_domain(col)
            if d: auto_map[d].append(col)

        mc1,mc2,mc3,mc4 = st.columns(4)
        for mcol,dom in zip([mc1,mc2,mc3,mc4],KDHS_KWS.keys()):
            with mcol: kpi(len(auto_map[dom]),dom.replace(" Index",""),C["blue"])

        unmapped = [c for c in raw_df.columns if not any(c in v for v in auto_map.values())]
        if unmapped:
            box(f"{len(unmapped)} columns not mapped and will be excluded.","warn")

        # Detect whether file already has pre-computed domain scores
        PRECOMP_COLS = {
            "Healthcare Access Index":        "Healthcare Access Index",
            "Population Vulnerability Index": "Population Vulnerability Index",
            "Immunization Coverage Index":    "Immunization Coverage Index",
            "Disease Burden Index":           "Disease Burden Index",
        }
        has_precomp = all(d in raw_df.columns for d in PRECOMP_COLS.values())
        has_ucs_col = "UCS" in raw_df.columns

        if has_precomp:
            box("Pre-computed domain scores detected - reading directly from your file.", "ok")
        else:
            mc1,mc2,mc3,mc4 = st.columns(4)
            for mcol,dom in zip([mc1,mc2,mc3,mc4],KDHS_KWS.keys()):
                with mcol: kpi(len(auto_map[dom]),dom.replace(" Index",""),C["blue"])
            unmapped = [c for c in raw_df.columns
                        if not any(c in v for v in auto_map.values())
                        and c not in list(PRECOMP_COLS.values())
                        + ["UCS","Cluster","Cluster_label","HDBSCAN_Cluster",
                           "Anomaly","Anomaly_score","Predicted_Anomaly",
                           "Predicted_Anomaly_label"]]
            if unmapped:
                box(f"{len(unmapped)} columns not mapped and will be excluded.", "warn")

        if st.button("Load and Score Counties", type="primary"):
            with st.spinner("Processing your data…"):
                try:
                    if has_precomp:
                        # Use the pre-computed domain columns directly
                        ds = pd.DataFrame(index=raw_df.index)
                        for dest, src_col in PRECOMP_COLS.items():
                            ds[dest] = pd.to_numeric(raw_df[src_col], errors="coerce")

                        if has_ucs_col:
                            ds["UCS"] = pd.to_numeric(raw_df["UCS"], errors="coerce")
                        else:
                            # Recompute UCS from domain scores using CV weighting
                            cv_wts = {}
                            for d in PRECOMP_COLS:
                                s = ds[d].dropna(); mu = s.mean()
                                cv_wts[d] = (s.std()/abs(mu)) if mu!=0 else 0.0
                            total_cv = sum(cv_wts.values())
                            wts = {d: cv_wts[d]/total_cv if total_cv>0 else 0.25
                                   for d in PRECOMP_COLS}
                            raw_ucs2 = sum(ds[d]*wts[d] for d in PRECOMP_COLS)
                            mn2,mx2 = raw_ucs2.min(), raw_ucs2.max()
                            ds["UCS"] = (raw_ucs2-mn2)/(mx2-mn2)*100 if mx2>mn2 else raw_ucs2

                        # Carry cluster and anomaly labels if present
                        if "Cluster_label" in raw_df.columns:
                            lbl = raw_df["Cluster_label"].astype(str)
                            ds["Cluster"] = lbl.str.extract(r"(Cluster \d)")[0].fillna(lbl)
                        elif "Cluster" in raw_df.columns:
                            ds["Cluster"] = raw_df["Cluster"].map(
                                {0:"Cluster 1",1:"Cluster 2",
                                 "0":"Cluster 1","1":"Cluster 2"}
                            ).fillna(raw_df["Cluster"].astype(str))
                        if "Anomaly" in raw_df.columns:
                            ds["Anomaly"] = raw_df["Anomaly"]
                        if "Anomaly_score" in raw_df.columns:
                            ds["IF_score"] = pd.to_numeric(raw_df["Anomaly_score"], errors="coerce")

                    else:
                        # Raw indicator mode - PCA + CV weighting
                        domain_scores = {}
                        for dom, cols in auto_map.items():
                            if not cols:
                                domain_scores[dom] = pd.Series(np.nan, index=raw_df.index)
                                continue
                            sub = raw_df[cols].copy().apply(pd.to_numeric, errors="coerce")
                            sub = sub.fillna(sub.median())
                            good = ["immuniz","vaccine","bcg","dpt","polio","measles",
                                    "vitamin","insur","skilled","antenatal"]
                            for c in cols:
                                if any(kw in c.lower() for kw in good):
                                    sub[c] = 100-sub[c].clip(0,100) if sub[c].max()<=100 else -sub[c]
                            if len(cols)==1 or not SKLEARN_OK:
                                score = sub.iloc[:,0]
                            else:
                                try:
                                    Xs = StandardScaler().fit_transform(sub)
                                    score = pd.Series(
                                        PCA(n_components=1).fit_transform(Xs).ravel(),
                                        index=raw_df.index)
                                except Exception:
                                    score = sub.mean(axis=1)
                            mn,mx = score.min(),score.max()
                            domain_scores[dom] = ((score-mn)/(mx-mn)*100
                                                  if mx>mn else pd.Series(50.0, index=raw_df.index))
                        ds = pd.DataFrame(domain_scores)
                        cv_wts = {}
                        for d in KDHS_KWS:
                            s = ds[d].dropna(); mu = s.mean()
                            cv_wts[d] = (s.std()/abs(mu)) if mu!=0 else 0.0
                        total_cv = sum(cv_wts.values())
                        wts = {d: cv_wts[d]/total_cv if total_cv>0 else 0.25 for d in KDHS_KWS}
                        raw_ucs = sum(ds[d]*wts[d] for d in KDHS_KWS)
                        mn,mx = raw_ucs.min(),raw_ucs.max()
                        ds["UCS"] = (raw_ucs-mn)/(mx-mn)*100 if mx>mn else raw_ucs

                    # Safe rank - handles NaN/inf
                    ucs_clean = pd.to_numeric(ds["UCS"], errors="coerce")
                    ds["UCS"] = ucs_clean
                    ds["Rank"] = (ucs_clean
                                  .fillna(0)
                                  .rank(ascending=False, na_option="bottom")
                                  .astype("Int64"))
                    st.session_state["pred_results"] = ds
                    box(f"Computed UCS for {len(ds)} counties.","ok")
                except Exception as e:
                    box(f"Error: {e}","err")

        if "pred_results" in st.session_state:
            results = st.session_state["pred_results"]
            st.markdown("---")
            rk1,rk2,rk3,rk4 = st.columns(4)
            with rk1: kpi(len(results),"Counties",C["blue"])
            with rk2: kpi(f"{results['UCS'].mean():.1f}","Mean UCS",C["orange"])
            worst_p = results["UCS"].idxmax()
            with rk3: kpi(f"{results['UCS'].max():.1f}",f"Worst: {worst_p[:8]}",C["cl1"])
            best_p = results["UCS"].idxmin()
            with rk4: kpi(f"{results['UCS'].min():.1f}",f"Best: {best_p[:8]}",C["cl2"])

            rl,rr = st.columns([1.3,1])
            with rl:
                tr = results["UCS"].sort_values(ascending=False).head(10).reset_index()
                tr.columns = ["County","UCS"]
                fig_pr = px.bar(tr,x="UCS",y="County",orientation="h",
                    color="UCS",color_continuous_scale=[C["cl2"],C["cl1"]],range_color=[0,100])
                fig_pr.update_traces(texttemplate="%{x:.1f}",textposition="outside",textfont_size=9)
                fig_pr.update_layout(height=280,margin=dict(l=5,r=40,t=5,b=5),
                    yaxis_autorange="reversed",xaxis_range=[0,115],
                    coloraxis_showscale=False,yaxis_tickfont_size=8,
                    plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pr,use_container_width=True)
            with rr:
                show = results[["UCS","Rank"]].sort_values("UCS",ascending=False).copy()
                show["UCS"] = show["UCS"].round(1)
                st.dataframe(show,use_container_width=True,height=280)

            csv_out = results.round(3).to_csv().encode()
            st.download_button("Download UCS Predictions (CSV)",csv_out,
                "ucs_predictions.csv","text/csv",type="primary")

            common = [c for c in results.index if c in df.index]
            if len(common)>=5:
                st.markdown("##### Comparison with Dissertation Reference")
                comp = pd.DataFrame({
                    "Predicted": results.loc[common,"UCS"],
                    "Reference": df.loc[common,"UCS"],
                })
                corr = comp.corr().iloc[0,1]
                fig_cmp = px.scatter(comp,x="Reference",y="Predicted",
                    hover_name=comp.index,
                    title=f"Predicted vs Reference UCS - r = {corr:.3f}")
                fig_cmp.add_trace(go.Scatter(x=[0,100],y=[0,100],mode="lines",
                    line=dict(dash="dash",color="grey"),name="Perfect agreement"))
                fig_cmp.update_layout(height=300,margin=dict(l=20,r=20,t=35,b=20))
                st.plotly_chart(fig_cmp,use_container_width=True)
                if corr>0.8: box(f"Strong agreement withreference (r={corr:.3f}).","ok")
                elif corr>0.5: box(f"Moderate agreement (r={corr:.3f}). Review column mappings.","warn")
                else: box(f"Weak agreement (r={corr:.3f}). Check assignments and data scale.","err")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Kenya Health Equity Monitor · UCS Methodology · KDHS 2020 & 2022 · 47 Counties · "
           "Cynthia Ngugi (138725) · MSc Data Science & Analytics · Strathmore University · March 2026")
