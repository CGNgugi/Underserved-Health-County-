# Kenya Health Gap Dashboard

A comprehensive Streamlit dashboard for healthcare access inequality analysis in Kenya's 47 counties using the UCS (Underserved County Score) methodology.

## Features

- **Overview** - Summary statistics, domain distributions, choropleth map
- **Interactive Map** - Geographic exploration with UCS filters & cluster overlay
- **PCA Analysis** - Principal Component Analysis for feature composition
- **County Deep Dive** - Individual county analysis with radar charts
- **ML & SHAP** - Machine learning predictions and SHAP values

## Tech Stack

- Python
- Streamlit
- Plotly
- scikit-learn
- Folium

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run kenya_health.py
```

## Deployment

This dashboard is ready for deployment to **Streamlit Community Cloud**:
1. Push to a public GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub and select the repository

## Author

Ngugi, Cynthia (138725) | MSc Data Science & Analytics, Strathmore University

## Data Sources

- KDHS 2020 & 2022
- Kenya Health Survey Data
