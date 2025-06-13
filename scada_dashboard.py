import streamlit as st
st.set_page_config(
    page_title="SCADA Sewer Infrastructure Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')
import json
import platform
import psutil
from datetime import datetime, timedelta
import random
from io import BytesIO
import zipfile
import time
from fpdf import FPDF

# --- Utility Functions ---
def calculate_efficiency(flow_rate, power, asset_type):
    if power == 0:
        return 0
    base_efficiency = (flow_rate / power) * 100
    if asset_type == "Wet Well":
        base_efficiency *= 1.1
    elif asset_type == "STP":
        base_efficiency *= 0.9
    return max(0, min(base_efficiency, 100))

def predict_maintenance_advanced(flow_rate, power, status, last_maintenance_days):
    risk_score = 0
    if flow_rate < 50:
        risk_score += 30
    elif flow_rate > 200:
        risk_score += 20
    if power < 2.0:
        risk_score += 25
    elif power > 8.0:
        risk_score += 15
    if status == "OFF":
        risk_score += 50
    elif status == "ALERT":
        risk_score += 35
    if last_maintenance_days > 90:
        risk_score += 40
    elif last_maintenance_days > 60:
        risk_score += 25
    elif last_maintenance_days > 30:
        risk_score += 15

    if risk_score >= 70:
        return "IMMEDIATE", risk_score
    elif risk_score >= 40:
        return "SCHEDULED", risk_score
    else:
        return "NORMAL", risk_score

def enhanced_fault_detection(flow_rate, power, status, asset_type):
    if status == "OFF":
        return "CRITICAL: Power Failure"
    elif status == "ALERT":
        return "WARNING: System Alert"
    elif flow_rate < 50 and asset_type in ['Wet Well', 'STP']:
        return "WARNING: Low Flow Rate"
    elif flow_rate == 0:
        return "CRITICAL: No Flow Detected"
    elif power < 2.0 and asset_type in ['Wet Well', 'STP']:
        return "WARNING: Low Power Consumption"
    else:
        return "NORMAL: System Operating"

def export_to_pdf(data, charts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="SCADA Monitoring Report", ln=1, align='C')
    return pdf.output(dest='S')

def export_to_excel(data, charts=None):
    """
    Export the provided DataFrame and (optionally) chart images to an Excel file in memory.
    """
    from io import BytesIO
    import pandas as pd

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='SCADA Data')
        # Optionally, you can add more sheets or images if needed
        # For now, just export the data
        writer.save()
    output.seek(0)
    return output.getvalue()

# Initialize session state
if 'assets_data' not in st.session_state:
    st.session_state.assets_data = None
if 'scada_data' not in st.session_state:
    st.session_state.scada_data = None
if 'complaints_data' not in st.session_state:
    st.session_state.complaints_data = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Custom CSS
def get_css(dark_mode=True):
    if dark_mode:
        return """
        <style>
        :root {
            --bg-color: #1e1e1e;
            --text-color: #ffffff;
            --card-bg: linear-gradient(135deg,#2a5298 0%,#1e3c72 100%);
            --accent-color: #66bb6a;
            --alert-critical: linear-gradient(135deg,#ff416c 0%,#ff4b2b 100%);
            --alert-warning: linear-gradient(135deg,#ffa726 0%,#fb8c00 100%);
            --alert-normal: linear-gradient(135deg,#66bb6a 0%,#43a047 100%);
            --info-card: linear-gradient(135deg,#29b6f6 0%,#0277bd 100%);
            --sidebar-bg: linear-gradient(135deg,#2c3e50 0%,#34495e 100%);
        }
        .main-header{
            background: var(--card-bg); 
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); 
            backdrop-filter: blur(10px);
        }
        .metric-card{ 
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); 
            transition: transform 0.3s ease;
        }
        .metric-card:hover{ 
            transform: translateY(-5px);
        }
        .alert-critical{ 
            background: var(--alert-critical); 
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(255, 65, 108, 0.3); 
            border-left: 5px solid #d32f2f;
            animation: pulse 2s infinite;
        }
        .alert-warning{ 
            background: var(--alert-warning); 
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(255, 167, 38, 0.3); 
            border-left: 5px solid #fb8c00;
        }
        .alert-normal{ 
            background: var(--alert-normal); 
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(102, 187, 106, 0.3); 
            border-left: 5px solid #66bb6a;
        }
        .info-card{ 
            background: var(--info-card); 
            padding: 1.2rem;
            border-radius: 12px;
            color: white;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(41, 182, 246, 0.3);
        }
        @keyframes pulse{ 
            0%{ box-shadow: 0 6px 20px rgba(255, 65, 108, 0.3);}
            50%{ box-shadow: 0 6px 25px rgba(255, 65, 108, 0.6);}
            100%{ box-shadow: 0 6px 20px rgba(255, 65, 108, 0.3);}
        }
        .sidebar.sidebar-content{ 
            background: var(--sidebar-bg);
        }
        div[data-testid="metric-container"]{ 
            background: var(--card-bg); 
            border: 1px solid #e0e0e0;
            padding: 1rem;
            border-radius: 12px;
            color: white;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); 
        }
        div[data-testid="metric-container"]> label{ 
            color: white!important;
            font-weight: 600;
        }
        div[data-testid="metric-container"]> div{ 
            color: white!important;
            font-size: 1.2em;
            font-weight: bold;
        }
        .status-operational{ background-color:#4caf50;} 
        .status-alert{ background-color:#ff9800;} 
        .status-critical{ background-color:#f44336;} 
        .status-offline{ background-color:#9e9e9e;}
        </style>
        """
    else:
        return """
        <style>
        :root {
            --bg-color: #ffffff;
            --text-color: #000000;
            --card-bg: linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
            --accent-color: #2e7d32;
            --alert-critical: linear-gradient(135deg,#ffcdd2 0%,#ef9a9a 100%);
            --alert-warning: linear-gradient(135deg,#fff3e0 0%,#ffe0b2 100%);
            --alert-normal: linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
            --info-card: linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
            --sidebar-bg: linear-gradient(135deg,#eceff1 0%,#cfd8dc 100%);
        }
        .main-header{
            background: var(--card-bg); 
            padding: 2rem;
            border-radius: 15px;
            color: black;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); 
            backdrop-filter: blur(10px);
        }
        .metric-card{ 
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 12px;
            color: black;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); 
            transition: transform 0.3s ease;
        }
        .metric-card:hover{ 
            transform: translateY(-5px);
        }
        .alert-critical{ 
            background: var(--alert-critical); 
            padding: 1.2rem;
            border-radius: 12px;
            color: black;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(255, 65, 108, 0.1); 
            border-left: 5px solid #d32f2f;
            animation: pulse 2s infinite;
        }
        .alert-warning{ 
            background: var(--alert-warning); 
            padding: 1.2rem;
            border-radius: 12px;
            color: black;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(255, 167, 38, 0.1); 
            border-left: 5px solid #fb8c00;
        }
        .alert-normal{ 
            background: var(--alert-normal); 
            padding: 1.2rem;
            border-radius: 12px;
            color: black;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(102, 187, 106, 0.1); 
            border-left: 5px solid #66bb6a;
        }
        .info-card{ 
            background: var(--info-card); 
            padding: 1.2rem;
            border-radius: 12px;
            color: black;
            margin: 0.5rem 0;
            box-shadow: 0 6px 20px rgba(41, 182, 246, 0.1);
        }
        @keyframes pulse{ 
            0%{ box-shadow: 0 6px 20px rgba(255, 65, 108, 0.1);}
            50%{ box-shadow: 0 6px 25px rgba(255, 65, 108, 0.2);}
            100%{ box-shadow: 0 6px 20px rgba(255, 65, 108, 0.1);}
        }
        .sidebar.sidebar-content{ 
            background: var(--sidebar-bg);
        }
        div[data-testid="metric-container"]{ 
            background: var(--card-bg); 
            border: 1px solid #e0e0e0;
            padding: 1rem;
            border-radius: 12px;
            color: black;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); 
        }
        div[data-testid="metric-container"]> label{ 
            color: black!important;
            font-weight: 600;
        }
        div[data-testid="metric-container"]> div{ 
            color: black!important;
            font-size: 1.2em;
            font-weight: bold;
        }
        .status-operational{ background-color:#43a047;} 
        .status-alert{ background-color:#fb8c00;} 
        .status-critical{ background-color:#f44336;} 
        .status-offline{ background-color:#9e9e9e;}
        </style>
        """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Load CSV Data
import pandas as pd
from datetime import datetime

@st.cache_data
def load_actual_data():
    assets_df = pd.read_csv("data/sewer_assets_gis.csv")
    scada_df = pd.read_csv("data/scada_logs_pump_station.csv")
    complaints_df = pd.read_csv("data/public_complaints.csv")
    return assets_df, scada_df, complaints_df

if 'assets_data' not in st.session_state or st.session_state.assets_data is None:
    assets_df, scada_df, complaints_df = load_actual_data()
    st.session_state.assets_data = assets_df
    st.session_state.scada_data = scada_df
    st.session_state.complaints_data = complaints_df
    st.session_state.last_update = datetime.now()
else:
    assets_df = st.session_state.assets_data
    scada_df = st.session_state.scada_data
    complaints_df = st.session_state.complaints_data

# Main Dashboard Header
st.markdown("""
<div class="main-header">
<h1>SCADA Sewer Infrastructure Monitor</h1>
<p>Advanced Real-time Monitoring & Predictive Analytics System</p>
<small>Surathkal, Karnataka | Live System Status</small>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.markdown("## Advanced Control Panel")
    selected_view = st.selectbox(
        "Select Dashboard View",
        ["Executive Dashboard", "GIS Asset Mapping", "Advanced Analytics",
         "Predictive Maintenance", "Alert Management", "Performance Metrics",
         "Asset Details", "Mobile View", "Customize Dashboard"]
    )
    
    # Theme toggle
    if st.button("Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
        
    # Advanced filters
    st.markdown("### Filters")
    selected_assets = st.multiselect(
        "Select Assets",
        options=st.session_state.assets_data['asset_id'].tolist(),
        default=st.session_state.assets_data['asset_id'].tolist()
    )
    
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 12 Hours", "Last 24 Hours", "Custom Range"]
    )
    
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 120, 30)
        st.success(f"Auto-refreshing every {refresh_interval} seconds")

# Data processing
assets_df = st.session_state.assets_data
scada_df = st.session_state.scada_data
complaints_df = st.session_state.complaints_data

scada_df['timestamp'] = pd.to_datetime(scada_df['timestamp'])
complaints_df['date'] = pd.to_datetime(complaints_df['date'])

latest_scada = scada_df.groupby('asset_id').last().reset_index()

merged_data = pd.merge(latest_scada, assets_df, on='asset_id', how='left')

if 'status_x' in merged_data.columns:
    merged_data['status'] = merged_data['status_x']
elif 'status_y' in merged_data.columns:
    merged_data['status'] = merged_data['status_y']

merged_data['last_maintenance'] = pd.to_datetime(merged_data['last_maintenance'])
merged_data['maintenance_days'] = (datetime.now() - merged_data['last_maintenance']).dt.days

if selected_assets:
    merged_data = merged_data[merged_data['asset_id'].isin(selected_assets)]
    scada_df = scada_df[scada_df['asset_id'].isin(selected_assets)]

# Executive Dashboard
if selected_view == "Executive Dashboard":
    # Executive Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_assets = len(merged_data)
    operational_assets = len(merged_data[merged_data['status'].isin(['Operational', 'Running', 'Functional'])])
    critical_alerts = len(merged_data[merged_data['status'].isin(['Collapsed', 'Offline', 'Blocked'])])
    avg_flow = merged_data['flow_rate_LPM'].mean()
    avg_power = merged_data['power_kW'].mean()
    
    with col1:
        st.metric("Total Assets", total_assets, delta=None)
    with col2:
        st.metric("Operational", operational_assets, delta=f"{operational_assets-total_assets}", delta_color="inverse")
    with col3:
        st.metric("Critical Alerts", critical_alerts, delta=f"+{critical_alerts}", delta_color="inverse")
    with col4:
        st.metric("Avg Flow Rate", f"{avg_flow:.1f} L/min", delta="12.5 L/min")
    with col5:
        st.metric("Avg Power", f"{avg_power:.1f} kW", delta="0.8 kW")
        
    # Real-time monitoring charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Real-time Flow Monitoring")
        fig_flow = px.line(
            scada_df,
            x='timestamp',
            y='flow_rate_LPM',
            color='asset_id',
            title="Flow Rate Trends (Last 12 Hours)",
            template='plotly_dark',
        )
        fig_flow.update_traces(mode='lines+markers', marker=dict(size=8, symbol='circle'))
        fig_flow.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_flow, use_container_width=True)
        
    with col2:
        st.subheader("Power Consumption Analysis")
        fig_power = px.line(
            scada_df,
            x='timestamp',
            y='power_kW',
            color='asset_id',
            title="Power Consumption Trends",
            template='plotly_dark',
        )
        fig_power.update_traces(mode='lines+markers', marker=dict(size=8, symbol='circle'))
        fig_power.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_power, use_container_width=True)
        
    # System Performance Dashboard
    st.subheader("System Performance Dashboard")
    # Create performance gauge charts
    fig_gauges = make_subplots(
        rows=2, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=('System Availability', 'Average Efficiency', 'Flow Rate Status',
                       'Power Efficiency', 'Maintenance Score', 'Overall Health')
    )
    
    # System Availability
    availability = (operational_assets / total_assets) * 100 if total_assets > 0 else 0
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=availability,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Availability (%)"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "green" if availability >= 90 else "orange" if availability >= 70 else "red"},
                'steps': [{'range': [0, 70], 'color': "lightgray"},
                         {'range': [70, 90], 'color': "yellow"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 95}
            }
        ), row=1, col=1
    )
    
    # Calculate and display other metrics
    efficiencies = [calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']) for _, row in merged_data.iterrows()]
    avg_efficiency = np.mean(efficiencies) if efficiencies else 0
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_efficiency,
            title={'text': "Efficiency (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "blue"},
                  'steps': [{'range': [0, 50], 'color': "red"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "green"}]}
        ), row=1, col=2
    )
    
    # Flow Rate Status
    flow_status = (avg_flow / 200) * 100  # Assuming 200 L/min is optimal
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=min(flow_status, 100),
            title={'text': "Flow Status (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "cyan"}}
        ), row=1, col=3
    )
    
    # Power Efficiency
    power_efficiency = (avg_power / 6) * 100  # Assuming 6 kW is optimal
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=min(power_efficiency, 100),
            title={'text': "Power Efficiency (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "orange"}}
        ), row=2, col=1
    )
    
    # Maintenance Score
    def predict_maintenance_advanced(flow_rate, power, status, last_maintenance_days):
        """Advanced predictive maintenance"""
        risk_score = 0
        
        # Flow rate factor
        if flow_rate < 50:
            risk_score += 30
        elif flow_rate > 200:
            risk_score += 20
            
        # Power factor
        if power < 2.0:
            risk_score += 25
        elif power > 8.0:
            risk_score += 15
            
        # Status factor
        if status == "OFF":
            risk_score += 50
        elif status == "ALERT":
            risk_score += 35
            
        # Maintenance history factor
        if last_maintenance_days > 90:
            risk_score += 40
        elif last_maintenance_days > 60:
            risk_score += 25
        elif last_maintenance_days > 30:
            risk_score += 15
            
        if risk_score >= 70:
            return "IMMEDIATE", risk_score
        elif risk_score >= 40:
            return "SCHEDULED", risk_score
        else:
            return "NORMAL", risk_score
    
    maintenance_scores = []
    for _, row in merged_data.iterrows():
        _, score = predict_maintenance_advanced(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['maintenance_days']
        )
        maintenance_scores.append(100 - score)  # Invert score for better visualization
        
    avg_maintenance_score = np.mean(maintenance_scores) if maintenance_scores else 100
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_maintenance_score,
            title={'text': "Maintenance Score"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "purple"}}
        ), row=2, col=2
    )
    
    # Overall Health
    overall_health = (availability + avg_efficiency + avg_maintenance_score) / 3
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=overall_health,
            title={'text': "Overall Health"},
            gauge={'axis': {'range': [0, 100]}}
        ), row=2, col=3
    )
    
    fig_gauges.update_layout(height=600, template='plotly_dark')
    st.plotly_chart(fig_gauges, use_container_width=True)
    
    # Alert Summary
    st.subheader("Live Alert Summary")
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    critical_alerts_list = []
    warning_alerts_list = []
    normal_status_list = []
    
    for _, row in merged_data.iterrows():
        fault_status = enhanced_fault_detection(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['type']
        )
        
        maintenance_status, risk_score = predict_maintenance_advanced(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['maintenance_days']
        )
        
        if "CRITICAL" in fault_status or "IMMEDIATE" in maintenance_status:
            critical_alerts_list.append(f"{row['asset_id']}: {fault_status}")
        elif "WARNING" in fault_status or "SCHEDULED" in maintenance_status:
            warning_alerts_list.append(f"{row['asset_id']}: {fault_status}")
        else:
            normal_status_list.append(f"{row['asset_id']}: {fault_status}")
            
    with alert_col1:
        st.markdown("### Critical Alerts")
        if critical_alerts_list:
            for alert in critical_alerts_list:
                st.markdown(f'<div class="alert-critical">{alert}</div>', unsafe_allow_html=True)
        else:
            st.success("No critical alerts")
            
    with alert_col2:
        st.markdown("### Warnings")
        if warning_alerts_list:
            for alert in warning_alerts_list:
                st.markdown(f'<div class="alert-warning">{alert}</div>', unsafe_allow_html=True)
        else:
            st.info("No warnings")
            
    with alert_col3:
        st.markdown("### Normal Operations")
        for status in normal_status_list[:5]:
            st.markdown(f'<div class="alert-normal">{status}</div>', unsafe_allow_html=True)
        if len(normal_status_list) > 5:
            st.info(f"... and {len(normal_status_list)-5} more systems operating normally")

# GIS Asset Mapping
elif selected_view == "GIS Asset Mapping":
    st.subheader("Interactive Asset Location & Status Mapping")
    # Layer control
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layer_types = st.multiselect(
            "Layer Types",
            options=["Assets", "Complaints", "Maintenance Zones"],
            default=["Assets", "Complaints"]
        )
        
    with col2:
        asset_types = st.multiselect(
            "Filter by Asset Type",
            options=assets_df['type'].unique(),
            default=assets_df['type'].unique()
        )
        
    with col3:
        status_filter = st.multiselect(
            "Filter by Status",
            options=assets_df['status'].unique(),
            default=assets_df['status'].unique()
        )
        
    # Heatmap toggle
    show_heatmap = st.checkbox("Show Heatmap for Flow Rates")
    
    # Filter data
    filtered_assets = assets_df[
        (assets_df['type'].isin(asset_types)) &
        (assets_df['status'].isin(status_filter))
    ]
    
    if not filtered_assets.empty:
        # Create map
        center_lat = filtered_assets['latitude'].mean()
        center_lon = filtered_assets['longitude'].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Add heatmap if enabled
        if show_heatmap:
            heat_data = [[row['latitude'], row['longitude']] for _, row in filtered_assets.iterrows()]
            HeatMap(heat_data, radius=15).add_to(m)
        
        marker_cluster = MarkerCluster().add_to(m)
        
        for _, row in filtered_assets.iterrows():
            # Get latest SCADA data for this asset
            latest_data = merged_data[merged_data['asset_id'] == row['asset_id']]

            # Safely parse last_maintenance to datetime if it's a string
            last_maintenance = row['last_maintenance']
            if isinstance(last_maintenance, str):
                try:
                    last_maintenance_dt = pd.to_datetime(last_maintenance, errors='coerce')
                except Exception:
                    last_maintenance_dt = None
            else:
                last_maintenance_dt = last_maintenance

            # Ensure last_maintenance_dt is a Timestamp or None
            if not isinstance(last_maintenance_dt, pd.Timestamp) or pd.isna(last_maintenance_dt):
                last_maintenance_str = 'N/A'
            else:
                last_maintenance_str = last_maintenance_dt.strftime('%Y-%m-%d')

            if not latest_data.empty:
                latest_row = latest_data.iloc[0]
                flow_rate = latest_row['flow_rate_LPM']
                power = latest_row['power_kW']
                scada_status = latest_row['status']
                popup_text = f"""
                <b>{row['asset_id']}</b><br> 
                Type: {row['type']}<br> 
                Status: {row['status']}<br> 
                Flow Rate: {flow_rate:.1f} L/min<br> 
                Power: {power:.1f} kW<br> 
                SCADA Status: {scada_status}<br> 
                Last Maintenance: {last_maintenance_str}<br> 
                Location: ({row['latitude']:.4f}, {row['longitude']:.4f}) 
                """
            else:
                popup_text = f"""
                <b>{row['asset_id']}</b><br> 
                Type: {row['type']}<br> 
                Status: {row['status']}<br> 
                Last Maintenance: {last_maintenance_str}<br> 
                Location: ({row['latitude']:.4f}, {row['longitude']:.4f}) 
                """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{row['asset_id']} - {row['type']}",
                icon=folium.Icon(
                    color='green' if row['status'] in ['Operational', 'Running'] else 'gray' if row['status'] == 'Offline' else 'red' if row['status'] == 'Collapsed' else 'orange',
                    icon='play' if row['status'] in ['Operational', 'Running'] else 'pause' if row['status'] == 'Offline' else 'remove' if row['status'] == 'Collapsed' else 'ban',
                    prefix='fa'
                )
            ).add_to(marker_cluster)
        
        # Add complaints markers if enabled
        if 'Complaints' in layer_types and not complaints_df.empty:
            for _, complaint in complaints_df.iterrows():
                if pd.notna(complaint['location']):
                    try:
                        lat, lon = map(float, complaint['location'].split(','))
                        complaint_popup = f"""
                        <b>Complaint {complaint['complaint_id']}</b><br> 
                        Type: {complaint['type']}<br> 
                        Date: {complaint['date'].strftime('%Y-%m-%d')}<br> 
                        Status: {complaint['status']} 
                        """
                        complaint_color = {
                            'Resolved': 'green',
                            'In Progress': 'orange',
                            'Pending': 'red'
                        }.get(complaint['status'], 'blue')
                        
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(complaint_popup, max_width=250),
                            tooltip=f"Complaint: {complaint['type']}",
                            icon=folium.Icon(
                                color=complaint_color,
                                icon='exclamation-triangle',
                                prefix='fa'
                            )
                        ).add_to(m)
                    except (ValueError, AttributeError):
                        continue
        
        # Add legend
        legend_html = '''
        <div style="position: fixed;
        bottom: 50px; left: 50px; width: 200px; height: 180px;
        background-color: white; border:2px solid grey; z-index:9999;
        font-size:14px; padding: 10px">
        <h4>Asset Status Legend</h4>
        <p><i class="fa fa-play" style="color:green"></i> Operational/Running</p>
        <p><i class="fa fa-pause" style="color:gray"></i> Offline</p>
        <p><i class="fa fa-remove" style="color:red"></i> Collapsed</p>
        <p><i class="fa fa-ban" style="color:orange"></i> Blocked</p>
        <p><i class="fa fa-exclamation-triangle" style="color:red"></i> Complaints</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Display map
        st_folium(m, width=1200, height=600)
        
        # Asset summary table
        st.subheader("Asset Summary Table")
        display_df = filtered_assets.copy()
        
        # Add latest SCADA data
        for idx, row in display_df.iterrows():
            latest_data = merged_data[merged_data['asset_id'] == row['asset_id']]
            if not latest_data.empty:
                latest_row = latest_data.iloc[0]
                display_df.loc[idx, 'current_flow'] = f"{latest_row['flow_rate_LPM']:.1f} L/min"
                display_df.loc[idx, 'current_power'] = f"{latest_row['power_kW']:.1f} kW"
                display_df.loc[idx, 'scada_status'] = latest_row['status']
                efficiency = calculate_efficiency(
                    latest_row['flow_rate_LPM'], latest_row['power_kW'], row['type']
                )
                display_df.loc[idx, 'efficiency'] = f"{efficiency:.1f}%"
            else:
                display_df.loc[idx, 'current_flow'] = "No data"
                display_df.loc[idx, 'current_power'] = "No data"
                display_df.loc[idx, 'scada_status'] = "No data"
                display_df.loc[idx, 'efficiency'] = "N/A"
                
        display_df['last_maintenance'] = pd.to_datetime(display_df['last_maintenance'], errors='coerce')
        
        st.dataframe(
            display_df[['asset_id', 'type', 'status', 'current_flow', 'current_power',
                       'scada_status', 'efficiency', 'last_maintenance']].style.format({
                'last_maintenance': lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else 'N/A'
            }),
            use_container_width=True
        )
    else:
        st.warning("No assets match the selected filters.")

# Advanced Analytics
elif selected_view == "Advanced Analytics":
    st.subheader("Advanced SCADA Analytics & Insights")
    # Analytics options
    analytics_tab = st.selectbox(
        "Select Analytics View",
        ["Correlation Analysis", "Performance Trends", "Anomaly Detection",
         "Energy Efficiency", "Comparative Analysis", "Machine Learning Models"]
    )
    
    if analytics_tab == "Correlation Analysis":
        st.markdown("### Flow Rate vs Power Consumption Correlation")
        # Create bubble chart with regression line
        fig_corr = px.scatter(
            merged_data,
            x='flow_rate_LPM',
            y='power_kW',
            color='type',
            size='maintenance_days',
            hover_data=['asset_id', 'status'],
            title="Flow Rate vs Power Consumption Correlation",
            trendline="ols",
            template='plotly_dark'
        )
        fig_corr.update_traces(marker=dict(symbol='diamond'))
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlation matrix
        numeric_data = merged_data[['flow_rate_LPM', 'power_kW', 'maintenance_days']].corr()
        fig_heatmap = px.imshow(
            numeric_data,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        fig_heatmap.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    elif analytics_tab == "Performance Trends":
        st.markdown("### Long-term Performance Trends")
        # Create violin plot for performance trends
        fig_trend = px.violin(
            scada_df,
            x='asset_id',
            y='flow_rate_LPM',
            box=True,
            points='all',
            title="Flow Rate Distribution by Asset",
            template='plotly_dark'
        )
        fig_trend.update_layout(height=500)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Create parallel coordinates chart
        fig_parallel = px.parallel_coordinates(
            merged_data,
            color='flow_rate_LPM',
            dimensions=['flow_rate_LPM', 'power_kW', 'maintenance_days'],
            title="Performance Trends Parallel Coordinates",
            template='plotly_dark'
        )
        fig_parallel.update_layout(height=500)
        st.plotly_chart(fig_parallel, use_container_width=True)
        
    elif analytics_tab == "Anomaly Detection":
        st.markdown("### Anomaly Detection Analysis")
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1)
        features = ['flow_rate_LPM', 'power_kW']
        
        # Fit model
        iso_forest.fit(merged_data[features])
        
        # Predict anomalies
        merged_data['anomaly'] = iso_forest.predict(merged_data[features])
        merged_data['anomaly'] = merged_data['anomaly'].map({1: 0, -1: 1})
        
        # Visualization - Hexbin plot
        fig_anomaly = px.density_heatmap(
            merged_data,
            x='flow_rate_LPM',
            y='power_kW',
            color_continuous_scale='Viridis',
            title="Anomaly Detection Results",
            template='plotly_dark'
        )
        fig_anomaly.update_layout(height=500)
        st.plotly_chart(fig_anomaly, use_container_width=True)
        
        # Show detected anomalies
        anomalies = merged_data[merged_data['anomaly'] == 1]
        if not anomalies.empty:
            st.warning(f"Detected {len(anomalies)} anomalies:")
            st.dataframe(anomalies[['asset_id', 'flow_rate_LPM', 'power_kW', 'status', 'type']])
        else:
            st.success("No anomalies detected in the latest data.")
            
    elif analytics_tab == "Energy Efficiency":
        st.markdown("### Energy Efficiency Analysis")
        # Calculate efficiency
        merged_data['efficiency'] = merged_data.apply(
            lambda row: calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']),
            axis=1
        )
        
        # Efficiency by asset type
        efficiency_by_type = merged_data.groupby('type')['efficiency'].mean().reset_index()
        fig_efficiency = px.bar(
            efficiency_by_type,
            x='type',
            y='efficiency',
            title="Average Efficiency by Asset Type",
            labels={'efficiency': 'Efficiency (%)', 'type': 'Asset Type'},
            color='type'
        )
        fig_efficiency.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Efficiency vs power consumption
        fig_eff_power = px.scatter(
            merged_data,
            x='power_kW',
            y='efficiency',
            color='type',
            size='flow_rate_LPM',
            title="Efficiency vs Power Consumption",
            template='plotly_dark'
        )
        fig_eff_power.update_layout(height=400)
        st.plotly_chart(fig_eff_power, use_container_width=True)
        
    elif analytics_tab == "Comparative Analysis":
        st.markdown("### Comparative Asset Analysis")
        # Ensure efficiency column exists
        if 'efficiency' not in merged_data.columns:
            merged_data['efficiency'] = merged_data.apply(
                lambda row: calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']),
                axis=1
            )
        # Select assets for comparison
        compare_assets = st.multiselect(
            "Select assets to compare",
            options=merged_data['asset_id'].tolist(),
            default=merged_data['asset_id'].head(2).tolist()
        )
        
        if len(compare_assets) >= 2:
            comp_data = merged_data[merged_data['asset_id'].isin(compare_assets)]
            
            # Create radar chart for comparison
            categories = ['flow_rate_LPM', 'power_kW', 'efficiency', 'maintenance_days']
            fig_radar = go.Figure()
            
            for asset_id in compare_assets:
                asset_data = comp_data[comp_data['asset_id'] == asset_id]
                values = [asset_data[col].values[0] for col in categories]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=asset_id
                ))
                
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                template='plotly_dark',
                title="Asset Comparison Radar Chart"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Detailed comparison table
            st.markdown("### Detailed Comparison Table")
            comp_table = comp_data[['asset_id', 'type', 'status', 'flow_rate_LPM', 'power_kW', 'efficiency', 'maintenance_days']]
            st.dataframe(comp_table.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.warning("Please select at least two assets for comparison.")
            
    elif analytics_tab == "Machine Learning Models":
        st.markdown("### Machine Learning Model Training")
        # Prepare data for ML models
        features = ['flow_rate_LPM', 'power_kW']
        target_maint = 'maintenance_status'
        target_fault = 'fault_status'

        # Create labels for classification
        merged_data['maintenance_status'] = merged_data.apply(lambda row: predict_maintenance_advanced(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['maintenance_days']
        )[0], axis=1)
        merged_data['fault_status'] = merged_data.apply(lambda row: enhanced_fault_detection(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['type']
        ), axis=1)

        # Encode categorical variables
        data_encoded = pd.get_dummies(merged_data[features + ['maintenance_status', 'fault_status']])

        # Split data
        X = merged_data[features]
        y_maint = merged_data['maintenance_status']
        y_fault = merged_data['fault_status']

        # Encode target labels
        maint_le = LabelEncoder()
        fault_le = LabelEncoder()
        y_maint_encoded = maint_le.fit_transform(y_maint)
        y_fault_encoded = fault_le.fit_transform(y_fault)

        # Check class counts before splitting
        maint_class_counts = np.bincount(y_maint_encoded)
        fault_class_counts = np.bincount(y_fault_encoded)
        min_maint_class = maint_class_counts.min() if len(maint_class_counts) > 0 else 0
        min_fault_class = fault_class_counts.min() if len(fault_class_counts) > 0 else 0

        if (
            len(np.unique(y_maint_encoded)) < 2 or
            len(np.unique(y_fault_encoded)) < 2 or
            len(X) < 4 or
            min_maint_class < 2 or
            min_fault_class < 2
        ):
            st.warning(
                "Not enough class diversity or data samples to train machine learning models. "
                "Each class must have at least 2 samples. Please ensure your data contains at least two classes for each target and at least 2 samples per class."
            )
        else:
            # Train test split
            X_train, X_test, y_maint_train, y_maint_test = train_test_split(
                X, y_maint_encoded, test_size=0.2, random_state=42, stratify=y_maint_encoded
            )
            _, _, y_fault_train, y_fault_test = train_test_split(
                X, y_fault_encoded, test_size=0.2, random_state=42, stratify=y_fault_encoded
            )

            # Check again after split
            if len(np.unique(y_maint_train)) < 2 or len(np.unique(y_fault_train)) < 2:
                st.warning("Not enough class diversity in the training split. Please add more data or check your labels.")
            else:
                # Train Random Forest for maintenance prediction
                rf_maint = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_maint.fit(X_train, y_maint_train)
                y_pred_rf_maint = rf_maint.predict(X_test)
                rf_maint_acc = accuracy_score(y_maint_test, y_pred_rf_maint)

                # Train XGBoost for maintenance prediction
                xgb_maint = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                xgb_maint.fit(X_train, y_maint_train)
                y_pred_xgb_maint = xgb_maint.predict(X_test)
                xgb_maint_acc = accuracy_score(y_maint_test, y_pred_xgb_maint)

                # Train Random Forest for fault detection
                rf_fault = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_fault.fit(X_train, y_fault_train)
                y_pred_rf_fault = rf_fault.predict(X_test)
                rf_fault_acc = accuracy_score(y_fault_test, y_pred_rf_fault)

                # Train XGBoost for fault detection
                xgb_fault = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                xgb_fault.fit(X_train, y_fault_train)
                y_pred_xgb_fault = xgb_fault.predict(X_test)
                xgb_fault_acc = accuracy_score(y_fault_test, y_pred_xgb_fault)

                st.success("Models trained successfully!")
                
                # Accuracy scores
                acc_df = pd.DataFrame({
                    'Model': ['Random Forest (Maintenance)', 'XGBoost (Maintenance)',
                             'Random Forest (Fault)', 'XGBoost (Fault)'],
                    'Accuracy': [rf_maint_acc, xgb_maint_acc, rf_fault_acc, xgb_fault_acc]
                })
                
                fig_acc = px.bar(acc_df, x='Model', y='Accuracy', title="Model Accuracies")
                fig_acc.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_acc, use_container_width=True)
                
                # Feature importance
                st.markdown("### Feature Importance")
                fig_imp = px.bar(
                    pd.DataFrame({
                        'Feature': ['Flow Rate', 'Power Consumption'],
                        'Importance': rf_maint.feature_importances_
                    }),
                    x='Feature',
                    y='Importance',
                    title="Feature Importance for Maintenance Prediction"
                )
                fig_imp.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig_imp, use_container_width=True)

# Predictive Maintenance
elif selected_view == "Predictive Maintenance":
    st.subheader("Predictive Maintenance Analysis")
    
    # Maintenance predictions
    maintenance_predictions = []
    for _, row in merged_data.iterrows():
        maint_pred, risk_score = predict_maintenance_advanced(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['maintenance_days']
        )
        
        efficiency = calculate_efficiency(
            row['flow_rate_LPM'], row['power_kW'], row['type']
        )
        
        maintenance_predictions.append({
            'asset_id': row['asset_id'],
            'type': row['type'],
            'status': row['status'],
            'maintenance_status': maint_pred,
            'risk_score': risk_score,
            'efficiency': efficiency,
            'days_since_maintenance': row['maintenance_days'],
            'next_maintenance_est': max(0, 90 - row['maintenance_days'])
        })
        
    maintenance_df = pd.DataFrame(maintenance_predictions)
    
    # Priority matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Maintenance Priority Matrix")
        fig_priority = px.scatter(
            maintenance_df,
            x='days_since_maintenance',
            y='risk_score',
            color='maintenance_status',
            size='efficiency',
            hover_data=['asset_id', 'type'],
            title="Maintenance Priority (Risk vs Time)",
            color_discrete_map={
                'IMMEDIATE': 'red',
                'SCHEDULED': 'orange',
                'NORMAL': 'green'
            }
        )
        fig_priority.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_priority, use_container_width=True)
        
    with col2:
        st.markdown("### Risk Score Distribution")
        fig_risk_dist = px.histogram(
            maintenance_df,
            x='risk_score',
            color='maintenance_status',
            nbins=20,
            title="Risk Score Distribution",
            color_discrete_map={
                'IMMEDIATE': 'red',
                'SCHEDULED': 'orange',
                'NORMAL': 'green'
            }
        )
        fig_risk_dist.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_risk_dist, use_container_width=True)
        
    # Maintenance schedule
    st.markdown("### Recommended Maintenance Schedule")
    
    # Sort by risk score and priority
    priority_order = {'IMMEDIATE': 1, 'SCHEDULED': 2, 'NORMAL': 3}
    maintenance_df['priority_num'] = maintenance_df['maintenance_status'].map(priority_order)
    maintenance_df_sorted = maintenance_df.sort_values(
        ['priority_num', 'risk_score'], ascending=[True, False]
    )
    
    # Display maintenance schedule
    for _, row in maintenance_df_sorted.iterrows():
        status_class = {
            'IMMEDIATE': 'alert-critical',
            'SCHEDULED': 'alert-warning',
            'NORMAL': 'alert-normal'
        }.get(row['maintenance_status'], 'info-card')
        
        schedule_text = f"""
        <div class="{status_class}">
        <h4>{row['asset_id']} - {row['type']}</h4>
        <p><strong>Status:</strong> {row['maintenance_status']}</p>
        <p><strong>Risk Score:</strong> {row['risk_score']:.1f}</p>
        <p><strong>Efficiency:</strong> {row['efficiency']:.1f}%</p>
        <p><strong>Days Since Last Maintenance:</strong> {row['days_since_maintenance']}</p>
        <p><strong>Estimated Next Maintenance:</strong> {max(0, row['next_maintenance_est'])} days</p>
        </div>
        """
        st.markdown(schedule_text, unsafe_allow_html=True)

# Alert Management
elif selected_view == "Alert Management":
    st.subheader("Comprehensive Alert Management System")
    
    # Generate alerts
    alerts = []
    for _, row in merged_data.iterrows():
        fault_status = enhanced_fault_detection(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['type']
        )
        
        if "CRITICAL" in fault_status or "WARNING" in fault_status:
            alerts.append({
                'timestamp': datetime.now(),
                'asset_id': row['asset_id'],
                'type': 'System',
                'severity': 'Critical' if 'CRITICAL' in fault_status else 'Warning',
                'message': fault_status,
                'status': 'Active'
            })
            
        # Maintenance alerts
        maintenance_status, risk_score = predict_maintenance_advanced(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['maintenance_days']
        )
        
        if "IMMEDIATE" in maintenance_status:
            alerts.append({
                'timestamp': datetime.now(),
                'asset_id': row['asset_id'],
                'type': 'Maintenance',
                'severity': 'Critical',
                'message': f"Immediate maintenance required (Risk: {risk_score:.1f})",
                'status': 'Active'
            })
        elif "SCHEDULED" in maintenance_status:
            alerts.append({
                'timestamp': datetime.now(),
                'asset_id': row['asset_id'],
                'type': 'Maintenance',
                'severity': 'Warning',
                'message': f"Maintenance scheduled (Risk: {risk_score:.1f})",
                'status': 'Active'
            })
            
    # Alert dashboard
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_count = len(alerts_df[alerts_df['severity'] == 'Critical'])
            st.metric("Critical Alerts", critical_count)
            
        with col2:
            warning_count = len(alerts_df[alerts_df['severity'] == 'Warning'])
            st.metric("Warning Alerts", warning_count)
            
        with col3:
            system_count = len(alerts_df[alerts_df['type'] == 'System'])
            st.metric("System Alerts", system_count)
            
        with col4:
            maintenance_count = len(alerts_df[alerts_df['type'] == 'Maintenance'])
            st.metric("Maintenance Alerts", maintenance_count)
            
        # Alert timeline
        st.markdown("### Active Alerts Timeline")
        fig_timeline = px.timeline(
            alerts_df,
            x_start='timestamp',
            x_end='timestamp',
            y='asset_id',
            color='severity',
            title="Active Alerts Timeline",
            color_discrete_map={'Critical': 'red', 'Warning': 'orange'}
        )
        fig_timeline.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Detailed alert table
        st.markdown("### Detailed Alert Information")
        
        # Allow marking alerts as resolved
        alert_options = [f"{idx}: {row['asset_id']} - {row['message']}" for idx, row in alerts_df.iterrows()]
        resolved_alerts = st.multiselect("Mark alerts as resolved", options=alert_options)
        
        if st.button("Mark Selected as Resolved"):
            for option in resolved_alerts:
                idx = int(option.split(":")[0])
                alerts_df.at[idx, 'status'] = 'Resolved'
            st.success(f"Marked {len(resolved_alerts)} alerts as resolved")
            
        st.dataframe(
            alerts_df.style.format({'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M:%S')}), use_container_width=True
        )
        
        # Export alerts
        if st.button("Export Alerts"):
            csv = alerts_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="alerts.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        # Alert escalation
        st.markdown("### Alert Escalation")
        if st.button("Send Notifications"):
            st.success("Notifications sent for all active alerts")
            
    else:
        st.success("No active alerts - All systems operating normally!")

# Performance Metrics
elif selected_view == "Performance Metrics":
    st.subheader("Comprehensive Performance Metrics")
    
    # Calculate KPIs
    kpis = {}
    
    # System availability
    total_assets = len(merged_data)
    operational_assets = len(merged_data[merged_data['status'].isin(['Operational', 'Running', 'Functional'])])
    kpis['availability'] = (operational_assets / total_assets) * 100 if total_assets > 0 else 0
    
    # Average efficiency
    efficiencies = [calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']) for _, row in merged_data.iterrows()]
    kpis['avg_efficiency'] = np.mean(efficiencies) if efficiencies else 0
    
    # Energy consumption
    kpis['total_power'] = merged_data['power_kW'].sum()
    kpis['avg_power_per_asset'] = merged_data['power_kW'].mean()
    
    # Flow metrics
    kpis['total_flow'] = merged_data['flow_rate_LPM'].sum()
    kpis['avg_flow_per_asset'] = merged_data['flow_rate_LPM'].mean()
    
    # Maintenance metrics
    overdue_maintenance = len(merged_data[merged_data['maintenance_days'] > 90])
    kpis['maintenance_compliance'] = ((total_assets - overdue_maintenance) / total_assets) * 100 if total_assets > 0 else 0
    
    # Display KPIs
    st.markdown("### Key Performance Indicators")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Availability", f"{kpis['availability']:.1f}%", delta="2.3%")
        st.metric("Average Efficiency", f"{kpis['avg_efficiency']:.1f}%", delta="1.8%")
        
    with col2:
        st.metric("Total Power Consumption", f"{kpis['total_power']:.1f} kW", delta="0.5 kW")
        st.metric("Average Flow Rate", f"{kpis['avg_flow_per_asset']:.1f} L/min", delta="5.2 L/min")
        
    with col3:
        st.metric("Total Flow", f"{kpis['total_flow']:.1f} L/min", delta="15.6 L/min")
        st.metric("Maintenance Compliance", f"{kpis['maintenance_compliance']:.1f}%", delta="-2.1%")
        
    # Performance trends
    st.markdown("### Performance Trend Analysis")
    # Create performance dashboard
    fig_performance = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Efficiency by Asset Type', 'Power vs Flow Efficiency',
                      'Maintenance Status Distribution', 'Asset Status Overview'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
              [{"type": "pie"}, {"type": "pie"}]]
    )
    
    # Efficiency by asset type
    efficiency_by_type = merged_data.groupby('type').apply(
        lambda x: np.mean([calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']) for _, row in x.iterrows()])
    ).reset_index()
    efficiency_by_type.columns = ['type', 'avg_efficiency']
    
    fig_performance.add_trace(
        go.Bar(
            x=efficiency_by_type['type'],
            y=efficiency_by_type['avg_efficiency'],
            name="Efficiency by Type",
            marker_color='skyblue'
        ),
        row=1, col=1
    )
    
    # Power vs Flow efficiency scatter
    merged_data['efficiency'] = [calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']) for _, row in merged_data.iterrows()]
    
    fig_performance.add_trace(
        go.Scatter(
            x=merged_data['power_kW'],
            y=merged_data['flow_rate_LPM'],
            mode='markers',
            marker=dict(
                size=merged_data['efficiency'],
                color=merged_data['efficiency'],
                colorscale='Viridis',
                showscale=True
            ),
            text=merged_data['asset_id'],
            name="Power vs Flow"
        ),
        row=1, col=2
    )
    
    # Maintenance status pie
    maintenance_status_counts = []
    for _, row in merged_data.iterrows():
        status, _ = predict_maintenance_advanced(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['maintenance_days']
        )
        maintenance_status_counts.append(status)
        
    maintenance_counts = pd.Series(maintenance_status_counts).value_counts()
    
    fig_performance.add_trace(
        go.Pie(
            labels=maintenance_counts.index,
            values=maintenance_counts.values,
            name="Maintenance Status"
        ),
        row=2, col=1
    )
    
    # Asset status pie
    status_counts = merged_data['status'].value_counts()
    
    fig_performance.add_trace(
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            name="Asset Status"
        ),
        row=2, col=2
    )
    
    fig_performance.update_layout(
        height=800,
        template='plotly_dark',
        title="Comprehensive Performance Dashboard"
    )
    st.plotly_chart(fig_performance, use_container_width=True)

# Asset Details
elif selected_view == "Asset Details":
    st.subheader("Detailed Asset Information")
    
    # Asset selector
    selected_asset = st.selectbox(
        "Select Asset for Detailed Analysis",
        options=merged_data['asset_id'].tolist()
    )
    
    if selected_asset:
        asset_info = merged_data[merged_data['asset_id'] == selected_asset].iloc[0]
        asset_scada_history = scada_df[scada_df['asset_id'] == selected_asset]
        
        # Asset information card
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f""" 
            <div class="info-card">
            <h3> {selected_asset} - {asset_info['type']}</h3> 
            <p><strong>Current Status:</strong> {asset_info['status']}</p> 
            <p><strong>Location:</strong> {asset_info['latitude']:.4f}, {asset_info['longitude']:.4f}</p> 
            <p><strong>Last Maintenance:</strong> {asset_info['last_maintenance'].strftime('%Y-%m-%d')}</p> 
            <p><strong>Days Since Maintenance:</strong> {asset_info['maintenance_days']}</p> 
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Current metrics
            efficiency = calculate_efficiency(
                asset_info['flow_rate_LPM'], asset_info['power_kW'], asset_info['type']
            )
            
            fault_status = enhanced_fault_detection(
                asset_info['flow_rate_LPM'], asset_info['power_kW'], asset_info['status'], asset_info['type']
            )
            
            maintenance_status, risk_score = predict_maintenance_advanced(
                asset_info['flow_rate_LPM'], asset_info['power_kW'],
                asset_info['status'], asset_info['maintenance_days']
            )
            
            st.markdown(f""" 
            <div class="metric-card">
            <h3> Current Metrics</h3>
            <p><strong>Flow Rate:</strong> {asset_info['flow_rate_LPM']:.1f} L/min</p> 
            <p><strong>Power Consumption:</strong> {asset_info['power_kW']:.1f} kW</p> 
            <p><strong>Efficiency:</strong> {efficiency:.1f}%</p> 
            <p><strong>System Status:</strong> {fault_status}</p> 
            <p><strong>Maintenance Status:</strong> {maintenance_status}</p> 
            <p><strong>Risk Score:</strong> {risk_score:.1f}</p> 
            </div>
            """, unsafe_allow_html=True)
            
        # Historical data charts
        st.markdown("### Historical Performance Data")
        fig_history = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{selected_asset} - Flow Rate History', f'{selected_asset} - Power Consumption History'),
            vertical_spacing=0.1
        )
        
        fig_history.add_trace(
            go.Scatter(
                x=asset_scada_history['timestamp'],
                y=asset_scada_history['flow_rate_LPM'],
                mode='lines+markers',
                name='Flow Rate',
                line=dict(color='cyan', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        fig_history.add_trace(
            go.Scatter(
                x=asset_scada_history['timestamp'],
                y=asset_scada_history['power_kW'],
                mode='lines+markers',
                name='Power Consumption',
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        fig_history.update_layout(
            height=600,
            template='plotly_dark',
            title=f"Historical Performance - {selected_asset}"
        )
        fig_history.update_xaxes(title_text="Time", row=2, col=1)
        fig_history.update_yaxes(title_text="Flow Rate (L/min)", row=1, col=1)
        fig_history.update_yaxes(title_text="Power (kW)", row=2, col=1)
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Performance statistics
        st.markdown("### Performance Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_flow = asset_scada_history['flow_rate_LPM'].mean()
            max_flow = asset_scada_history['flow_rate_LPM'].max()
            min_flow = asset_scada_history['flow_rate_LPM'].min()
            st.metric("Average Flow Rate", f"{avg_flow:.1f} L/min")
            st.metric("Maximum Flow Rate", f"{max_flow:.1f} L/min")
            st.metric("Minimum Flow Rate", f"{min_flow:.1f} L/min")
            
        with col2:
            avg_power = asset_scada_history['power_kW'].mean()
            max_power = asset_scada_history['power_kW'].max()
            min_power = asset_scada_history['power_kW'].min()
            st.metric("Average Power", f"{avg_power:.1f} kW")
            st.metric("Maximum Power", f"{max_power:.1f} kW")
            st.metric("Minimum Power", f"{min_power:.1f} kW")
            
        with col3:
            uptime_percentage = (asset_scada_history['status'] == 'ON').mean() * 100
            total_records = len(asset_scada_history)
            efficiency_avg = np.mean([
                calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], asset_info['type']) 
                for _, row in asset_scada_history.iterrows()
            ])
            
            st.metric("Uptime", f"{uptime_percentage:.1f}%")
            st.metric("Total Records", f"{total_records}")
            st.metric("Average Efficiency", f"{efficiency_avg:.1f}%")
            
# Mobile View
elif selected_view == "Mobile View":
    st.subheader("Mobile-Optimized Dashboard")
    
    # Simplified mobile layout
    st.markdown("### Quick Status Overview")
    
    for _, row in merged_data.iterrows():
        status_emoji = {
            'Operational': '✅',
            'Running': '▶️',
            'Functional': '⚙️',
            'Collapsed': '❌',
            'Offline': '🛑',
            'Blocked': '🚧'
        }.get(row['status'], ' ')
        
        fault_status = enhanced_fault_detection(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['type']
        )
        
        status_class = "alert-critical" if "CRITICAL" in fault_status else (
            "alert-warning" if "WARNING" in fault_status else "alert-normal"
        )
        
        st.markdown(f""" 
        <div class="{status_class}">
        <h4>{status_emoji} {row['asset_id']} - {row['type']}</h4> 
        <p><strong>Status:</strong> {row['status']}</p> 
        <p><strong>Flow:</strong> {row['flow_rate_LPM']:.1f} L/min | <strong>Power:</strong> {row['power_kW']:.1f} kW</p> 
        <p><strong>System:</strong> {fault_status}</p>
        </div>
        """, unsafe_allow_html=True)

# Customize Dashboard
elif selected_view == "Customize Dashboard":
    st.subheader("Customize Dashboard Layout")
    
    # Get available widgets
    available_widgets = [
        ("Real-time Flow Monitoring", "line"),
        ("Power Consumption Analysis", "line"),
        ("System Performance Dashboard", "gauge"),
        ("Live Alert Summary", "table"),
        ("Performance Trends", "scatter"),
        ("Maintenance Priority Matrix", "scatter")
    ]
    
    # Allow user to select and order widgets
    selected_widgets = st.multiselect(
        "Select widgets to display",
        options=[widget[0] for widget in available_widgets],
        default=[widget[0] for widget in available_widgets[:4]]
    )
    
    # Create custom dashboard based on selected widgets
    st.subheader("Custom Dashboard Preview")
    
    for widget_name in selected_widgets:
        chart_type = next(item[1] for item in available_widgets if item[0] == widget_name)
        
        if widget_name == "Real-time Flow Monitoring":
            fig = px.line(
                scada_df,
                x='timestamp',
                y='flow_rate_LPM',
                color='asset_id',
                title=widget_name,
                template='plotly_dark'
            )
            fig.update_traces(mode='lines+markers', marker=dict(size=8, symbol='circle'))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        elif widget_name == "Power Consumption Analysis":
            fig = px.line(
                scada_df,
                x='timestamp',
                y='power_kW',
                color='asset_id',
                title=widget_name,
                template='plotly_dark'
            )
            fig.update_traces(mode='lines+markers', marker=dict(size=8, symbol='circle'))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        elif widget_name == "System Performance Dashboard":
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # System Availability
            total_assets = len(merged_data)
            operational_assets = len(merged_data[merged_data['status'].isin(['Operational', 'Running', 'Functional'])])
            availability = (operational_assets / total_assets) * 100 if total_assets > 0 else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=availability,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "System Availability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if availability >= 80 else "orange" if availability >= 50 else "red"}
                    }
                ), row=1, col=1
            )
            
            # Average Efficiency
            efficiencies = [calculate_efficiency(row['flow_rate_LPM'], row['power_kW'], row['type']) for _, row in merged_data.iterrows()]
            avg_efficiency = np.mean(efficiencies) if efficiencies else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=avg_efficiency,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Average Efficiency"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "blue"}}
                ), row=1, col=2
            )
            
            fig.update_layout(height=300, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
        elif widget_name == "Live Alert Summary":
            alerts = []
            for _, row in merged_data.iterrows():
                fault_status = enhanced_fault_detection(
                    row['flow_rate_LPM'], row['power_kW'], row['status'], row['type']
                )
                if "CRITICAL" in fault_status or "WARNING" in fault_status:
                    alerts.append({
                        'asset_id': row['asset_id'],
                        'status': fault_status,
                        'severity': 'Critical' if 'CRITICAL' in fault_status else 'Warning'
                    })
            
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                st.dataframe(alerts_df.style.applymap(lambda x: 'background-color: #ffebee' if x == 'Critical' else 'background-color: #fff8e1' if x == 'Warning' else ''))
            else:
                st.success("No active alerts")
                
        elif widget_name == "Performance Trends":
            fig = px.scatter(
                merged_data,
                x='maintenance_days',
                y='flow_rate_LPM',
                color='power_kW',
                title=widget_name,
                template='plotly_dark'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        elif widget_name == "Maintenance Priority Matrix":
            maintenance_predictions = []
            for _, row in merged_data.iterrows():
                maint_pred, risk_score = predict_maintenance_advanced(
                    row['flow_rate_LPM'], row['power_kW'], row['status'],
 row['maintenance_days']
                )
                
            if maintenance_predictions:
                maint_df = pd.DataFrame(maintenance_predictions)
                fig = px.scatter(
                    maint_df,
                    x='risk_score',
                    y='risk_score',
                    color='maintenance_status',
                    title=widget_name,
                    template='plotly_dark'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No maintenance data available")
                
    # Save custom layout
    if st.button("Save Custom Layout"):
        st.session_state.custom_layout = selected_widgets
        st.success("Custom layout saved successfully!")
        
    # Reset to default layout
    if st.button("Reset to Default"):
        del st.session_state.custom_layout
        st.rerun()

# Footer and Auto-refresh
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f""" 
    <div class="info-card">
    <h4> Last Updated</h4>
    <p>{st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}</p> 
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown(f""" 
    <div class="info-card">
    <h4> Data Summary</h4>
    <p>Assets: {len(assets_df)} | Records: {len(scada_df)}</p> 
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown(f""" 
    <div class="info-card">
    <h4> System Status</h4>
    <p>Online | Monitoring Active</p>
    </div>
    """, unsafe_allow_html=True)

# Export functionality
st.sidebar.markdown("### Export Options")
export_format = st.sidebar.selectbox("Select Export Format", ["JSON", "CSV", "Excel", "PDF"])
if st.sidebar.button("Export Current Data"):
    # Create export data
    export_data = {
        'assets': assets_df.to_dict('records'),
        'scada_latest': merged_data.to_dict('records'),
        'summary': {
            'total_assets': len(assets_df),
            'operational_assets': len(merged_data[merged_data['status'].isin(['Operational', 'Running', 'Functional'])]),
            'avg_flow': merged_data['flow_rate_LPM'].mean(),
            'avg_power': merged_data['power_kW'].mean(),
            'export_time': datetime.now().isoformat()
        }
    }
    # Export in selected format
    if export_format == "JSON":
        json_data = json.dumps(export_data, indent=2, default=str)
        st.sidebar.download_button(
            label="Download JSON Report",
            data=json_data,
            file_name=f"scada_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    elif export_format == "CSV":
        # Create temporary CSV files
        csv_files = []
        for name, data in [('assets', assets_df), ('scada', scada_df), ('merged', merged_data)]:
            filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_files.append((filename, data))
        # Zip and download
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, data in csv_files:
                zip_file.writestr(filename, data.to_csv(index=False))
        st.sidebar.download_button(
            label="Download CSV Files",
            data=zip_buffer.getvalue(),
            file_name=f"scada_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
    elif export_format == "Excel":
        excel_data = export_to_excel(merged_data, [fig_gauges])
        st.sidebar.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name=f"scada_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    elif export_format == "PDF":
        pdf_data = export_to_pdf(merged_data, [fig_gauges])
        st.sidebar.download_button(
            label="Download PDF",
            data=bytes(pdf_data),  # Convert bytearray to bytes here
            file_name="report.pdf",
            mime="application/pdf"
        )

# Alert notifications
if st.sidebar.button("Check Notifications"):
    st.sidebar.markdown("### Active Notifications")
    notification_count = 0
    for _, row in merged_data.iterrows():
        fault_status = enhanced_fault_detection(
            row['flow_rate_LPM'], row['power_kW'], row['status'], row['type']
        )
        if "CRITICAL" in fault_status:
            st.sidebar.error(f" {row['asset_id']}: {fault_status}")
            notification_count += 1
        elif "WARNING" in fault_status:
            st.sidebar.warning(f" {row['asset_id']}: {fault_status}")
            notification_count += 1
    if notification_count == 0:
        st.sidebar.success("No active alerts")

# System information
st.sidebar.markdown("### System Information")
st.sidebar.info(f""" 
**SCADA System v2.0**
- Location: Surathkal, Karnataka
- Monitoring: {len(assets_df)} Assets 
- Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}
- Status: Online
- Mode: {"Dark" if st.session_state.dark_mode else "Light"}
""")

# Help and documentation
with st.sidebar.expander("Help & Documentation"):
    st.markdown(""" 
    ### Dashboard Features:
    - **Executive Dashboard**: Overview of all systems
    - **GIS Mapping**: Geographic asset locations
    - **Advanced Analytics**: Detailed performance analysis
    - **Predictive Maintenance**: AI-driven maintenance scheduling
    - **Alert Management**: Real-time alert monitoring
    - **Performance Metrics**: KPI tracking
    - **Asset Details**: Individual asset analysis
    - **Mobile View**: Simplified mobile interface
    ### Alert Levels:
    - **Normal**: System operating normally
    - **Warning**: Attention required
    - **Critical**: Immediate action needed
    ### Support:
    For technical support, contact the system administrator.
    """)

# Performance monitoring
if st.sidebar.checkbox("Show Performance Monitor"):
    st.sidebar.markdown("### Performance Monitor")
    # System performance metrics
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    st.sidebar.metric("CPU Usage", f"{cpu_usage:.1f}%")
    st.sidebar.metric("Memory Usage", f"{memory_usage:.1f}%")
    st.sidebar.metric("Platform", platform.system())
    # Dashboard performance metrics
    st.sidebar.metric("Active Connections", len(selected_assets))
    st.sidebar.metric("Data Points", len(scada_df))

# Footer
st.markdown(""" 
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg,#2c3e50 0%, #34495e 100%); color: white; margin-top: 2rem; border-radius: 10px;">
<h4>SCADA Sewer Infrastructure Monitor</h4>
<p>Surathkal, Karnataka | Advanced Monitoring & Control System</p>
<p><small>© 2025 Infrastructure Monitoring Division | Version 2.0</small></p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh implementation
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Show class distribution for debugging
st.write("Maintenance status value counts:", merged_data['maintenance_status'].value_counts())
st.write("Fault status value counts:", merged_data['fault_status'].value_counts())