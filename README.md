 **SCADA Sewer Infrastructure Monitor**

A powerful dashboard built using Streamlit , Plotly , and Folium to monitor sewer infrastructure in real-time with GIS mapping, predictive maintenance, fault detection, and alert management capabilities.

**Live Demo**
https://vasavijoshi-udal-scada-dashboard-zqftyo.streamlit.app/

**Overview**
This system provides real-time monitoring of sewer assets such as Wet Wells , Sewage Treatment Plants (STP) , and other pumping stations. It combines SCADA data , GIS asset locations, and public complaints to deliver actionable insights through an interactive dashboard.

Key components:

        ◦ Live flow rate and power consumption tracking
        ◦ Predictive maintenance scheduling
        ◦ Fault detection and alerting  
        ◦ GIS-based asset mapping 
        ◦ Customizable dashboard views
        ◦ Data export options (PDF, JSON, CSV)


**Features**

1. Executive Dashboard - High-level overview of all assets and system health
2. GIS Asset Mapping - Interactive map showing asset locations and statuses
3. Advanced Analytics - Correlation analysis, anomaly detection, energy efficiency reports
4. Predictive Maintenance - AI-driven maintenance scheduling and risk scoring
5. Alert Management - Real-time alerts and escalation handling
6. Performance Metrics - Key performance indicators (KPIs) and trends
7. Asset Details - In-depth view of individual asset history and metrics
8. Mobile View - Simplified layout optimized for mobile devices
9. Customize Dashboard - Select and arrange widgets to suit your needs


**Core Functionalities**

1. Fault Detection : Detects critical or warning states based on sensor data
2. Anomaly Detection : Uses Isolation Forest for identifying abnormal behavior
3. Machine Learning Models : Trains models like Random Forest and XGBoost for classification
4. Dynamic Visualizations : Plotly charts and Folium maps
5. GIS Integration : Shows asset locations and overlays complaint points
6. Export Options : Export data in JSON, CSV, Excel, or PDF format
7. Auto-refresh : Configurable refresh interval for live monitoring


**Technology Stack**
1. Streamlit - Frontend UI framework
2. Plotly Express / Graph Objects - Interactive visualizations
3. Folium - Interactive maps with Leaflet.js
4. Pandas / NumPy - Data manipulation and analysis
5. FPDF - PDF report generation
6. Base64 / ZIP - File encoding and packaging
7. Psutil - System resource monitoring



**How to Run**
1. pip install streamlit pandas numpy plotly folium streamlit-folium matplotlib scikit-learn xgboost fpdf2 psutil
2. streamlit scada_dashboard.py


**UI Highlights**
1. Dark/Light Mode Toggle for user preference
2. Gauge Charts for KPI visualization
3. Interactive Plots for time-series and comparative analysis
4. Heatmaps and Marker Clusters for spatial distribution
5. Alert Cards with severity-based styling (Critical, Warning, Normal)


**Live Demo**
https://vasavijoshi-udal-scada-dashboard-zqftyo.streamlit.app/
