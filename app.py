import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from PIL import Image, ImageDraw, ImageFont
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64
import folium
from streamlit_folium import folium_static
import geocoder
import time

# Page configuration
st.set_page_config(
    page_title="🛡️ Marque AI - Crime Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .alert-red {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .alert-yellow {
        background-color: #fff8e1;
        border: 2px solid #ff9800;
        color: #f57c00;
    }
    .alert-green {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'incidents' not in st.session_state:
    st.session_state.incidents = []
if 'current_location' not in st.session_state:
    st.session_state.current_location = None

# Kenya administrative regions data
KENYA_COUNTIES = [
    "Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita-Taveta", "Garissa",
    "Wajir", "Mandera", "Marsabit", "Isiolo", "Meru", "Tharaka-Nithi", "Embu",
    "Kitui", "Machakos", "Makueni", "Nyandarua", "Nyeri", "Kirinyaga", "Murang'a",
    "Kiambu", "Turkana", "West Pokot", "Samburu", "Trans-Nzoia", "Uasin Gishu",
    "Elgeyo-Marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru", "Narok", "Kajiado",
    "Kericho", "Bomet", "Kakamega", "Vihiga", "Bungoma", "Busia", "Siaya", "Kisumu",
    "Homa Bay", "Migori", "Kisii", "Nyamira", "Nairobi"
]

# Weapon classification and severity mapping
WEAPON_SEVERITY = {
    "gun": {"severity": "SERIOUS-URGENT", "alert_level": "🔴 Red", "risk": "High"},
    "pistol": {"severity": "SERIOUS-URGENT", "alert_level": "🔴 Red", "risk": "High"},
    "rifle": {"severity": "SERIOUS-URGENT", "alert_level": "🔴 Red", "risk": "High"},
    "knife": {"severity": "SERIOUS", "alert_level": "🟡 Yellow", "risk": "Medium"},
    "machete": {"severity": "SERIOUS", "alert_level": "🟡 Yellow", "risk": "Medium"},
    "sword": {"severity": "SERIOUS", "alert_level": "🟡 Yellow", "risk": "Medium"},
    "stick": {"severity": "LOW", "alert_level": "🟢 Green", "risk": "Low"},
    "stone": {"severity": "LOW", "alert_level": "🟢 Green", "risk": "Low"},
    "club": {"severity": "LOW", "alert_level": "🟢 Green", "risk": "Low"}
}

@st.cache_resource
def load_models():
    """Load AI models for object detection and sentiment analysis"""
    try:
        # Object detection model (using a general object detection model)
        object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        
        # Sentiment analysis model
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        return object_detector, sentiment_analyzer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def detect_weapons(image, object_detector):
    """Detect weapons in the uploaded image"""
    if object_detector is None:
        return []
    
    try:
        # Convert PIL image to format expected by the model
        results = object_detector(image)
        
        # Filter for potential weapons (this is a simplified approach)
        weapon_keywords = ["knife", "gun", "pistol", "rifle", "stick", "baseball bat"]
        detected_weapons = []
        
        for result in results:
            label = result['label'].lower()
            for weapon in weapon_keywords:
                if weapon in label or any(w in label for w in weapon.split()):
                    weapon_type = weapon
                    if weapon_type in WEAPON_SEVERITY:
                        detected_weapons.append({
                            'weapon': weapon_type,
                            'confidence': result['score'],
                            'box': result['box'],
                            'severity': WEAPON_SEVERITY[weapon_type]['severity'],
                            'alert_level': WEAPON_SEVERITY[weapon_type]['alert_level'],
                            'risk': WEAPON_SEVERITY[weapon_type]['risk']
                        })
        
        return detected_weapons
    except Exception as e:
        st.error(f"Error in weapon detection: {str(e)}")
        return []

def analyze_threat_level(detected_weapons, sentiment_analyzer):
    """Analyze overall threat level based on detected weapons and context"""
    if not detected_weapons:
        return "🟢 Green", "LOW", "No immediate threat detected"
    
    # Determine highest risk level
    risk_levels = [weapon['risk'] for weapon in detected_weapons]
    
    if 'High' in risk_levels:
        return "🔴 Red", "SERIOUS-URGENT", "Immediate action required - High-risk weapons detected"
    elif 'Medium' in risk_levels:
        return "🟡 Yellow", "SERIOUS", "Caution advised - Medium-risk weapons detected"
    else:
        return "🟢 Green", "LOW", "Monitor situation - Low-risk items detected"

def get_current_location():
    """Get current location (simplified - in real implementation would use GPS)"""
    try:
        # Using a mock location for demonstration
        # In real implementation, you would use browser geolocation API
        return {
            'latitude': -1.2921,  # Nairobi coordinates
            'longitude': 36.8219,
            'address': 'Nairobi, Kenya'
        }
    except:
        return None

def draw_detection_boxes(image, detections):
    """Draw bounding boxes around detected weapons"""
    if not detections:
        return image
    
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    for detection in detections:
        box = detection['box']
        weapon = detection['weapon']
        confidence = detection['confidence']
        alert_level = detection['alert_level']
        
        # Color based on risk level
        if '🔴' in alert_level:
            color = 'red'
        elif '🟡' in alert_level:
            color = 'orange'
        else:
            color = 'green'
        
        # Draw rectangle
        draw.rectangle([
            (box['xmin'], box['ymin']),
            (box['xmax'], box['ymax'])
        ], outline=color, width=3)
        
        # Draw label
        label = f"{weapon} ({confidence:.2f})"
        if font:
            draw.text((box['xmin'], box['ymin'] - 20), label, fill=color, font=font)
        else:
            draw.text((box['xmin'], box['ymin'] - 20), label, fill=color)
    
    return image

def generate_incident_report(incident_data):
    """Generate PDF report for incident"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.darkblue,
        alignment=1,  # Center alignment
        spaceAfter=30
    )
    
    # Title
    title = Paragraph("🛡️ AI - INCIDENT REPORTING", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # Incident details table
    incident_info = [
        ['Incident ID:', incident_data['id']],
        ['Date & Time:', incident_data['timestamp']],
        ['Alert Level:', incident_data['alert_level']],
        ['Severity:', incident_data['severity']],
        ['Location:', f"{incident_data['location']['county']}, {incident_data['location']['ward']}"],
        ['Coordinates:', f"Lat: {incident_data['location']['latitude']}, Lon: {incident_data['location']['longitude']}"],
        ['Detected Weapons:', ', '.join([w['weapon'] for w in incident_data['detected_weapons']])],
        ['Risk Assessment:', incident_data['risk_assessment']],
        ['Status:', incident_data['status']]
    ]
    
    table = Table(incident_info, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    # Recommendations
    recommendations = Paragraph("<b>Recommended Actions:</b><br/>" + incident_data['recommendations'], styles['Normal'])
    elements.append(recommendations)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def create_incident_map(incidents):
    """Create a map showing incident locations"""
    if not incidents:
        return None
    
    # Create a map centered on Kenya
    m = folium.Map(location=[-1.2921, 36.8219], zoom_start=6)
    
    for incident in incidents:
        location = incident['location']
        
        # Color based on alert level
        if '🔴' in incident['alert_level']:
            color = 'red'
        elif '🟡' in incident['alert_level']:
            color = 'orange'
        else:
            color = 'green'
        
        folium.CircleMarker(
            location=[location['latitude'], location['longitude']],
            radius=8,
            popup=f"ID: {incident['id']}<br>Alert: {incident['alert_level']}<br>Time: {incident['timestamp']}",
            color=color,
            fill=True,
            fillColor=color
        ).add_to(m)
    
    return m

# Main application
def main():
    st.markdown('<h1 class="main-header">🛡️ AI - Smart Crime Detection System</h1>', unsafe_allow_html=True)
    
    # Load models
    object_detector, sentiment_analyzer = load_models()
    
    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox("Select Page", 
                               ["🚨 Report Incident", "📊 Dashboard", "🗺️ Incident Map", "📋 Incident History"])
    
    if page == "🚨 Report Incident":
        st.header("🚨 Report New Incident")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                # Analyze button
                if st.button("🔍 Analyze Threat", type="primary"):
                    with st.spinner("Analyzing image for threats..."):
                        # Detect weapons
                        detected_weapons = detect_weapons(image, object_detector)
                        
                        # Analyze threat level
                        alert_level, severity, risk_assessment = analyze_threat_level(detected_weapons, sentiment_analyzer)
                        
                        # Display results
                        if detected_weapons:
                            # Draw detection boxes
                            annotated_image = draw_detection_boxes(image.copy(), detected_weapons)
                            st.image(annotated_image, caption="Detected Threats", use_container_width=True)
                            
                            # Alert box
                            if "🔴" in alert_level:
                                st.markdown(f'<div class="alert-box alert-red">⚠️ {alert_level} - {risk_assessment}</div>', unsafe_allow_html=True)
                            elif "🟡" in alert_level:
                                st.markdown(f'<div class="alert-box alert-yellow">⚠️ {alert_level} - {risk_assessment}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="alert-box alert-green">✅ {alert_level} - {risk_assessment}</div>', unsafe_allow_html=True)
                            
                            # Detection details
                            st.subheader("🔍 Detection Details")
                            for weapon in detected_weapons:
                                st.write(f"**{weapon['weapon'].title()}** - Confidence: {weapon['confidence']:.2f} - Risk: {weapon['risk']}")
                        else:
                            st.success("✅ No weapons detected in the image")
                            alert_level, severity, risk_assessment = "🟢 Green", "LOW", "No immediate threat detected"
                        
                        # Store results in session state for location input
                        st.session_state.analysis_results = {
                            'detected_weapons': detected_weapons,
                            'alert_level': alert_level,
                            'severity': severity,
                            'risk_assessment': risk_assessment,
                            'image': image
                        }
        
        # Location input section
        if 'analysis_results' in st.session_state:
            st.subheader("📍 Location Information")
            
            location_method = st.radio("Location Input Method", ["📍 Manual Input", "🗺️ GPS (Simulated)"])
            
            if location_method == "📍 Manual Input":
                col1, col2 = st.columns(2)
                
                with col1:
                    county = st.selectbox("County", KENYA_COUNTIES)
                    ward = st.text_input("Ward")
                    latitude = st.number_input("Latitude", value=-1.2921, format="%.6f")
                
                with col2:
                    sub_county = st.text_input("Sub-county")
                    location = st.text_input("Location/Area")
                    longitude = st.number_input("Longitude", value=36.8219, format="%.6f")
                
                location_data = {
                    'county': county,
                    'sub_county': sub_county,
                    'ward': ward,
                    'location': location,
                    'latitude': latitude,
                    'longitude': longitude
                }
            
            else:
                # Simulated GPS
                st.info("🗺️ Using simulated GPS location (Nairobi)")
                location_data = {
                    'county': 'Nairobi',
                    'sub_county': 'Westlands',
                    'ward': 'Parklands',
                    'location': 'Nairobi CBD',
                    'latitude': -1.2921,
                    'longitude': 36.8219
                }
            
            # Submit incident
            if st.button("📝 Submit Incident Report", type="primary"):
                # Create incident record
                incident_id = f"INC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                incident_data = {
                    'id': incident_id,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'detected_weapons': st.session_state.analysis_results['detected_weapons'],
                    'alert_level': st.session_state.analysis_results['alert_level'],
                    'severity': st.session_state.analysis_results['severity'],
                    'risk_assessment': st.session_state.analysis_results['risk_assessment'],
                    'location': location_data,
                    'status': 'Active',
                    'recommendations': get_recommendations(st.session_state.analysis_results['severity'])
                }
                
                # Add to session state
                st.session_state.incidents.append(incident_data)
                
                # Success message
                st.success(f"✅ Incident {incident_id} successfully reported!")
                
                # Generate PDF report
                pdf_buffer = generate_incident_report(incident_data)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"incident_report_{incident_id}.pdf",
                    mime="application/pdf"
                )
                
                # Clear analysis results
                del st.session_state.analysis_results
    
    elif page == "📊 Dashboard":
        st.header("📊 Crime Detection Dashboard")
        
        if st.session_state.incidents:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Incidents", len(st.session_state.incidents))
            
            with col2:
                high_risk = sum(1 for i in st.session_state.incidents if '🔴' in i['alert_level'])
                st.metric("High Risk Incidents", high_risk)
            
            with col3:
                medium_risk = sum(1 for i in st.session_state.incidents if '🟡' in i['alert_level'])
                st.metric("Medium Risk Incidents", medium_risk)
            
            with col4:
                low_risk = sum(1 for i in st.session_state.incidents if '🟢' in i['alert_level'])
                st.metric("Low Risk Incidents", low_risk)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Alert level distribution
                alert_counts = {}
                for incident in st.session_state.incidents:
                    level = incident['alert_level']
                    alert_counts[level] = alert_counts.get(level, 0) + 1
                
                fig = px.pie(
                    values=list(alert_counts.values()),
                    names=list(alert_counts.keys()),
                    title="Alert Level Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Incidents by county
                county_counts = {}
                for incident in st.session_state.incidents:
                    county = incident['location']['county']
                    county_counts[county] = county_counts.get(county, 0) + 1
                
                fig = px.bar(
                    x=list(county_counts.keys()),
                    y=list(county_counts.values()),
                    title="Incidents by County"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent incidents table
            st.subheader("📋 Recent Incidents")
            df = pd.DataFrame(st.session_state.incidents)
            st.dataframe(df[['id', 'timestamp', 'alert_level', 'severity', 'status']], use_container_width=True)
        
        else:
            st.info("📊 No incidents reported yet. Use the Report Incident page to add data.")
    
    elif page == "🗺️ Incident Map":
        st.header("🗺️ Incident Location Map")
        
        if st.session_state.incidents:
            incident_map = create_incident_map(st.session_state.incidents)
            if incident_map:
                folium_static(incident_map, width=700, height=500)
            
            # Map legend
            st.markdown("""
            **Map Legend:**
            - 🔴 Red: High Risk (Serious-Urgent)
            - 🟡 Orange: Medium Risk (Serious)
            - 🟢 Green: Low Risk (Monitor)
            """)
        else:
            st.info("🗺️ No incidents to display on map. Report incidents to see them here.")
    
    elif page == "📋 Incident History":
        st.header("📋 Incident History & Reports")
        
        if st.session_state.incidents:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_county = st.selectbox("Filter by County", ["All"] + KENYA_COUNTIES)
            
            with col2:
                filter_severity = st.selectbox("Filter by Severity", ["All", "SERIOUS-URGENT", "SERIOUS", "LOW"])
            
            with col3:
                filter_status = st.selectbox("Filter by Status", ["All", "Active", "Resolved", "Under Investigation"])
            
            # Apply filters
            filtered_incidents = st.session_state.incidents.copy()
            
            if filter_county != "All":
                filtered_incidents = [i for i in filtered_incidents if i['location']['county'] == filter_county]
            
            if filter_severity != "All":
                filtered_incidents = [i for i in filtered_incidents if i['severity'] == filter_severity]
            
            if filter_status != "All":
                filtered_incidents = [i for i in filtered_incidents if i['status'] == filter_status]
            
            # Display filtered incidents
            st.subheader(f"📊 Showing {len(filtered_incidents)} incidents")
            
            for incident in filtered_incidents:
                with st.expander(f"🔍 {incident['id']} - {incident['alert_level']} - {incident['timestamp']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Severity:** {incident['severity']}")
                        st.write(f"**Location:** {incident['location']['county']}, {incident['location']['ward']}")
                        st.write(f"**Status:** {incident['status']}")
                        st.write(f"**Risk Assessment:** {incident['risk_assessment']}")
                    
                    with col2:
                        if incident['detected_weapons']:
                            st.write("**Detected Weapons:**")
                            for weapon in incident['detected_weapons']:
                                st.write(f"- {weapon['weapon'].title()} (Confidence: {weapon['confidence']:.2f})")
                        else:
                            st.write("**No weapons detected**")
                    
                    st.write(f"**Recommendations:** {incident['recommendations']}")
                    
                    # Download PDF for this incident
                    pdf_buffer = generate_incident_report(incident)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"incident_report_{incident['id']}.pdf",
                        mime="application/pdf",
                        key=f"download_{incident['id']}"
                    )
            
            # Export all data
            if st.button("📥 Export All Data (JSON)"):
                json_data = json.dumps(st.session_state.incidents, indent=2)
                st.download_button(
                    label="📥 Download JSON Data",
                    data=json_data,
                    file_name=f"incidents_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("📋 No incident history available. Report incidents to build history.")

def get_recommendations(severity):
    """Get recommendations based on severity level"""
    if severity == "SERIOUS-URGENT":
        return "Immediate dispatch of armed response unit. Establish perimeter. Evacuate civilians if necessary. Contact emergency services."
    elif severity == "SERIOUS":
        return "Dispatch patrol unit for investigation. Increase surveillance in the area. Consider community alert."
    else:
        return "Monitor situation. Routine patrol check recommended. Document for trend analysis."

if __name__ == "__main__":
    main()