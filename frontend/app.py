import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import io
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Any

# Add parent dir so we can import src.*
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.inference import load_artifacts, clinical_decision_support, validate_patient_input, preprocess_like_training
from src import config

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

st.set_page_config(
    page_title="Microbiome Autoimmune Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    defaults = {
        'artifacts_loaded': False,
        'artifacts': None,
        'patient_id': '',
        'sample_date': date.today(),
        'clinical_notes': '',
        'uploaded_csv': None,
        'processed_data': None,
        'prediction_result': None,
        'selected_model': 'ensemble',
        'csv_validated': False,
        'prediction_history': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

def init_database():
    db_path = config.RESULTS_DIR / "clinical_predictions.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinical_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            sample_date TEXT NOT NULL,
            analysis_date TEXT NOT NULL,
            model_used TEXT NOT NULL,
            predicted_condition TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            clinical_notes TEXT,
            csv_filename TEXT,
            features_data TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

init_database()

@st.cache_resource
def load_model_artifacts():
    try:
        return load_artifacts(prefer="ensemble")
    except Exception as e:
        st.error(f"Failed to load model artifacts: {str(e)}")
        return None

   

def auto_format_csv_data(df, selected_features):
    """Match uploaded CSV columns to expected features and validate."""
    st.subheader("Formatting & Validation")
    
    st.write(f"**Uploaded CSV:** {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.write("**Column matching:**")
    
    exact_matches = []
    partial_matches = []
    missing_features = []
    
    for feature in selected_features:
        if feature in df.columns:
            exact_matches.append(feature)
        else:
            found_partial = False
            for col in df.columns:
                if (feature.lower() in col.lower() or 
                    col.lower() in feature.lower() or
                    feature.replace('_', ' ').lower() == col.replace('_', ' ').lower()):
                    partial_matches.append((feature, col))
                    found_partial = True
                    break
            
            if not found_partial:
                missing_features.append(feature)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"✅ **Matched:** {len(exact_matches)}")
        if exact_matches:
            for feature in exact_matches[:5]:  # Show first 5
                st.text(f"• {feature}")
            if len(exact_matches) > 5:
                st.text(f"... and {len(exact_matches) - 5} more")
    
    with col2:
        if partial_matches:
            st.warning(f"⚠️ **Partial matches:** {len(partial_matches)}")
            for req_feat, csv_col in partial_matches[:5]:
                st.text(f"• {req_feat} → {csv_col}")
            if len(partial_matches) > 5:
                st.text(f"... and {len(partial_matches) - 5} more")
    
    with col3:
        if missing_features:
            st.error(f"❌ **Missing:** {len(missing_features)}")
            for feature in missing_features[:5]:
                st.text(f"• {feature}")
            if len(missing_features) > 5:
                st.text(f"... and {len(missing_features) - 5} more")
    
    formatted_data = {}
    warnings = []
    
    for feature in exact_matches:
        formatted_data[feature] = df[feature].iloc[0] if len(df) > 0 else 0.0
    
    if partial_matches:
        st.write("**Confirm these column mappings:**")
        for req_feat, csv_col in partial_matches:
            if st.checkbox(f"Use '{csv_col}' for '{req_feat}'", value=True, key=f"match_{req_feat}"):
                formatted_data[req_feat] = df[csv_col].iloc[0] if len(df) > 0 else 0.0
            else:
                missing_features.append(req_feat)
    
    if missing_features:
        st.write("**Missing features (filled with 0):**")
        for feature in missing_features:
            formatted_data[feature] = 0.0
            warnings.append(f"Missing feature '{feature}' set to 0.0")
    
    formatted_df = pd.DataFrame([formatted_data])
    
    if warnings:
        st.warning("**Formatting notes:**")
        for warning in warnings:
            st.write(f"• {warning}")
    
    try:
        validated_df, validation_warnings = validate_patient_input(formatted_df, selected_features)
        
        if validation_warnings:
            st.info("**Validation notes:**")
            for warning in validation_warnings:
                st.write(f"• {warning}")
        
        st.success("✅ Data formatted and validated")
        
        st.write("**Formatted data preview:**")
        st.dataframe(validated_df.T, use_container_width=True)
        
        return validated_df, True
        
    except Exception as e:
        st.error(f"Validation failed: {str(e)}")
        return None, False

def main():
    st.title("🧬 Microbiome Autoimmune Classifier")
    st.markdown("### Predict autoimmune conditions from stool microbiome data")
    
    
    if not st.session_state.artifacts_loaded:
        with st.spinner("Loading models..."):
            artifacts = load_model_artifacts()
            if artifacts:
                st.session_state.artifacts = artifacts
                st.session_state.artifacts_loaded = True
                st.success(f"✅ Models loaded — {artifacts['model_name']} ({len(artifacts['selected_features'])} features)")
            else:
                st.error("Failed to load model files. Make sure you've run the training pipeline first.")
                st.stop()
    
    st.sidebar.header("⚙️ Model Settings")
    
    available_models = ["Ensemble"]
    try:
        xgb_path = config.MODELS_DIR / "model_XGBoost.joblib"
        if xgb_path.exists():
            available_models.append("XGBoost")
    except:
        pass
    
    model_choice = st.sidebar.selectbox(
        "Select Model:", 
        options=available_models,
        help="Ensemble uses stacking to combine multiple classifiers"
    )
    st.session_state.selected_model = model_choice.lower()
    
    st.sidebar.write("**Test Set Performance:**")
    if model_choice == "Ensemble":
        st.sidebar.metric("Accuracy", "92.4%")
        st.sidebar.metric("F1-Score", "91.5%")
    else:
        st.sidebar.metric("Accuracy", "92.3%")
        st.sidebar.metric("F1-Score", "91.3%")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Patient Info", 
        "📄 Upload CSV", 
        "🔬 Run Analysis", 
        "📊 Results",
        "📋 History"
    ])
    
    with tab1:
        patient_info_tab()
    
    with tab2:
        csv_upload_tab()
    
    with tab3:
        analysis_tab()
    
    with tab4:
        results_explanation_tab()
    
    with tab5:
        history_tab()

def patient_info_tab():
    st.header("👤 Patient Info")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.session_state.patient_id = st.text_input(
            "Patient ID *", 
            value=st.session_state.patient_id,
            placeholder="e.g. PT-001",
            help="Used to track this sample"
        )
        
        st.session_state.sample_date = st.date_input(
            "Sample Collection Date *",
            value=st.session_state.sample_date,
            help="When the stool sample was collected"
        )
    
    with col2:
        st.session_state.clinical_notes = st.text_area(
            "Clinical Notes",
            value=st.session_state.clinical_notes,
            height=120,
            placeholder="Any symptoms, medications, or notes about the patient...",
            help="Optional — gets included in the report"
        )
    
    if st.session_state.patient_id:
        st.success(f"Patient ID: {st.session_state.patient_id}")
    else:
        st.warning("Enter a Patient ID to continue.")

def csv_upload_tab():
    st.header("📄 Upload Sample Data")
    
    if not st.session_state.artifacts_loaded:
        st.error("Models haven't loaded yet — wait a moment.")
        return
    
    st.info("""
    **How to upload:**
    
    1. Upload a CSV with your microbiome feature data
    2. The system will try to match columns to the expected features
    3. Missing features get filled with zeros
    4. Data is validated before analysis
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV with microbiome abundance data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Uploaded data preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
            
            selected_features = st.session_state.artifacts['selected_features']
            
            formatted_data, validation_success = auto_format_csv_data(df, selected_features)
            
            if validation_success and formatted_data is not None:
                st.session_state.processed_data = formatted_data
                st.session_state.csv_validated = True
                st.session_state.uploaded_csv = uploaded_file.name
                
                st.success("CSV processed — ready for analysis!")
            else:
                st.session_state.csv_validated = False
                st.error("CSV processing failed. Check the data format.")
                
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            st.write("**Tips:**")
            st.write("- Make sure it's a valid CSV")
            st.write("- Check for non-numeric values in data columns")
            st.write("- Try re-exporting from your analysis tool")

def analysis_tab():
    st.header("🔬 Run Analysis")
    
    if not st.session_state.patient_id:
        st.warning("Enter a Patient ID first (Patient Info tab).")
        return
    
    if not st.session_state.csv_validated or st.session_state.processed_data is None:
        st.warning("Upload and validate a CSV first.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Patient ID", st.session_state.patient_id)
    
    with col2:
        st.metric("Sample Date", str(st.session_state.sample_date))
    
    with col3:
        st.metric("Model", st.session_state.selected_model.title())
    
    if st.button("🔬 **Run Analysis**", type="primary", use_container_width=True):
        with st.spinner("Analyzing microbiome data..."):
            try:
                patient_data = st.session_state.processed_data.iloc[0].to_dict()
                
                result = clinical_decision_support(
                    patient_data=patient_data,
                    prefer_model=st.session_state.selected_model
                )
                
                st.session_state.prediction_result = result
                
                save_analysis_to_db(result, patient_data)
                
                st.success("Analysis complete!")
                
                st.write("### Quick Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Condition", result['predicted_label'])
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    certainty_color = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}
                    certainty_icon = certainty_color.get(result['clinical_certainty'], "⚪")
                    st.metric("Certainty", f"{certainty_icon} {result['clinical_certainty']}")
                
                st.info("Check the **Results** tab for the full breakdown.")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.write("**Check that:**")
                st.write("- The CSV data is in the right format")
                st.write("- Models are loaded")
                st.write("- Try refreshing the page")

def results_explanation_tab():
    st.header("📊 Results")
    
    if st.session_state.prediction_result is None:
        st.warning("Run the analysis first (Analysis tab).")
        return
    
    result = st.session_state.prediction_result
    
    st.subheader("Prediction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Predicted Condition", 
            value=result['predicted_label'],
            help="Most likely autoimmune condition"
        )
    
    with col2:
        confidence_pct = f"{result['confidence']:.1%}"
        st.metric(
            label="Confidence", 
            value=confidence_pct,
            help="How sure the model is"
        )
    
    with col3:
        certainty = result['clinical_certainty']
        certainty_colors = {"High": "🟢", "Moderate": "🟡", "Low": "🔴"}
        icon = certainty_colors.get(certainty, "⚪")
        st.metric(
            label="Certainty Level", 
            value=f"{icon} {certainty}",
            help="High/Moderate/Low based on confidence threshold"
        )
    
    with col4:
        st.metric(
            label="Model", 
            value=result['model_used'],
            help="Which model made this prediction"
        )
    
    st.subheader("Interpretation")
    
    if result['confidence'] >= 0.8:
        st.success(f"""
        **High confidence:** The model strongly predicts **{result['predicted_label']}** 
        ({result['confidence']:.1%}). The microbiome profile closely matches this condition's signature.
        """)
    elif result['confidence'] >= 0.6:
        st.warning(f"""
        **Moderate confidence:** The model leans toward **{result['predicted_label']}** 
        ({result['confidence']:.1%}). Additional tests would help confirm.
        """)
    else:
        st.error(f"""
        **Low confidence:** The model suggests **{result['predicted_label']}** 
        ({result['confidence']:.1%}), but isn't very sure. More testing recommended.
        """)
    
    st.subheader("Probability Breakdown")
    
    prob_data = []
    for condition, probability in result['probabilities'].items():
        prob_data.append({
            "Condition": condition,
            "Probability": probability,
            "Percentage": f"{probability:.2%}",
            "Confidence Level": "High" if probability >= 0.8 else "Moderate" if probability >= 0.6 else "Low"
        })
    
    prob_df = pd.DataFrame(prob_data).sort_values('Probability', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        conditions = prob_df['Condition'].tolist()
        probabilities = prob_df['Probability'].tolist()
        
        bars = ax.barh(range(len(conditions)), probabilities)
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            if prob >= 0.8:
                bar.set_color('green')
            elif prob >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.set_xlabel('Probability')
        ax.set_title('Predicted Probabilities by Condition')
        ax.set_xlim(0, 1)
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**Probabilities:**")
        st.dataframe(prob_df[['Condition', 'Percentage', 'Confidence Level']], 
                    use_container_width=True, hide_index=True)
    
    st.subheader("Feature Analysis")
    
    if st.session_state.processed_data is not None:
        input_values = st.session_state.processed_data.iloc[0]
        
        feature_importance = pd.DataFrame({
            'Feature': input_values.index,
            'Value': input_values.values,
            'Absolute_Value': np.abs(input_values.values)
        }).sort_values('Absolute_Value', ascending=False)
        
        st.write("**Top 10 features by value:**")
        top_features = feature_importance.head(10)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_pos = range(len(top_features))
            ax.barh(y_pos, top_features['Absolute_Value'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Absolute Value')
            ax.set_title('Top Microbiome Features')
            ax.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.dataframe(top_features[['Feature', 'Value']].round(6), 
                        use_container_width=True, hide_index=True)
    
   
    st.subheader("Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download PDF Report", use_container_width=True):
            if REPORTLAB_AVAILABLE:
                pdf_buffer = generate_clinical_report()
                if pdf_buffer:
                    st.download_button(
                        label="Save PDF",
                        data=pdf_buffer.getvalue(),
                        file_name=f"stool_analysis_report_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("PDF generation requires the reportlab package.")
    
    with col2:
        export_data = {
            'patient_id': [st.session_state.patient_id],
            'sample_date': [str(st.session_state.sample_date)],
            'analysis_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'predicted_condition': [result['predicted_label']],
            'confidence': [result['confidence']],
            'clinical_certainty': [result['clinical_certainty']]
        }
        
        for condition, prob in result['probabilities'].items():
            export_data[f'prob_{condition}'] = [prob]
        
        export_df = pd.DataFrame(export_data)
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"analysis_results_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def generate_clinical_report():
    """Generate a PDF report of the current analysis."""
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20,
        textColor=colors.darkblue,
        alignment=1  # Center
    )
    
    story.append(Paragraph("MICROBIOME AUTOIMMUNE ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.red,
        borderColor=colors.red,
        borderWidth=1,
        borderPadding=8,
        spaceAfter=20
    )
    
    story.append(Paragraph(
        "<b>⚠️ DISCLAIMER:</b> This report was generated by a student research project. ",
        disclaimer_style
    ))
    
    story.append(Paragraph("PATIENT INFORMATION", styles['Heading2']))
    
    patient_data = [
        ["Patient ID:", st.session_state.patient_id],
        ["Sample Collection Date:", str(st.session_state.sample_date)],
        ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["CSV File:", st.session_state.uploaded_csv or "Manual Entry"],
        ["Model Used:", st.session_state.prediction_result['model_used']]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("ANALYSIS RESULTS", styles['Heading2']))
    
    result = st.session_state.prediction_result
    results_data = [
        ["Predicted Condition:", result['predicted_label']],
        ["Model Confidence:", f"{result['confidence']:.2%}"],
        ["Clinical Certainty:", result['clinical_certainty']],
        ["Clinical Recommendation:", result['recommendation']]
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 4*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("CONDITION PROBABILITIES", styles['Heading2']))
    
    prob_data = [["Condition", "Probability", "Confidence Level"]]
    for condition, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        confidence_level = "High" if prob >= 0.8 else "Moderate" if prob >= 0.6 else "Low"
        prob_data.append([condition, f"{prob:.4f}", confidence_level])
    
    prob_table = Table(prob_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    if st.session_state.clinical_notes:
        story.append(Paragraph("CLINICAL NOTES", styles['Heading2']))
        story.append(Paragraph(st.session_state.clinical_notes, styles['Normal']))
        story.append(Spacer(1, 20))
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    
    story.append(Paragraph(
        f"Microbiome Autoimmune Classifier | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Research Use Only",
        footer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def save_analysis_to_db(result, patient_data):
    """Save prediction results to the SQLite history database."""
    try:
        db_path = config.RESULTS_DIR / "clinical_predictions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clinical_predictions 
            (patient_id, sample_date, analysis_date, model_used, predicted_condition, 
             confidence, probabilities, clinical_notes, csv_filename, features_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            st.session_state.patient_id,
            str(st.session_state.sample_date),
            datetime.now().isoformat(),
            result['model_used'],
            result['predicted_label'],
            result['confidence'],
            json.dumps(result['probabilities']),
            st.session_state.clinical_notes,
            st.session_state.uploaded_csv or "Manual Entry",
            json.dumps(patient_data)
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        st.error(f"Failed to save to database: {str(e)}")

def history_tab():
    st.header("📋 Analysis History")
    
    try:
        db_path = config.RESULTS_DIR / "clinical_predictions.db"
        
        if not db_path.exists():
            st.info("No history yet — run an analysis first.")
            return
            
        conn = sqlite3.connect(db_path)
        
        df = pd.read_sql_query('''
            SELECT patient_id, sample_date, analysis_date, model_used, 
                   predicted_condition, confidence, clinical_notes, csv_filename
            FROM clinical_predictions 
            ORDER BY analysis_date DESC 
            LIMIT 50
        ''', conn)
        
        conn.close()
        
        if len(df) > 0:
            st.write(f"Showing {len(df)} most recent analyses.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                patient_filter = st.text_input("Filter by Patient ID:")
            
            with col2:
                condition_filter = st.selectbox(
                    "Filter by Condition:", 
                    options=["All"] + list(df['predicted_condition'].unique())
                )
            
            filtered_df = df.copy()
            
            if patient_filter:
                filtered_df = filtered_df[filtered_df['patient_id'].str.contains(patient_filter, case=False)]
            
            if condition_filter != "All":
                filtered_df = filtered_df[filtered_df['predicted_condition'] == condition_filter]
            
            filtered_df['confidence_pct'] = (filtered_df['confidence'] * 100).round(1).astype(str) + '%'
            
            display_df = filtered_df[['patient_id', 'sample_date', 'analysis_date', 
                                    'predicted_condition', 'confidence_pct', 'model_used']].copy()
            display_df.columns = ['Patient ID', 'Sample Date', 'Analysis Date', 
                                'Predicted Condition', 'Confidence', 'Model Used']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            csv_buffer = io.StringIO()
            filtered_df.drop('confidence_pct', axis=1).to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download History CSV",
                data=csv_buffer.getvalue(),
                file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            st.subheader("Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", len(df))
            
            with col2:
                st.metric("Unique Patients", df['patient_id'].nunique())
            
            with col3:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col4:
                most_common = df['predicted_condition'].mode()[0] if len(df) > 0 else "N/A"
                st.metric("Most Common", most_common)
            
        else:
            st.info("No analyses recorded yet.")
            
    except Exception as e:
        st.error(f"Couldn't load history: {str(e)}")

if __name__ == "__main__":
    main()
