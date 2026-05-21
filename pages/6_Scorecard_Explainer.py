import patch_sklearn
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(page_title="Scorecard & Explainer", page_icon="fnb logo.jpg", layout="wide")
apply_fnb_theme()

render_fnb_header("Scorecard & Explainability", "DataQuest 2026 | Bonus - Points System & Feature Contributions")

@st.cache_data
def load_data():
    file_path = "loan_book.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        from model_pipeline import add_engineered_features
        df = add_engineered_features(df)
        return df
    return None

@st.cache_resource
def load_model():
    pkl_path = "improved_model_pipeline.pkl"
    if os.path.exists(pkl_path):
        saved = joblib.load(pkl_path)
        if isinstance(saved, dict):
            # Use sign-corrected model for scorecard (all negative coefs)
            if 'sign_corrected_pipeline' in saved:
                return saved['sign_corrected_pipeline'], saved['sign_corrected_features'], 'sign_corrected'
            elif 'primary_pipeline' in saved:
                return saved['primary_pipeline'], saved['primary_features'], 'primary'
            elif 'pipeline' in saved:
                return saved['pipeline'], saved['features'], 'legacy'
    return None, None, None


def build_scorecard(pipeline, feature_names, base_score=600, pdo=20):
    """
    Convert logistic regression WoE coefficients into a scorecard points system.
    
    Standard scorecard formula:
      Score = Base Score + sum(Points_i)
      Points_i = -(beta_i * WoE_i + alpha/n) * factor
      
    Where:
      factor = pdo / ln(2)
      alpha = intercept
      n = number of features
    """
    lr = pipeline.named_steps['logistic_regression']
    bp = pipeline.named_steps['binning_process']
    
    intercept = lr.intercept_[0]
    coefficients = lr.coef_[0]
    n_features = len(coefficients)
    
    factor = pdo / np.log(2)
    offset = base_score - factor * intercept
    
    scorecard_data = []
    for i, (feat, coef) in enumerate(zip(feature_names, coefficients)):
        # Get the binning table for this feature
        try:
            optb = bp.get_binned_variable(feat)
            bin_table = optb.binning_table.build()
            # Remove totals row
            bin_table = bin_table[bin_table.index != 'Totals'].copy()
            
            for _, row in bin_table.iterrows():
                woe_val = row.get('WoE', 0)
                try:
                    woe_val = float(woe_val)
                except (ValueError, TypeError):
                    woe_val = 0
                    
                # Points for this bin
                points = -(coef * woe_val + intercept / n_features) * factor
                
                bin_label = row.get('Bin', 'Unknown')
                if hasattr(bin_label, 'tolist'):
                    bin_label = ', '.join(str(x) for x in bin_label.tolist())
                
                scorecard_data.append({
                    'Feature': feat,
                    'Bin': str(bin_label),
                    'WoE': round(woe_val, 4),
                    'Coefficient': round(coef, 4),
                    'Points': round(points, 1),
                    'Event Rate': round(float(row.get('Event rate', 0)) * 100, 2)
                })
        except Exception:
            # Feature not found in binning process, skip
            continue
    
    return pd.DataFrame(scorecard_data), offset, factor


df = load_data()
pipeline, feature_names, model_type = load_model()
template = get_fnb_plotly_template()

if df is not None and pipeline is not None:
    target = 'default_flag'
    test_df = df[df['set'] == 'test'].copy() if 'set' in df.columns else df.sample(frac=0.2, random_state=42)
    
    # =========================================================
    # TAB LAYOUT
    # =========================================================
    tab1, tab2 = st.tabs(["Scorecard Points System", "Applicant Explainability"])
    
    # =========================================================
    # TAB 1: SCORECARD POINTS SYSTEM
    # =========================================================
    with tab1:
        st.markdown("Convert WoE model coefficients into a traditional **scorecard points system** used in production lending systems.")
        
        col1, col2 = st.columns(2)
        with col1:
            base_score = st.number_input("Base Score", value=600, min_value=100, max_value=1000, step=50)
        with col2:
            pdo = st.number_input("Points to Double Odds (PDO)", value=20, min_value=5, max_value=100, step=5)
        
        scorecard_df, offset, factor = build_scorecard(pipeline, feature_names, base_score, pdo)
        
        if not scorecard_df.empty:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Base Score", f"{base_score}")
            col2.metric("PDO (Points to Double Odds)", f"{pdo}")
            col3.metric("Scaling Factor", f"{factor:.2f}")
            
            st.markdown("---")
            
            # Feature selector for scorecard
            st.subheader("Scorecard by Feature")
            selected_feat = st.selectbox("Select Feature", scorecard_df['Feature'].unique())
            
            feat_card = scorecard_df[scorecard_df['Feature'] == selected_feat].copy()
            
            # Display table
            st.dataframe(feat_card[['Bin', 'WoE', 'Points', 'Event Rate']].rename(
                columns={'Event Rate': 'Default Rate (%)'}
            ), width='stretch', hide_index=True)
            
            # Points bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=feat_card['Bin'], y=feat_card['Points'],
                text=[f"{p:+.0f}" for p in feat_card['Points']],
                textposition='outside',
                marker_color=['#006B6B' if p >= 0 else '#F5A623' for p in feat_card['Points']],
                marker_line_color='rgba(255,255,255,0.2)', marker_line_width=1
            ))
            fig.update_layout(**template, height=400,
                              title=f"Scorecard Points: {selected_feat}",
                              xaxis_title="Bin", yaxis_title="Points")
            st.plotly_chart(fig, width='stretch')
            
            st.info("""
            **How to read this:** Higher points = lower risk = better score. 
            In a production system, each applicant's total score is the sum of points across all features.
            A higher total score means the applicant is more creditworthy.
            """)
            
            # Full scorecard table
            with st.expander("View Full Scorecard Table"):
                st.dataframe(scorecard_df, width='stretch', hide_index=True)
    
    # =========================================================
    # TAB 2: APPLICANT EXPLAINABILITY
    # =========================================================
    with tab2:
        st.markdown("Select an applicant to see **exactly how each feature contributed** to their predicted probability of default.")
        
        # Sample applicants for demo
        n_sample = min(500, len(test_df))
        sample_df = test_df.head(n_sample).copy()
        
        # Get predictions
        try:
            sample_df['predicted_pd'] = pipeline.predict_proba(sample_df[feature_names])[:, 1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
        
        # Applicant selector
        if 'applicant_id_hash' in sample_df.columns:
            display_ids = sample_df['applicant_id_hash'].astype(str).tolist()
        else:
            display_ids = [f"Applicant {i}" for i in range(len(sample_df))]
        
        selected_idx = st.selectbox("Select Applicant", range(len(display_ids)), 
                                     format_func=lambda i: f"{display_ids[i]} (PD: {sample_df.iloc[i]['predicted_pd']:.1%})")
        
        applicant = sample_df.iloc[selected_idx]
        
        # Key info
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted PD", f"{applicant['predicted_pd']:.2%}")
        col2.metric("Actual Default", "Yes" if applicant.get(target, 0) == 1 else "No")
        if 'loan_amount' in applicant.index:
            col3.metric("Loan Amount", f"{applicant['loan_amount']:,.0f}")
        if 'annual_income' in applicant.index:
            col4.metric("Annual Income", f"{applicant['annual_income']:,.0f}")
        
        st.markdown("---")
        st.subheader("Feature Contribution Waterfall")
        
        # Calculate WoE contributions
        lr = pipeline.named_steps['logistic_regression']
        bp = pipeline.named_steps['binning_process']
        
        intercept = lr.intercept_[0]
        coefficients = lr.coef_[0]
        
        # Transform single applicant through binning process
        applicant_features = applicant[feature_names].to_frame().T
        applicant_woe = bp.transform(applicant_features)
        
        contributions = []
        for feat, coef, woe_val in zip(feature_names, coefficients, applicant_woe.values[0]):
            contribution = coef * woe_val
            raw_val = applicant.get(feat, 'N/A')
            try:
                raw_val = f"{float(raw_val):,.2f}" if isinstance(raw_val, (int, float, np.floating)) else str(raw_val)
            except (ValueError, TypeError):
                raw_val = str(raw_val)
            contributions.append({
                'Feature': feat,
                'Raw Value': raw_val,
                'WoE': round(float(woe_val), 4),
                'Coefficient': round(coef, 4),
                'Contribution': round(contribution, 4)
            })
        
        contrib_df = pd.DataFrame(contributions)
        contrib_df = contrib_df.sort_values('Contribution', key=abs, ascending=False)
        
        # Waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Contributions",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(contrib_df) + ["total"],
            x=["Intercept"] + contrib_df['Feature'].tolist() + ["Final Log-Odds"],
            y=[intercept] + contrib_df['Contribution'].tolist() + [0],
            connector=dict(line=dict(color="rgba(255,255,255,0.1)")),
            increasing=dict(marker_color="#F5A623"),  # Increases PD (bad)
            decreasing=dict(marker_color="#006B6B"),   # Decreases PD (good)
            totals=dict(marker_color="#48BFB5"),
            textposition="outside",
            text=[f"{intercept:.3f}"] + [f"{c:.3f}" for c in contrib_df['Contribution']] + [f"{intercept + contrib_df['Contribution'].sum():.3f}"]
        ))
        
        fig.update_layout(**template, height=550,
                          title="Feature Contribution Waterfall (Log-Odds Space)",
                          yaxis_title="Log-Odds Contribution",
                          showlegend=False)
        st.plotly_chart(fig, width='stretch')
        
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#48BFB5 !important;">How to Read This</h4>
        <p><span style="color:#006B6B;">Teal bars</span> = features that <strong>decrease</strong> default risk (protective factors)</p>
        <p><span style="color:#F5A623;">Gold bars</span> = features that <strong>increase</strong> default risk (risk factors)</p>
        <p>The waterfall shows how the model builds up from the intercept to the final prediction, feature by feature.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed contribution table
        st.subheader("Detailed Contributions")
        st.dataframe(contrib_df, width='stretch', hide_index=True)

else:
    if pipeline is None:
        st.error("Model not found. Please run model_pipeline.py first to generate 'improved_model_pipeline.pkl'.")
    else:
        st.error("Please ensure the data is loaded correctly.")

