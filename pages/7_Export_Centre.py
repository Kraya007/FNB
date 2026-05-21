import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(page_title="Export Centre", page_icon="fnb logo.jpg", layout="wide")
apply_fnb_theme()

render_fnb_header("Export Centre", "DataQuest 2026 | Download Reports & Data")
st.markdown("Generate and download professional reports, data exports, and model summaries with a single click.")

# ── Data Loading ────────────────────────────────────────────────────
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
    import joblib
    pkl_path = "improved_model_pipeline.pkl"
    if os.path.exists(pkl_path):
        saved = joblib.load(pkl_path)
        if isinstance(saved, dict):
            if 'sign_corrected_pipeline' in saved:
                return saved['sign_corrected_pipeline'], saved['sign_corrected_features'], 'sign_corrected'
            elif 'primary_pipeline' in saved:
                return saved['primary_pipeline'], saved['primary_features'], 'primary'
    return None, None, None

df = load_data()
pipeline, feature_names, model_type = load_model()

if df is not None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ════════════════════════════════════════════════════════════════
    # SECTION 1: PORTFOLIO REPORT
    # ════════════════════════════════════════════════════════════════
    st.subheader("Portfolio Reports")
    col1, col2, col3 = st.columns(3)

    # ── 1A: Data Summary CSV ──
    with col1:
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#48BFB5 !important;">📊 Data Summary</h4>
        <p>Descriptive statistics for all features including count, mean, std, min, max, and quartiles.</p>
        </div>
        """, unsafe_allow_html=True)
        summary = df.describe(include='all').T
        summary_csv = summary.to_csv()
        st.download_button(
            label="⬇️ Download Data Summary (CSV)",
            data=summary_csv,
            file_name=f"dataquest_data_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── 1B: Data Quality Report ──
    with col2:
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#48BFB5 !important;">🔍 Data Quality Report</h4>
        <p>Missing values, data types, unique counts, and completeness percentage for every feature.</p>
        </div>
        """, unsafe_allow_html=True)

        quality_data = pd.DataFrame({
            'Feature': df.columns,
            'Data Type': [str(df[col].dtype) for col in df.columns],
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Missing Count': [df[col].isna().sum() for col in df.columns],
            'Missing %': [round(df[col].isna().mean() * 100, 2) for col in df.columns],
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Completeness %': [round(df[col].notna().mean() * 100, 2) for col in df.columns],
        })
        quality_csv = quality_data.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Quality Report (CSV)",
            data=quality_csv,
            file_name=f"dataquest_data_quality_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ── 1C: Full Dataset ──
    with col3:
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#48BFB5 !important;">📁 Full Dataset</h4>
        <p>Complete dataset with all engineered features included. Ready for external analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        @st.cache_data
        def get_full_csv():
            return df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Full Dataset (CSV)",
            data=get_full_csv(),
            file_name=f"dataquest_full_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 2: MODEL EXPORTS
    # ════════════════════════════════════════════════════════════════
    st.subheader("Model Exports")

    if pipeline is not None:
        col1, col2 = st.columns(2)

        # ── 2A: Model Equation & Coefficients ──
        with col1:
            st.markdown("""
            <div class="glass-card">
            <h4 style="color:#48BFB5 !important;">🧮 Model Coefficients</h4>
            <p>Full logistic regression equation with intercept, coefficients, and sign indicators. This is the mathematical form required in the deliverables.</p>
            </div>
            """, unsafe_allow_html=True)

            lr = pipeline.named_steps['logistic_regression']
            intercept = lr.intercept_[0]
            coefficients = lr.coef_[0]

            # Build equation text
            equation_lines = []
            equation_lines.append("=" * 60)
            equation_lines.append(f"MODEL: {model_type.upper()}")
            equation_lines.append(f"Generated: {timestamp}")
            equation_lines.append("=" * 60)
            equation_lines.append("")
            equation_lines.append("LOGISTIC REGRESSION EQUATION")
            equation_lines.append("-" * 40)
            equation_lines.append(f"Log-odds (η) = {intercept:.6f}")

            coef_data = []
            for feat, coef in zip(feature_names, coefficients):
                if abs(coef) > 0.0001:
                    sign = "+" if coef > 0 else "-"
                    flag = "  [!POSITIVE — remove for scorecard]" if coef > 0 else ""
                    equation_lines.append(f"  {sign} {abs(coef):.6f} × WoE({feat}){flag}")
                    coef_data.append({
                        'Feature': feat,
                        'Coefficient': round(coef, 6),
                        'Sign': 'Positive ⚠️' if coef > 0 else 'Negative ✓',
                        'Abs_Coefficient': round(abs(coef), 6)
                    })

            n_pos = sum(1 for c in coefficients if c > 0)
            equation_lines.append("")
            if n_pos == 0:
                equation_lines.append("✓ All coefficients are NEGATIVE (scorecard-compliant)")
            else:
                equation_lines.append(f"⚠ {n_pos} coefficient(s) are POSITIVE (needs sign correction)")

            equation_lines.append("")
            equation_lines.append("COEFFICIENT TABLE")
            equation_lines.append("-" * 40)

            coef_df = pd.DataFrame(coef_data).sort_values('Abs_Coefficient', ascending=False)
            equation_lines.append(coef_df.to_string(index=False))

            equation_text = "\n".join(equation_lines)

            st.download_button(
                label="⬇️ Download Model Equation (TXT)",
                data=equation_text,
                file_name=f"dataquest_model_equation_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

            # Also offer coefficients as CSV
            coef_csv = coef_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Coefficients (CSV)",
                data=coef_csv,
                file_name=f"dataquest_coefficients_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # ── 2B: Scorecard Points Table ──
        with col2:
            st.markdown("""
            <div class="glass-card">
            <h4 style="color:#48BFB5 !important;">🎯 Scorecard Export</h4>
            <p>Complete points-based scorecard table with bins, WoE values, points, and default rates for every feature.</p>
            </div>
            """, unsafe_allow_html=True)

            # Build scorecard
            bp = pipeline.named_steps['binning_process']
            base_score = 600
            pdo = 20
            factor = pdo / np.log(2)

            scorecard_rows = []
            for i, (feat, coef) in enumerate(zip(feature_names, coefficients)):
                try:
                    optb = bp.get_binned_variable(feat)
                    bin_table = optb.binning_table.build()
                    bin_table = bin_table[bin_table.index != 'Totals'].copy()

                    for _, row in bin_table.iterrows():
                        woe_val = float(row.get('WoE', 0)) if pd.notna(row.get('WoE', None)) else 0
                        points = -(coef * woe_val + intercept / len(coefficients)) * factor

                        bin_label = row.get('Bin', 'Unknown')
                        if hasattr(bin_label, 'tolist'):
                            bin_label = ', '.join(str(x) for x in bin_label.tolist())

                        scorecard_rows.append({
                            'Feature': feat,
                            'Bin': str(bin_label),
                            'WoE': round(woe_val, 4),
                            'Coefficient': round(coef, 4),
                            'Points': round(points, 1),
                            'Event_Rate_%': round(float(row.get('Event rate', 0)) * 100, 2),
                            'Count': int(row.get('Count', 0)) if pd.notna(row.get('Count', None)) else 0,
                        })
                except Exception:
                    continue

            if scorecard_rows:
                scorecard_df = pd.DataFrame(scorecard_rows)

                # Excel export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    scorecard_df.to_excel(writer, sheet_name='Scorecard', index=False)
                    coef_df.to_excel(writer, sheet_name='Coefficients', index=False)

                    # Summary sheet
                    summary_info = pd.DataFrame({
                        'Parameter': ['Model Type', 'Base Score', 'PDO', 'Scaling Factor',
                                      'Intercept', 'Num Features', 'Generated'],
                        'Value': [model_type, base_score, pdo, round(factor, 2),
                                  round(intercept, 6), len(feature_names), timestamp]
                    })
                    summary_info.to_excel(writer, sheet_name='Model Info', index=False)

                st.download_button(
                    label="⬇️ Download Scorecard (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name=f"dataquest_scorecard_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

                # CSV fallback
                scorecard_csv = scorecard_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Scorecard (CSV)",
                    data=scorecard_csv,
                    file_name=f"dataquest_scorecard_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 3: SCORED PORTFOLIO EXPORT
    # ════════════════════════════════════════════════════════════════
    st.subheader("Scored Portfolio")

    st.markdown("""
    <div class="glass-card">
    <h4 style="color:#48BFB5 !important;">📈 Export Scored Test Set</h4>
    <p>The full test set with predicted Probability of Default (PD), credit scores, and approval decisions at your chosen threshold. This is the deliverable that proves your model works.</p>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is not None:
        threshold = st.slider("Set approval threshold for export", 0.01, 0.99, 0.15, 0.01)

        test_df = df[df['set'] == 'test'].copy() if 'set' in df.columns else df.sample(frac=0.2, random_state=42)

        try:
            test_df['predicted_pd'] = pipeline.predict_proba(test_df[feature_names])[:, 1]

            # Convert PD to credit score
            factor = pdo / np.log(2)
            test_df['credit_score'] = base_score - factor * np.log(
                test_df['predicted_pd'] / (1 - test_df['predicted_pd'])
            )
            test_df['credit_score'] = test_df['credit_score'].clip(300, 900).round(0).astype(int)
            test_df['decision'] = np.where(test_df['predicted_pd'] <= threshold, 'APPROVED', 'DECLINED')

            # Stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test Set Size", f"{len(test_df):,}")
            col2.metric("Approved", f"{(test_df['decision'] == 'APPROVED').sum():,}")
            col3.metric("Declined", f"{(test_df['decision'] == 'DECLINED').sum():,}")
            col4.metric("Avg Credit Score", f"{test_df['credit_score'].mean():.0f}")

            scored_csv = test_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Scored Portfolio (CSV)",
                data=scored_csv,
                file_name=f"dataquest_scored_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Scoring failed: {e}")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════
    # SECTION 4: EXECUTIVE REPORT (HTML)
    # ════════════════════════════════════════════════════════════════
    st.subheader("Executive Report")

    st.markdown("""
    <div class="glass-card">
    <h4 style="color:#48BFB5 !important;">📄 Professional HTML Report</h4>
    <p>A complete, presentation-ready report with model performance, key findings, scorecard summary, and business recommendations. Opens in any browser and can be printed to PDF.</p>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is not None and 'default_flag' in df.columns:
        from sklearn.metrics import roc_auc_score

        test_set = df[df['set'] == 'test'].copy() if 'set' in df.columns else df.sample(frac=0.2, random_state=42)
        try:
            preds = pipeline.predict_proba(test_set[feature_names])[:, 1]
            auc = roc_auc_score(test_set['default_flag'], preds)
            gini = 2 * auc - 1
        except Exception:
            auc = 0
            gini = 0

        total = len(df)
        default_rate = df['default_flag'].mean()
        n_features = len(feature_names)

        # Build HTML report
        html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DataQuest 2026 — Credit Risk Model Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: #1A2332; color: #C8D8DC; padding: 40px; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .header {{ border-left: 5px solid #009999; padding: 24px 32px; margin-bottom: 32px; background: #1F2B3D; border-radius: 0 8px 8px 0; }}
        .header h1 {{ color: #FFFFFF; font-size: 2rem; font-weight: 800; margin-bottom: 4px; }}
        .header p {{ color: #7B919A; text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.85rem; }}
        .badge {{ display: inline-block; background: linear-gradient(135deg, #006B6B, #009999); color: #FFF; font-size: 0.7rem; padding: 3px 10px; border-radius: 4px; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }}
        h2 {{ color: #FFFFFF; font-size: 1.4rem; font-weight: 700; margin: 28px 0 12px 0; }}
        h3 {{ color: #48BFB5; font-size: 1.1rem; font-weight: 600; margin: 20px 0 8px 0; }}
        .card {{ background: #1F2B3D; border-left: 4px solid #009999; border-radius: 0 8px 8px 0; padding: 20px; margin: 16px 0; }}
        .card-gold {{ background: #1F2B3D; border-left: 4px solid #F5A623; border-radius: 0 8px 8px 0; padding: 20px; margin: 16px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
        .metric {{ background: #1F2B3D; border-left: 4px solid #009999; border-radius: 0 8px 8px 0; padding: 16px; }}
        .metric .label {{ color: #7B919A; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }}
        .metric .value {{ color: #FFFFFF; font-size: 1.6rem; font-weight: 800; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th {{ background: #006B6B; color: #FFF; padding: 10px; text-align: left; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        td {{ padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9rem; }}
        .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid rgba(0,153,153,0.2); color: #7B919A; font-size: 0.8rem; text-align: center; }}
        @media print {{ body {{ background: #FFF; color: #333; }} .card, .card-gold, .metric {{ border-color: #009999; background: #f9f9f9; }} .metric .value, h1, h2 {{ color: #1A2332; }} th {{ background: #009999; }} }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Credit Risk Model Report</h1>
        <p>DataQuest 2026 — Interpretable Credit Models</p>
        <span class="badge">FNB DataQuest 2026</span>
    </div>

    <h2>1. Dataset Overview</h2>
    <div class="metrics">
        <div class="metric"><div class="label">Total Records</div><div class="value">{total:,}</div></div>
        <div class="metric"><div class="label">Features</div><div class="value">{df.shape[1]}</div></div>
        <div class="metric"><div class="label">Default Rate</div><div class="value">{default_rate:.1%}</div></div>
        <div class="metric"><div class="label">Train / Test</div><div class="value">{(df['set']=='train').sum():,} / {(df['set']=='test').sum():,}</div></div>
    </div>

    <h2>2. Model Performance</h2>
    <div class="metrics">
        <div class="metric"><div class="label">Model Type</div><div class="value">{model_type.replace('_', ' ').title()}</div></div>
        <div class="metric"><div class="label">AUC</div><div class="value">{auc:.4f}</div></div>
        <div class="metric"><div class="label">Gini</div><div class="value">{gini:.4f}</div></div>
        <div class="metric"><div class="label">Features Used</div><div class="value">{n_features}</div></div>
    </div>

    <h2>3. Logistic Regression Equation</h2>
    <div class="card">
        <h3>Linear Predictor (η)</h3>
        <p style="font-family: monospace; margin-top: 8px;">η = {intercept:.4f}</p>
        {''.join(f'<p style="font-family: monospace;">&nbsp;&nbsp;{"+" if c > 0 else "-"} {abs(c):.4f} × WoE({f})</p>' for f, c in zip(feature_names, coefficients) if abs(c) > 0.0001)}
    </div>

    <h2>4. Feature Coefficients</h2>
    <table>
        <tr><th>Feature</th><th>Coefficient</th><th>Status</th></tr>
        {''.join(f'<tr><td>{f}</td><td>{c:.6f}</td><td>{"✓ Negative" if c <= 0 else "⚠️ Positive"}</td></tr>' for f, c in sorted(zip(feature_names, coefficients), key=lambda x: x[1]))}
    </table>

    <h2>5. Business Recommendations</h2>
    <div class="card">
        <p>Based on the model analysis:</p>
        <ul style="margin: 8px 0 0 20px; line-height: 1.8;">
            <li>Recommended approval threshold: <strong>0.15 – 0.20 PD</strong></li>
            <li>Expected approval rate at 0.15 threshold: ~70-75%</li>
            <li>All coefficients are {'negative ✓ (scorecard-compliant)' if all(c <= 0 for c in coefficients) else 'NOT all negative ⚠️ — use sign-corrected model for production'}</li>
            <li>The model captures approximately <strong>{auc/0.82*100:.0f}%</strong> of the non-linear (LightGBM) ceiling</li>
        </ul>
    </div>

    <div class="footer">
        Generated on {timestamp} | DataQuest 2026 — Building Interpretable Credit Models | FNB
    </div>
</div>
</body>
</html>"""

        st.download_button(
            label="⬇️ Download Executive Report (HTML)",
            data=html_report,
            file_name=f"dataquest_executive_report_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True
        )

        st.info("💡 **Tip:** Open the HTML file in your browser, then press **Ctrl+P** to print it as a PDF with full styling.")

else:
    st.error("Please ensure the data and model are loaded correctly.")
