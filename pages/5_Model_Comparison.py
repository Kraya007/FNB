import patch_sklearn
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from optbinning import BinningProcess
from sklearn.pipeline import Pipeline
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(page_title="Model Comparison", page_icon="fnb logo.jpg", layout="wide")
apply_fnb_theme()

render_fnb_header("Model Comparison", "DataQuest 2026 | Bonus - Performance Benchmarking")

@st.cache_data
def load_data():
    file_path = "loan_book.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        from model_pipeline import add_engineered_features
        df = add_engineered_features(df)
        return df
    return None

@st.cache_data
def train_all_models(train_data, test_data, features, num_cols, cat_cols, target):
    """Train all models and return predictions + metrics."""
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    results = {}
    
    # 1. Baseline — Raw numerics, no WoE
    X_train_base = X_train[num_cols].fillna(X_train[num_cols].median())
    X_test_base = X_test[num_cols].fillna(X_train[num_cols].median())
    baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr.fit(X_train_base, y_train)
    baseline_preds = baseline_lr.predict_proba(X_test_base)[:, 1]
    results['Baseline LR\n(Raw Numerics)'] = {
        'preds': baseline_preds,
        'auc': roc_auc_score(y_test, baseline_preds),
        'n_features': len(num_cols),
        'description': 'Logistic Regression on raw numeric features with median imputation. No feature engineering.'
    }
    
    # 2. Primary Model — WoE + all features
    bp1 = BinningProcess(variable_names=features, categorical_variables=cat_cols,
                         selection_criteria={"iv": {"min": 0.01}})
    primary_lr = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    pipe1 = Pipeline([("binning_process", bp1), ("logistic_regression", primary_lr)])
    pipe1.fit(X_train, y_train)
    primary_preds = pipe1.predict_proba(X_test)[:, 1]
    primary_feats = list(bp1.transform(X_train).columns)
    results['Primary WoE LR\n(Max AUC)'] = {
        'preds': primary_preds,
        'auc': roc_auc_score(y_test, primary_preds),
        'n_features': len(primary_feats),
        'description': 'WoE-transformed features with domain-engineered ratios. Maximises AUC; 3 features have positive coefficients due to multicollinearity.'
    }
    
    # 3. Sign-Corrected Model
    bp2 = BinningProcess(variable_names=features, categorical_variables=cat_cols,
                         selection_criteria={"iv": {"min": 0.02}})
    bp2.fit(X_train, y_train)
    X_tr_woe = bp2.transform(X_train)
    X_te_woe = bp2.transform(X_test)
    sel_cols = list(X_tr_woe.columns)
    for _ in range(10):
        sc_lr = LogisticRegression(max_iter=3000, random_state=42, C=0.1)
        sc_lr.fit(X_tr_woe[sel_cols], y_train)
        pos = [c for c, coef in zip(sel_cols, sc_lr.coef_[0]) if coef > 0]
        if not pos:
            break
        sel_cols = [c for c in sel_cols if c not in pos]
    sc_preds = sc_lr.predict_proba(X_te_woe[sel_cols])[:, 1]
    results['Sign-Corrected LR\n(Scorecard)'] = {
        'preds': sc_preds,
        'auc': roc_auc_score(y_test, sc_preds),
        'n_features': len(sel_cols),
        'description': 'Iteratively removes features with positive coefficients. All signs negative = production-ready scorecard.'
    }
    
    # 4. LightGBM Benchmark (non-linear ceiling)
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        num_leaves=31, random_state=42, verbosity=-1)
        # Use numeric cols only for LightGBM (it handles them natively)
        X_tr_lgb = X_train[num_cols].fillna(-999)
        X_te_lgb = X_test[num_cols].fillna(-999)
        lgb_model.fit(X_tr_lgb, y_train)
        lgb_preds = lgb_model.predict_proba(X_te_lgb)[:, 1]
        results['LightGBM\n(Non-Linear Ceiling)'] = {
            'preds': lgb_preds,
            'auc': roc_auc_score(y_test, lgb_preds),
            'n_features': len(num_cols),
            'description': 'Gradient Boosted Trees — included ONLY as a performance ceiling benchmark. NOT the submitted model.'
        }
    except ImportError:
        pass
    
    return results, y_test.values

df = load_data()

if df is not None:
    target = 'default_flag'
    exclude_cols = [target, 'applicant_id_hash', 'application_date', 'set']
    features = [col for col in df.columns if col not in exclude_cols]
    
    train_df = df[df['set'] == 'train'].copy()
    test_df = df[df['set'] == 'test'].copy()
    
    num_cols = train_df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df[features].select_dtypes(exclude=[np.number]).columns.tolist()
    
    with st.spinner("Training all models for comparison..."):
        results, y_test = train_all_models(train_df, test_df, features, num_cols, cat_cols, target)
    
    template = get_fnb_plotly_template()
    
    # --- AUC BAR CHART ---
    st.subheader("AUC Comparison")
    
    model_names = list(results.keys())
    aucs = [results[m]['auc'] for m in model_names]
    n_feats = [results[m]['n_features'] for m in model_names]
    colors = ['#7B919A', '#006B6B', '#48BFB5', '#F5A623'][:len(model_names)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=model_names, y=aucs, text=[f"{a:.4f}" for a in aucs],
        textposition='outside', marker_color=colors,
        marker_line_color='rgba(255,255,255,0.2)', marker_line_width=1
    ))
    fig.update_layout(**template, height=400, yaxis_range=[0.65, max(aucs)+0.03],
                      title="AUC by Model", yaxis_title="AUC (Test Set)")
    st.plotly_chart(fig, width='stretch')
    
    # --- METRICS TABLE ---
    st.subheader("Detailed Metrics")
    metrics_data = []
    for m in model_names:
        r = results[m]
        gini = 2 * r['auc'] - 1
        fpr, tpr, _ = roc_curve(y_test, r['preds'])
        ks = np.max(tpr - fpr)
        metrics_data.append({
            'Model': m.replace('\n', ' '),
            'AUC': f"{r['auc']:.4f}",
            'Gini': f"{gini:.4f}",
            'KS Statistic': f"{ks:.4f}",
            'Features': r['n_features'],
            'Description': r['description']
        })
    st.dataframe(pd.DataFrame(metrics_data), width='stretch', hide_index=True)
    
    # --- ROC CURVES ---
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    for i, m in enumerate(model_names):
        fpr, tpr, _ = roc_curve(y_test, results[m]['preds'])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{m.replace(chr(10),' ')} (AUC={results[m]['auc']:.4f})",
            line=dict(color=colors[i], width=2.5)
        ))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random", 
                                  line=dict(color='rgba(255,255,255,0.2)', dash='dash')))
    fig_roc.update_layout(**template, height=500, title="ROC Curves",
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, width='stretch')
    
    # --- PRECISION-RECALL CURVES ---
    st.subheader("Precision-Recall Curves")
    fig_pr = go.Figure()
    for i, m in enumerate(model_names):
        prec, rec, _ = precision_recall_curve(y_test, results[m]['preds'])
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, name=m.replace(chr(10), ' '),
            line=dict(color=colors[i], width=2.5)
        ))
    base_rate = y_test.mean()
    fig_pr.add_trace(go.Scatter(x=[0,1], y=[base_rate, base_rate], name="Baseline (random)",
                                line=dict(color='rgba(255,255,255,0.2)', dash='dash')))
    fig_pr.update_layout(**template, height=500, title="Precision-Recall Curves",
                          xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, width='stretch')
    
    # --- KEY TAKEAWAYS ---
    st.markdown("---")
    st.subheader("Key Takeaways")
    
    primary_auc = results[list(results.keys())[1]]['auc']
    baseline_auc = results[list(results.keys())[0]]['auc']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
        <h4 style="color:#48BFB5 !important;">WoE Feature Engineering Impact</h4>
        <p>WoE transformation improved AUC from <strong>{baseline_auc:.4f}</strong> to <strong>{primary_auc:.4f}</strong>
        — a <strong>+{(primary_auc - baseline_auc):.4f}</strong> improvement.</p>
        <p>This demonstrates that optimal binning + domain feature engineering captures nearly all the signal available in the data.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if len(results) >= 4:
            lgb_auc = results[list(results.keys())[3]]['auc']
            pct_ceiling = primary_auc / lgb_auc * 100
            st.markdown(f"""
            <div class="glass-card-gold">
            <h4 style="color:#F5A623 !important;">Linear vs Non-Linear Gap</h4>
            <p>Our logistic regression captures <strong>{pct_ceiling:.1f}%</strong> of the LightGBM ceiling ({lgb_auc:.4f}).</p>
            <p>The remaining gap is due to higher-order interactions that cannot be captured by a linear model — but the trade-off is <strong>full interpretability</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.error("Please ensure the data is loaded correctly.")

