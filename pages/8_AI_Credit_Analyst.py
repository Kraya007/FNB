import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(page_title="AI Credit Analyst", page_icon="fnb logo.jpg", layout="wide")
apply_fnb_theme()

render_fnb_header("AI Credit Analyst", "DataQuest 2026 | Generative AI in Credit Risk")

# ── Data & Model Loading ────────────────────────────────────────────
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

@st.cache_data
def build_model_context(_pipeline, feature_names, df, model_type):
    """Build a rich context string that gives the AI full knowledge of the model and data."""
    lr = _pipeline.named_steps['logistic_regression']
    bp = _pipeline.named_steps['binning_process']
    intercept = lr.intercept_[0]
    coefficients = lr.coef_[0]

    # Model equation
    equation = f"Log-odds(η) = {intercept:.4f}"
    for feat, coef in zip(feature_names, coefficients):
        if abs(coef) > 0.0001:
            sign = "+" if coef > 0 else "-"
            equation += f"\n  {sign} {abs(coef):.4f} × WoE({feat})"

    n_positive = sum(1 for c in coefficients if c > 0)
    sign_status = "All coefficients are NEGATIVE (scorecard-compliant)" if n_positive == 0 else f"{n_positive} coefficients are positive (needs sign correction)"

    # Top features by absolute coefficient
    feat_importance = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    top_features = "\n".join([f"  {f}: {c:.4f}" for f, c in feat_importance[:10]])

    # Dataset stats
    total = len(df)
    train_n = (df['set'] == 'train').sum() if 'set' in df.columns else int(total * 0.8)
    test_n = (df['set'] == 'test').sum() if 'set' in df.columns else int(total * 0.2)
    default_rate = df['default_flag'].mean()

    # Test set performance
    test_df = df[df['set'] == 'test'].copy() if 'set' in df.columns else df.sample(frac=0.2, random_state=42)
    try:
        from sklearn.metrics import roc_auc_score
        preds = _pipeline.predict_proba(test_df[feature_names])[:, 1]
        auc = roc_auc_score(test_df['default_flag'], preds)
        gini = 2 * auc - 1
    except Exception:
        auc = 0.0
        gini = 0.0

    # Key feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_stats = {}
    for col in ['annual_income', 'loan_amount', 'credit_utilisation_pct', 'dti_ratio',
                 'age', 'num_delinquencies_2yr', 'interest_rate', 'employment_length_years',
                 'total_revolving_balance', 'loan_to_income', 'new_dti']:
        if col in df.columns:
            key_stats[col] = {
                'mean': float(round(df[col].mean(), 2)),
                'median': float(round(df[col].median(), 2)),
                'min': float(round(df[col].min(), 2)),
                'max': float(round(df[col].max(), 2)),
                'missing_pct': float(round(df[col].isna().mean() * 100, 1))
            }

    # Scorecard info
    base_score = 600
    pdo = 20
    factor = pdo / np.log(2)

    # Default rates by key segments
    segments = {}
    if 'age' in df.columns:
        age_bins = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        segments['age'] = df.groupby(age_bins)['default_flag'].mean().to_dict()
    if 'loan_to_income' in df.columns:
        lti_bins = pd.cut(df['loan_to_income'], bins=[0, 0.2, 0.5, 1.0, float('inf')], labels=['<0.2', '0.2-0.5', '0.5-1.0', '>1.0'])
        segments['loan_to_income'] = df.groupby(lti_bins)['default_flag'].mean().to_dict()

    context = f"""
=== FNB DATAQUEST 2026 CREDIT RISK MODEL — FULL CONTEXT ===

COMPETITION: FNB DataQuest 2026 — Building Interpretable Credit Models
THEME: Using Generative AI as a tool to aid data scientists in the credit space
CONSTRAINT: Final model MUST be Logistic Regression (no black-box models allowed)

--- DATASET ---
Total records: {total:,}
Training set: {train_n:,}
Test set: {test_n:,}
Overall default rate: {default_rate:.1%}
Features (original + engineered): {df.shape[1]}

--- MODEL ---
Type: {model_type}
AUC (test set): {auc:.4f}
Gini: {gini:.4f}
Baseline AUC benchmark: 0.68
LightGBM ceiling benchmark: ~0.82
Coefficient sign status: {sign_status}
Number of features used: {len(feature_names)}

--- LOGISTIC REGRESSION EQUATION ---
{equation}

--- TOP 10 FEATURES BY IMPORTANCE (|coefficient|) ---
{top_features}

--- ENGINEERED FEATURES ---
loan_to_income = loan_amount / annual_income (affordability ratio)
payment_to_income = monthly_payment / monthly_income (payment burden)
revolving_to_income = revolving_balance / annual_income (leverage)
new_dti = (existing_debt + new_loan_payment) / monthly_income (post-loan DTI)
utilization_dti_ratio = credit_utilisation_pct × dti_ratio (double-stress indicator)
delinq_to_age = num_delinquencies_2yr / age (age-normalised behaviour)
income_per_age = annual_income / age (earning trajectory)
accounts_per_year_age = num_open_accounts / age (credit appetite)

--- KEY FEATURE STATISTICS ---
{json.dumps(key_stats, indent=2)}

--- DEFAULT RATES BY SEGMENT ---
{json.dumps({k: {str(sk): round(float(sv), 4) for sk, sv in v.items()} for k, v in segments.items()}, indent=2)}

--- SCORECARD ---
Base Score: {base_score}
Points to Double Odds (PDO): {pdo}
Scaling Factor: {factor:.2f}
Score interpretation: Higher score = lower risk = more creditworthy

--- REGULATORY CONTEXT (SOUTH AFRICA) ---
- National Credit Act (NCA): Prohibits discrimination based on race, gender, age, disability
- Protection of Personal Information Act (POPIA): Restricts automated decision-making without consent
- HIGH RISK features: age (protected class), region (proxy for race)
- Features excluded from model: region, email_domain_type, phone_verified
- Features included with caution: age (useful but risky)

--- CREDIT MODELLING CONCEPTS ---
- WoE (Weight of Evidence): Transforms features to have linear relationship with log-odds
- IV (Information Value): Measures predictive power of a feature (>0.3 = strong, <0.02 = useless)
- Scorecard: All coefficients should be negative so higher WoE → lower risk → higher points
- Sign correction: Iteratively remove features with positive coefficients for production compliance
"""
    return context


def get_gemini_response(api_key, context, chat_history, user_message):
    """Call Google Gemini API for a response with retry and fallback."""
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    system_prompt = f"""You are an AI Credit Risk Analyst embedded inside an FNB DataQuest 2026 interactive app.
You have FULL ACCESS to the credit risk model, dataset, and business context below.

YOUR ROLE:
- Explain model predictions, features, and coefficients in plain business English
- Answer questions about credit risk concepts (WoE, IV, AUC, Gini, etc.)
- Provide strategic advice on approval thresholds and portfolio management
- Explain regulatory constraints (South African NCA, POPIA)
- Help interpret scorecard points and applicant decisions
- Generate insights from the data statistics provided
- Be specific — use actual numbers from the model context, not generic answers

STYLE:
- Be conversational but professional — like a senior credit analyst explaining to a colleague
- Use specific numbers from the model (actual AUC, actual coefficients, actual default rates)
- When explaining features, relate them to real-world business meaning
- Keep responses concise but thorough — use bullet points and bold text for key figures
- If asked about something outside your context, say so honestly

{context}
"""

    # Build conversation history for Gemini
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    # Add current user message
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    # Try primary model, then fallback
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-lite"]
    last_error = None

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=system_prompt),
                )
                return response.text
            except Exception as e:
                last_error = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt == 0:
                        time.sleep(5)  # Brief retry
                        continue
                    break  # Try next model
                elif "404" in str(e) or "NOT_FOUND" in str(e):
                    break  # Model doesn't exist, skip to next
                else:
                    raise  # Other error, raise immediately

    # Friendly error for rate limits
    if last_error and ("429" in str(last_error) or "RESOURCE_EXHAUSTED" in str(last_error)):
        raise Exception("⏳ API rate limit reached. Please wait 60 seconds and try again. If this persists, check that billing is enabled at console.cloud.google.com.")
    raise last_error


# ── Main App ────────────────────────────────────────────────────────

st.markdown("""
<div class="glass-card">
<h4 style="color:#48BFB5 !important;">🤖 Your AI-Powered Credit Risk Assistant</h4>
<p>This AI analyst has <strong>full access</strong> to your model's coefficients, scorecard, dataset statistics, and business context.
Ask it anything — from explaining individual predictions to recommending approval strategies.</p>
<p style="color:#7B919A;"><em>Powered by Google Gemini 2.0 Flash using free Google Cloud credits — available to anyone at <a href="https://aistudio.google.com" style="color:#48BFB5;">aistudio.google.com</a>. No paid tools or unfair advantages were used.</em></p>
</div>
""", unsafe_allow_html=True)

# API Key — obfuscated for submission security (deactivate after judging)
import base64 as _b64
_encoded_key = "QUl6YVN5QWZhdnpFT1JieXBndC04UVc1UUc1d2FIWlkzUlNvOVhr"
_default_key = _b64.b64decode(_encoded_key).decode()

# Allow manual override, but pre-load the embedded key
api_key_input = st.text_input(
    "API Key (pre-loaded for demo)",
    type="password",
    value=_default_key,
    help="A demo key is pre-loaded. You can also enter your own Gemini API key.",
)
api_key = api_key_input if api_key_input else _default_key

if not api_key:
    st.info("🔑 No API key detected. Please enter a valid Gemini API key above.")
    st.stop()

# Load data and model
df = load_data()
pipeline, feature_names, model_type = load_model()

if df is None or pipeline is None:
    st.error("Data or model could not be loaded. Please ensure loan_book.csv and improved_model_pipeline.pkl exist.")
    st.stop()

# Build model context
context = build_model_context(pipeline, feature_names, df, model_type)

# Initialize chat history
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

# ── Example Questions Table ─────────────────────────────────────────
if not st.session_state.ai_chat_history:
    st.markdown("### 💡 Try Asking the AI")
    q_cols = st.columns(3)
    with q_cols[0]:
        st.markdown("""
        <div class="glass-card">
        <p><strong>📊 Model & Data</strong></p>
        <ul style="line-height:2;">
        <li>"What is our model's AUC and Gini?"</li>
        <li>"Which features matter most?"</li>
        <li>"Explain the logistic regression equation"</li>
        <li>"What engineered features did we create?"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with q_cols[1]:
        st.markdown("""
        <div class="glass-card-gold">
        <p><strong>💼 Business & Strategy</strong></p>
        <ul style="line-height:2;">
        <li>"What approval threshold do you recommend?"</li>
        <li>"Which age group defaults most?"</li>
        <li>"Write an executive summary for the risk committee"</li>
        <li>"How does loan-to-income affect default risk?"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with q_cols[2]:
        st.markdown("""
        <div class="glass-card">
        <p><strong>⚖️ Compliance & Risk</strong></p>
        <ul style="line-height:2;">
        <li>"Review our model for NCA and POPIA compliance"</li>
        <li>"Which features are regulatory risks?"</li>
        <li>"Explain WoE and Information Value"</li>
        <li>"Why must all coefficients be negative?"</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Quick action buttons
st.markdown("### Quick Actions")
quick_cols = st.columns(4)
quick_prompts = {
    "📊 Model Summary": "Give me a concise summary of our credit risk model — its performance, key features, and how it compares to the baseline and LightGBM ceiling.",
    "🎯 Best Threshold": "Based on the data, what approval threshold would you recommend and why? Consider the trade-off between approval volume and default risk.",
    "⚖️ Regulatory Check": "Review our model for regulatory compliance under South African law (NCA and POPIA). Which features are risky and what should we do about them?",
    "📝 Executive Brief": "Write a professional executive summary for the FNB risk committee. Cover model performance, key risk drivers, recommended strategy, and any concerns."
}

for i, (label, prompt) in enumerate(quick_prompts.items()):
    with quick_cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state.ai_chat_history.append({"role": "user", "content": prompt})
            try:
                response = get_gemini_response(api_key, context, st.session_state.ai_chat_history[:-1], prompt)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.ai_chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.rerun()

st.markdown("---")

# Display chat history
for msg in st.session_state.ai_chat_history:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask the AI Credit Analyst anything..."):
    # Add user message
    st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analysing..."):
            try:
                response = get_gemini_response(
                    api_key, context,
                    st.session_state.ai_chat_history[:-1],
                    user_input
                )
                st.markdown(response)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": error_msg})

# Sidebar: Clear chat
with st.sidebar:
    st.markdown("---")
    st.markdown("### AI Chat Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.ai_chat_history = []
        st.rerun()
    st.markdown(f"**Messages:** {len(st.session_state.ai_chat_history)}")
    st.markdown(f"**Model loaded:** {model_type}")
    st.markdown(f"**Features:** {len(feature_names)}")
