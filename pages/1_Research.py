import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header

st.set_page_config(page_title="Research", page_icon="fnb logo.jpg", layout="wide")
apply_fnb_theme()

render_fnb_header("Research & Concepts", "DataQuest 2026 | Task 1 - Theory")

# =========================================================
# SECTION 1: GLM vs Non-Linear with PLOT
# =========================================================
st.markdown(r"""
### 1. Generalised Linear Models (GLMs) vs Non-Linear Models
In classification, a **Logistic Regression** (a type of GLM) models the log-odds of the probability of the positive class as a linear combination of the features:
$$ \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n $$
Because the relationship is linear in the transformed space, the impact of each feature $x_i$ on the log-odds is constant, measured exactly by $\beta_i$.

**Non-Linear Models** (like Random Forests or Gradient Boosting Machines such as LightGBM) create complex decision boundaries. They capture non-linear relationships and interactions automatically through tree-building or other mechanisms. While they often achieve higher accuracy, they operate as "black boxes" where individual feature contributions are difficult to isolate purely through an equation.
""")

# Visual comparison plot
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fnb_theme import get_fnb_plotly_template

template = get_fnb_plotly_template()

np.random.seed(42)
n = 200
X1 = np.concatenate([np.random.randn(n//2) - 1, np.random.randn(n//2) + 1])
X2 = np.concatenate([np.random.randn(n//2) - 0.5, np.random.randn(n//2) + 0.5])
y = np.array([0]*(n//2) + [1]*(n//2))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Logistic Regression (Linear Boundary)", "Non-Linear Model (Complex Boundary)"))

# Plot 1: Linear boundary
fig.add_trace(go.Scatter(x=X1[y==0], y=X2[y==0], mode='markers', name='Good (No Default)',
    marker=dict(color='#006B6B', size=6, opacity=0.6)), row=1, col=1)
fig.add_trace(go.Scatter(x=X1[y==1], y=X2[y==1], mode='markers', name='Bad (Default)',
    marker=dict(color='#F5A623', size=6, opacity=0.6)), row=1, col=1)
x_line = np.linspace(-4, 4, 100)
fig.add_trace(go.Scatter(x=x_line, y=-0.8*x_line, mode='lines', name='Decision Boundary',
    line=dict(color='white', width=2, dash='dash'), showlegend=False), row=1, col=1)

# Plot 2: Non-linear boundary
fig.add_trace(go.Scatter(x=X1[y==0], y=X2[y==0], mode='markers', showlegend=False,
    marker=dict(color='#006B6B', size=6, opacity=0.6)), row=1, col=2)
fig.add_trace(go.Scatter(x=X1[y==1], y=X2[y==1], mode='markers', showlegend=False,
    marker=dict(color='#F5A623', size=6, opacity=0.6)), row=1, col=2)
theta = np.linspace(0, 2*np.pi, 100)
fig.add_trace(go.Scatter(x=1.8*np.cos(theta), y=1.2*np.sin(theta), mode='lines',
    line=dict(color='white', width=2, dash='dash'), showlegend=False), row=1, col=2)

fig.update_layout(**template, height=380, title_text="Decision Boundary Comparison")
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.3))
fig.update_xaxes(title_text="Feature 1", row=1, col=1)
fig.update_xaxes(title_text="Feature 1", row=1, col=2)
fig.update_yaxes(title_text="Feature 2", row=1, col=1)
st.plotly_chart(fig, width='stretch')

st.info("**Key takeaway:** Logistic Regression can only draw a straight line (hyperplane) to separate classes. Non-linear models can draw curved or irregular boundaries. In credit modelling, we sacrifice the curved boundary for **full transparency** — every coefficient has a direct, auditable interpretation.")

# =========================================================
# SECTION 2-4: Interpretability, WoE/IV, Metrics (unchanged)
# =========================================================
st.markdown(r"""
### 2. The Balance Between Interpretability and Complexity
In the credit space, interpretability is a non-negotiable requirement due to regulatory oversight. A model must be explainable to:
- **Risk Managers**: Need to understand the exact drivers of a score to make informed policy rules.
- **Customers**: Under regulations like GDPR or the Equal Credit Opportunity Act, customers have a right to adverse action notices explaining exactly why they were rejected.
- **Regulators**: Must ensure the model doesn't exhibit bias or use prohibited factors.

While deep learning or ensembles (high complexity) might yield a higher Area Under the Curve (AUC), they sacrifice this transparency. Logistic Regression with proper feature engineering strikes the required balance.

### 3. Weight of Evidence (WoE) and Information Value (IV)
**Weight of Evidence (WoE)** is a transformation technique used primarily in credit scoring. It measures the predictive power of an independent variable in relation to the dependent variable.
$$ WoE_i = \\ln\\left( \\frac{\\% \\text{ of Good Customers in Bin } i}{\\% \\text{ of Bad Customers in Bin } i} \\right) $$
Replacing categorical or continuous values with their WoE establishes a strict linear relationship with the log-odds, satisfying the core assumption of Logistic Regression.

**Why WoE is especially useful in credit:**
- Handles missing values naturally (as a separate bin)
- Neutralises outliers through binning
- Forces monotonic relationships interpretable by risk managers
- Converts all features to the same scale, making coefficients directly comparable

**Information Value (IV)** measures the total predictive power of the feature:
$$ IV = \\sum (\\% \\text{ of Good} - \\% \\text{ of Bad}) \\times WoE $$
- IV < 0.02: Useless for prediction
- 0.02 to 0.1: Weak predictor
- 0.1 to 0.3: Medium predictor
- 0.3 to 0.5: Strong predictor
- \> 0.5: Suspicious or too good to be true

### 4. Key Metrics in Credit Modelling
- **Accuracy**: The overall percentage of correct predictions. Rarely useful in credit risk due to imbalanced datasets (defaults are usually a small percentage). A model predicting "no default" for everyone would achieve ~85% accuracy but be useless.
- **AUC (Area Under the ROC Curve)**: Measures the model's ability to rank-order risk. An AUC of 0.5 is random, 1.0 is perfect. The primary benchmark metric in credit scoring because it is threshold-independent.
- **Gini Coefficient**: Mathematically $Gini = 2 \\times AUC - 1$. Used extensively in credit scoring to summarize predictive power. A Gini > 0.40 is considered strong in retail lending.
- **Precision**: Of all loans we predicted would default, how many actually defaulted? High precision means fewer good customers are wrongly rejected (lower opportunity cost).
- **Recall**: Of all actual defaults, how many did we predict? High recall minimizes missed defaults (lower financial loss from write-offs).
- **$F_1$ Score**: The harmonic mean of precision and recall. Useful when you need a single number, but in credit modelling, the asymmetric cost of errors means precision and recall should be evaluated separately.
""")

# =========================================================
# SECTION 5: EXPANDED Regulatory Constraints
# =========================================================
st.markdown(r"""
### 5. Regulatory Constraints & Prohibited Features
Even though our data is simulated, a responsible model developer must consider which features a regulator may object to.
""")

reg_data = {
    'Feature': ['age', 'region', 'home_ownership', 'email_domain_type', 'phone_verified'],
    'Regulatory Risk': ['HIGH', 'MEDIUM', 'LOW', 'MEDIUM', 'LOW'],
    'Concern': [
        'Direct proxy for a protected class. Under ECOA (US) and the SA National Credit Act (NCA), age-based discrimination is prohibited unless it explicitly favours elderly applicants.',
        'Can serve as a proxy for race or ethnicity due to geographic segregation patterns. Regulators may require proof that regional effects are not driven by demographic composition.',
        'Generally permissible as it reflects financial stability, but could correlate with socioeconomic status in ways that disadvantage certain groups.',
        'Could act as a proxy for income level or digital literacy, which may correlate with protected characteristics. Requires careful justification.',
        'Low risk as it reflects identity verification, not a demographic trait.'
    ],
    'Included in Model?': ['Yes (with caution)', 'No (excluded)', 'Yes', 'No (excluded)', 'No (excluded)']
}

import pandas as pd
st.dataframe(pd.DataFrame(reg_data), width='stretch', hide_index=True)

st.markdown("""
<div class="glass-card">
<h4 style="color:#48BFB5 !important;">South African Context</h4>
<p>Under the <strong>National Credit Act (NCA)</strong> of South Africa, lenders must conduct affordability assessments 
and cannot discriminate based on race, gender, age, or disability. The <strong>Protection of Personal Information Act (POPIA)</strong> 
further restricts use of personal data for automated decision-making without consent.</p>
<p>While our simulated data does not contain explicit race or gender fields, features like <code>age</code> and <code>region</code> 
can act as <strong>proxies</strong> for protected characteristics. A production model would need to undergo <strong>fairness testing</strong> 
(e.g., disparate impact analysis) before deployment.</p>
</div>
""", unsafe_allow_html=True)
