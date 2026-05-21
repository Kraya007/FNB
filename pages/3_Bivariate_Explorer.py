import streamlit as st
import pandas as pd
import plotly.express as px
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(page_title="Bivariate Explorer", page_icon="fnb logo.jpg", layout="wide")
apply_fnb_theme()

@st.cache_data
def load_data():
    file_path = "loan_book.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        from model_pipeline import add_engineered_features
        df = add_engineered_features(df)
        return df
    return None

render_fnb_header("Bivariate Explorer", "DataQuest 2026 | Feature Interactions")
st.markdown("Inspect relationships between two variables and how they interact with the target.")

df = load_data()

if df is not None:
    features = [col for col in df.columns if col not in ['applicant_id_hash', 'application_date', 'set']]
    template = get_fnb_plotly_template()
    
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Select Feature X", features, index=0)
    with col2:
        feature_y = st.selectbox("Select Feature Y", features, index=1)
        
    if feature_x == feature_y:
        st.warning("Please select two different features.")
    else:
        st.subheader(f"Interaction: {feature_x} vs {feature_y}")
        
        is_x_num = pd.api.types.is_numeric_dtype(df[feature_x]) and df[feature_x].nunique() > 10
        is_y_num = pd.api.types.is_numeric_dtype(df[feature_y]) and df[feature_y].nunique() > 10
        
        sample_size = min(10000, len(df))
        plot_df = df.sample(sample_size, random_state=42)
        
        if 'default_flag' in plot_df.columns:
            plot_df['default_flag_str'] = plot_df['default_flag'].map({0: 'No Default', 1: 'Default'})
            color_col = 'default_flag_str'
        else:
            color_col = None
        
        if is_x_num and is_y_num:
            fig = px.scatter(
                plot_df, x=feature_x, y=feature_y, color=color_col,
                opacity=0.5,
                color_discrete_map={'No Default': '#006B6B', 'Default': '#F5A623'},
                title=f"Scatter Plot (Sampled to {sample_size:,})"
            )
            fig.update_layout(**template, height=500)
            st.plotly_chart(fig, width='stretch')
            
            corr = df[[feature_x, feature_y]].corr().iloc[0, 1]
            st.info(f"Pearson Correlation Coefficient: **{corr:.4f}**")
            
        elif (not is_x_num) and (not is_y_num):
            st.markdown("### Default Rate Heatmap")
            if 'default_flag' in df.columns:
                heatmap_data = df.groupby([feature_x, feature_y])['default_flag'].mean().reset_index()
                heatmap_pivot = heatmap_data.pivot(index=feature_y, columns=feature_x, values='default_flag')
                
                fig = px.imshow(
                    heatmap_pivot,
                    labels=dict(x=feature_x, y=feature_y, color="Default Rate"),
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    color_continuous_scale=[[0, '#004D4D'], [0.5, '#006B6B'], [1, '#48BFB5']],
                    text_auto=".2%"
                )
                fig.update_layout(**template, height=500)
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Needs 'default_flag' to show default rate heatmap.")
                
        else:
            num_feature = feature_x if is_x_num else feature_y
            cat_feature = feature_y if is_x_num else feature_x
            
            fig = px.box(
                plot_df, x=cat_feature, y=num_feature, color=color_col,
                title=f"Box Plot: {num_feature} by {cat_feature}",
                color_discrete_map={'No Default': '#006B6B', 'Default': '#F5A623'}
            )
            fig.update_layout(**template, height=500)
            st.plotly_chart(fig, width='stretch')
        
        # Insight callout — how bivariate exploration informs feature engineering
        st.markdown("---")
        st.subheader("How This Informs Feature Engineering")
        st.markdown("""
        <div class="glass-card">
        <h4 style="color:#48BFB5 !important;">Key Interactions Discovered</h4>
        <p>Through bivariate exploration, we identified several feature <strong>interactions</strong> that informed our engineering in Task 2:</p>
        <ul>
        <li><strong>loan_amount x annual_income</strong> — Large loans are only risky when income is low. This led to the <code>loan_to_income</code> ratio feature.</li>
        <li><strong>credit_utilisation_pct x dti_ratio</strong> — Applicants with <em>both</em> high utilisation and high DTI default at dramatically higher rates than either alone. This led to the <code>utilization_dti_ratio</code> interaction feature.</li>
        <li><strong>num_delinquencies_2yr x age</strong> — Younger borrowers with delinquencies are far riskier than older borrowers with the same count. This led to the <code>delinq_to_age</code> normalised feature.</li>
        <li><strong>total_revolving_balance x annual_income</strong> — High revolving balances are only concerning relative to income. This led to the <code>revolving_to_income</code> feature.</li>
        </ul>
        <p style="color:#8DA3A8;"><strong>Try it:</strong> Select <code>loan_amount</code> (X) and <code>annual_income</code> (Y) above, then compare the teal (No Default) vs gold (Default) clusters to see the affordability separation in action.</p>
        </div>
        """, unsafe_allow_html=True)
            
else:
    st.error("Please ensure the data is loaded correctly.")


