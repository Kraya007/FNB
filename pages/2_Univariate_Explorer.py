import patch_sklearn
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from optbinning import OptimalBinning
from plotly.subplots import make_subplots
import os, sys
from pandas.api.types import is_numeric_dtype

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(page_title="Univariate Explorer", page_icon="fnb logo.jpg", layout="wide")
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

render_fnb_header("Univariate Explorer", "DataQuest 2026 | Feature Analysis & OptBinning")

df = load_data()

if df is not None:
    st.markdown("Explore individual features, their optimal bins, and how they relate to `default_flag`.")
    
    target = 'default_flag'
    
    if target in df.columns:
        features = [col for col in df.columns if col not in [target, 'applicant_id_hash', 'application_date', 'set']]
        
        selected_feature = st.selectbox("Select a Feature to Explore", features)
        
        # Using training set only for binning to avoid data leakage
        if 'set' in df.columns:
            train_df = df[df['set'] == 'train'].copy()
        else:
            train_df = df.copy()
            
        x = train_df[selected_feature].values
        y = pd.to_numeric(train_df[target], errors='coerce').fillna(0).astype(int).values
        
        # Dtype detection
        if not is_numeric_dtype(train_df[selected_feature]) or train_df[selected_feature].nunique() < 10:
            dtype = 'categorical'
        else:
            dtype = 'numerical'
            
        st.subheader(f"Optimal Binning Analysis: {selected_feature}")
        
        with st.spinner("Calculating Optimal Bins..."):
            optb = OptimalBinning(name=selected_feature, dtype=dtype, solver="cp")
            optb.fit(x, y)
            
            binning_table = optb.binning_table.build()
            iv_value = binning_table.loc['Totals', 'IV']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Information Value (IV)", value=f"{iv_value:.4f}")
            with col2:
                if iv_value < 0.02:
                    st.warning("Useless predictor")
                elif iv_value < 0.1:
                    st.info("Weak predictor")
                elif iv_value < 0.3:
                    st.success("Medium predictor")
                else:
                    st.success("Strong predictor")
            
            st.markdown("### Binning Table")
            st.markdown("This table demonstrates exactly how the feature has been discretized. Check the **Event rate** and **WoE** columns to see if the monotonic relationship makes business sense.")
            
            # Clean up PyArrow array representations
            if 'Bin' in binning_table.columns:
                def format_bin_label(b):
                    if isinstance(b, str):
                        return b
                    try:
                        if hasattr(b, "tolist"):
                            b = b.tolist()
                        return ", ".join([str(item) for item in b])
                    except TypeError:
                        return str(b)
                binning_table['Bin'] = binning_table['Bin'].apply(format_bin_label)

            display_table = binning_table.copy()
            # Convert numeric columns to proper types (fixes PyArrow serialization)
            for col in ['Count', 'Count (%)', 'Non-event', 'Event', 'Event rate', 'WoE', 'IV', 'JS']:
                if col in display_table.columns:
                    display_table[col] = pd.to_numeric(display_table[col], errors='coerce')
            
            def safe_format(x):
                try:
                    return '{:.4f}'.format(float(x))
                except (ValueError, TypeError):
                    return str(x)
                    
            st.dataframe(display_table.style.format({
                'WoE': safe_format,
                'Event rate': safe_format,
                'IV': safe_format
            }), width='stretch')
            
            st.markdown("### Visualization")
            
            plot_data = binning_table.drop('Totals')
            
            template = get_fnb_plotly_template()
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Bar(x=plot_data['Bin'], y=plot_data['Count'], name="Count", 
                       marker_color='#006B6B',
                       marker_line_color='#48BFB5', marker_line_width=1),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=plot_data['Bin'], y=plot_data['WoE'], name="WoE", 
                          mode='lines+markers', 
                          line=dict(color='#48BFB5', width=3),
                          marker=dict(size=8, color='#48BFB5', line=dict(color='#FFFFFF', width=2))),
                secondary_y=True,
            )

            fig.update_layout(
                **template,
                title_text=f"Bin Counts and WoE for {selected_feature}",
                xaxis_title="Bins",
                height=450,
            )
            fig.update_yaxes(title_text="Count", secondary_y=False)
            fig.update_yaxes(title_text="Weight of Evidence (WoE)", secondary_y=True)

            st.plotly_chart(fig, width='stretch')
            
            st.markdown("### Business Interpretation")
            st.info("**Judges Note:** Do the calculated bins align with business intuition? For instance, does higher income show monotonically lower risk? Optbinning guarantees optimal statistical splits, but a risk analyst must verify the logic.")
            
    else:
        st.error("Target variable 'default_flag' not found.")
else:
    st.error("Please ensure the data is loaded correctly.")


