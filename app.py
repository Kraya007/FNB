import streamlit as st
import pandas as pd
import plotly.express as px
import os
from fnb_theme import apply_fnb_theme, render_fnb_header, get_fnb_plotly_template

st.set_page_config(
    page_title="DataQuest 2026 - Credit Risk",
    page_icon="fnb logo.jpg",
    layout="wide"
)

apply_fnb_theme()

@st.cache_data
def load_data():
    file_path = "loan_book.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        from model_pipeline import add_engineered_features
        df = add_engineered_features(df)
        return df
    else:
        st.error(f"Data file not found at {file_path}. Please ensure it's in the root directory.")
        return None

render_fnb_header("Interpretable Credit Models", "DataQuest 2026 | Home & Data Quality")

df = load_data()

if df is not None:
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Features", f"{df.shape[1]}")
    col3.metric("Default Rate", f"{df['default_flag'].mean():.1%}")
    col4.metric("Train / Test", f"{(df['set']=='train').sum():,} / {(df['set']=='test').sum():,}")
    
    st.markdown("---")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width='stretch')
    
    st.markdown("---")
    st.subheader("Data Quality Overview")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Missing Values by Feature**")
        if not missing_data.empty:
            missing_df = pd.DataFrame({
                'Missing Count': missing_data, 
                'Percentage (%)': (missing_data / len(df) * 100).round(2)
            })
            fig = px.bar(
                missing_df.reset_index(), 
                x='index', y='Percentage (%)',
                text='Missing Count',
                labels={'index': 'Feature'},
            )
            fig.update_layout(**get_fnb_plotly_template(), showlegend=False, height=350)
            fig.update_traces(marker_color='#006B6B', textposition='outside')
            st.plotly_chart(fig, width='stretch')
        else:
            st.success("No missing values found in the dataset!")
            
    with col2:
        st.markdown("**Target Variable Distribution**")
        if 'default_flag' in df.columns:
            target_counts = df['default_flag'].value_counts().reset_index()
            target_counts.columns = ['Default Flag', 'Count']
            target_counts['Default Flag'] = target_counts['Default Flag'].map({0: 'No Default', 1: 'Default'})
            
            total = target_counts['Count'].sum()
            target_counts['Percentage'] = (target_counts['Count'] / total * 100).round(1).astype(str) + '%'
            
            fig = px.bar(
                target_counts, 
                x='Default Flag', 
                y='Count', 
                text='Percentage',
                color='Default Flag',
                color_discrete_map={'No Default': '#006B6B', 'Default': '#F5A623'}
            )
            fig.update_layout(**get_fnb_plotly_template(), showlegend=False, height=350)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Column 'default_flag' not found.")
    
    st.markdown("---")
    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), width='stretch')

