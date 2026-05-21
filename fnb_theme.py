"""
FNB DataQuest 2026 Theme — Winner-Inspired Clean Professional
Inspired by the DataQuest 2025 winning presentation aesthetic.
Dark charcoal + dual-teal palette with bold, presentation-weight typography.
"""

# ── FNB Brand Colors (Winner-Inspired Palette) ──────────────────────
FNB_TEAL_PRIMARY = "#009999"      # Core teal (FNB brand)
FNB_TEAL_DARK = "#006B6B"         # Dark teal (bar charts primary)
FNB_TEAL_LIGHT = "#48BFB5"        # Light teal (bar charts secondary)
FNB_TEAL_ACCENT = "#00CED1"       # Bright teal (highlights, links)
FNB_TEAL_DEEPER = "#004D4D"       # Deepest teal (gradients)
FNB_GOLD = "#F5A623"              # Gold — warnings/danger only
FNB_GOLD_LIGHT = "#FFD580"        # Light gold — subtle accents
FNB_BG_DARK = "#1A2332"           # Clean dark charcoal background
FNB_BG_CARD = "#1F2B3D"           # Card background — slightly lighter
FNB_BG_SURFACE = "#243347"        # Surface elements
FNB_TEXT = "#FFFFFF"              # Primary text
FNB_TEXT_SECONDARY = "#C8D8DC"    # Secondary text
FNB_TEXT_MUTED = "#7B919A"        # Muted text


def get_fnb_plotly_template():
    """Return Plotly layout defaults matching the winner's clean chart style."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8F0F2", family="Inter, sans-serif", size=13),
        title_font=dict(color="#FFFFFF", size=18, family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.06)",
            title_font=dict(size=13),
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.06)",
            title_font=dict(size=13),
            tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        colorway=[
            "#006B6B", "#48BFB5", "#009999", "#00CED1", "#004D4D",
            "#F5A623", "#66CDAA", "#20B2AA", "#5FCFCF", "#2D9F9F",
        ],
        margin=dict(t=60, b=40, l=40, r=20),
    )


def apply_fnb_theme():
    """Inject the winner-inspired professional CSS into the Streamlit app."""
    import streamlit as st

    st.markdown("""
    <style>
    /* ========== GOOGLE FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ========== ANIMATED TOP ACCENT BAR ========== */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #004D4D, #009999, #48BFB5, #009999, #004D4D);
        background-size: 200% 100%;
        animation: shimmer 4s ease-in-out infinite;
        z-index: 9999;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ========== GLOBAL ========== */
    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: #1A2332 !important;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #15202E 0%, #1A2332 100%) !important;
        border-right: 1px solid rgba(0,153,153,0.15) !important;
    }
    /* Sidebar top accent strip */
    section[data-testid="stSidebar"]::before {
        content: '';
        display: block;
        height: 3px;
        background: linear-gradient(90deg, #009999, #48BFB5);
        margin: 0 0 12px 0;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #C8D8DC !important;
    }

    /* ========== HEADERS ========== */
    h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.5px !important;
    }
    h2 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    h3 {
        color: #48BFB5 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    h4, h5, h6 {
        color: #C8D8DC !important;
        font-weight: 600 !important;
    }

    /* ========== TEXT ========== */
    .stMarkdown, .stMarkdown p, .stText, label, span {
        color: #C8D8DC !important;
    }
    a { color: #48BFB5 !important; }
    a:hover { color: #00CED1 !important; }

    /* ========== CLEAN CARD CONTAINERS ========== */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #1F2B3D !important;
        border: 1px solid rgba(0,153,153,0.12) !important;
        border-left: 4px solid #009999 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }

    /* ========== METRICS ========== */
    div[data-testid="stMetric"] {
        background: #1F2B3D !important;
        border: 1px solid rgba(0,153,153,0.12) !important;
        border-left: 4px solid #009999 !important;
        border-radius: 8px !important;
        padding: 18px 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }
    div[data-testid="stMetric"] label {
        color: #7B919A !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }

    /* ========== DATAFRAMES ========== */
    .stDataFrame, div[data-testid="stDataFrame"] {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    .stDataFrame table {
        background: #1F2B3D !important;
    }
    .stDataFrame th {
        background: #006B6B !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border-bottom: 2px solid #009999 !important;
    }
    .stDataFrame td {
        color: #C8D8DC !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* ========== SELECT BOX / INPUTS ========== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #1F2B3D !important;
        border: 1px solid rgba(0,153,153,0.2) !important;
        border-radius: 6px !important;
        color: #E8F0F2 !important;
    }
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover {
        border-color: #009999 !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #009999 !important;
        box-shadow: 0 0 0 2px rgba(0,153,153,0.2) !important;
    }

    /* ========== SLIDER ========== */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #006B6B, #48BFB5) !important;
    }
    .stSlider > div > div > div > div > div {
        background: #FFFFFF !important;
        border: 2px solid #009999 !important;
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #006B6B, #009999) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0,153,153,0.2) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #009999, #48BFB5) !important;
        box-shadow: 0 4px 12px rgba(0,153,153,0.3) !important;
    }

    /* ========== ALERTS / INFO BOXES ========== */
    .stAlert, div[data-testid="stAlert"] {
        background: #1F2B3D !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="notification"][kind="info"],
    .element-container .stAlert[data-baseweb="notification"] {
        border-left: 4px solid #009999 !important;
    }
    .stSuccess {
        border-left: 4px solid #48BFB5 !important;
    }
    .stWarning {
        border-left: 4px solid #F5A623 !important;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent !important;
        border-bottom: 2px solid rgba(0,153,153,0.15) !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7B919A !important;
        font-weight: 600 !important;
        border-bottom: 3px solid transparent !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #C8D8DC !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 3px solid #009999 !important;
        background: transparent !important;
    }

    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: #1F2B3D !important;
        border-radius: 8px !important;
        color: #E8F0F2 !important;
        font-weight: 600 !important;
    }

    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-top-color: #009999 !important;
    }

    /* ========== SEPARATOR ========== */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,153,153,0.2), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #15202E;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(0,153,153,0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0,153,153,0.4);
    }

    /* ========== PLOTLY CHARTS ========== */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ========== WINNER-STYLE HEADER BAR ========== */
    .fnb-header {
        background: #1F2B3D;
        border-left: 5px solid transparent;
        border-image: linear-gradient(180deg, #009999, #48BFB5) 1;
        border-radius: 0 8px 8px 0;
        padding: 24px 32px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
        position: relative;
    }
    .fnb-header img {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .fnb-header-text h1 {
        margin: 0 !important;
        padding: 0 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        color: #FFFFFF !important;
    }
    .fnb-header-text p {
        color: #7B919A !important;
        margin: 6px 0 0 0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    .fnb-header .fnb-badge {
        display: inline-block;
        background: linear-gradient(135deg, #006B6B, #009999);
        color: #FFFFFF;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* ========== CLEAN CARD UTILITY ========== */
    .glass-card {
        background: #1F2B3D;
        border: 1px solid rgba(0,153,153,0.12);
        border-left: 4px solid #009999;
        border-radius: 0 8px 8px 0;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin-bottom: 16px;
    }
    .glass-card-gold {
        background: #1F2B3D;
        border: 1px solid rgba(245,166,35,0.15);
        border-left: 4px solid #F5A623;
        border-radius: 0 8px 8px 0;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin-bottom: 16px;
    }

    /* ========== HIDE DEFAULT STREAMLIT ELEMENTS ========== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: #1A2332 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_fnb_header(title, subtitle="DataQuest 2026"):
    """Render a branded FNB header with teal gradient accent strip (winner's style)."""
    import streamlit as st
    import base64
    import os

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fnb logo.jpg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" width="52">'
    else:
        logo_html = ""

    st.markdown(f"""
    <div class="fnb-header">
        {logo_html}
        <div class="fnb-header-text">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="fnb-badge">FNB DataQuest 2026</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
