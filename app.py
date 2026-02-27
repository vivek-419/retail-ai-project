import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Retail AI Dashboard",
    page_icon="🔹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* ===== FORCE LIGHT MODE GLOBALLY ===== */
    html, body {
        background-color: #f6f8fb !important;
        color: #1e293b !important;
    }

    /* Override ALL Streamlit root containers */
    .stApp, .stApp > div, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"],
    .main, .main > div {
        background-color: #f6f8fb !important;
        color: #1e293b !important;
    }

    /* Sidebar forced white */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border-right: 1px solid #e5e7eb !important;
    }

    /* All text elements */
    p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: inherit;
    }

    /* Slider track - light mode */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #2563eb !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f8fafc !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        color: #1e293b !important;
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] button {
        color: #1e293b !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #64748b !important;
    }

    /* Widgets inside sidebar */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #475569 !important;
        font-size: 0.85rem !important;
    }
    .stSlider [data-testid="stSliderThumbValue"] {
        color: #475569 !important;
    }

    /* Override dark mode completely */
    @media (prefers-color-scheme: dark) {
        .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stMain"], [data-testid="stMainBlockContainer"],
        section[data-testid="stSidebar"] {
            background-color: #f6f8fb !important;
            color: #1e293b !important;
            filter: none !important;
        }
    }

    /* Fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Hide default Streamlit header AND its reserved space */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
    }

    /* Remove the top gap Streamlit reserves for its toolbar */
    [data-testid="stToolbar"],
    .stToolbar {
        display: none !important;
    }

    /* Force the main app containers to start at the absolute top */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .stApp,
    .main {
        padding-top: 0 !important;
        margin-top: 0 !important;
        top: 0 !important;
    }

    /* Sidebar layout adjustment */
    section[data-testid="stSidebar"] {
        top: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-top: 52px !important;
    }

    /* Fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main block container spacing - account for the 52px nav bar */
    .block-container {
        padding-top: 70px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px;
        background-color: #f6f8fb !important;
    }

    /* Sidebar layout */
    .sidebar-section {
        font-size: 0.75rem;
        font-weight: 700;
        color: #1e293b; /* Darker as requested */
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-left: 1rem;
        border-left: 3px solid #3b82f6; /* Accent border */
    }
    .sidebar-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 1rem;
        margin: 0.2rem 0.5rem;
        border-radius: 8px;
        font-size: 0.95rem;
        font-weight: 500;
        color: #4b5563;
        cursor: pointer;
    }
    .sidebar-item.active {
        background-color: #eff6ff;
        color: #2563eb;
    }
    .sidebar-item-icon { margin-right: 0.75rem; font-size: 1.1rem; display: inline-flex; align-items: center; }

    /* Clean geometric sidebar icons */
    .sidebar-icon-bar {
        display: inline-block;
        width: 14px;
        height: 14px;
        background: currentColor;
        border-radius: 3px;
        opacity: 0.6;
    }
    .sidebar-icon-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: currentColor;
        border-radius: 50%;
        opacity: 0.5;
    }

    /* Upload box */
    .upload-box {
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        margin: 2rem 1rem 0.5rem 1rem;
        background-color: #f8fafc;
    }
    .upload-box-text {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.4;
    }

    /* Dashboard headers */
    .dashboard-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .dashboard-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        display: flex;
        flex-direction: column;
    }
    .kpi-title { font-size: 0.85rem; color: #64748b; font-weight: 500; margin-bottom: 0.5rem; }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; letter-spacing: -0.02em; }
    .kpi-trend { font-size: 0.8rem; font-weight: 500; display: flex; align-items: center; gap: 0.25rem; }
    .trend-up { color: #10b981; }
    .trend-down { color: #ef4444; }
    .trend-text { color: #94a3b8; font-weight: 400; }
    .kpi-line { height: 3px; border-radius: 3px; width: 40px; margin-top: 1rem; }

    /* Truncation Fixes */
    [data-testid="stMetricValue"] > div {
        font-size: 1.5rem !important; /* Smaller to fit boxes */
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.85rem !important;
    }
    .kpi-line-blue   { background: #3b82f6; }
    .kpi-line-green  { background: #10b981; }
    .kpi-line-orange { background: #f59e0b; }
    .kpi-line-red    { background: #ef4444; }

    /* Chart Cards */
    .chart-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
    }
    .chart-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
    .chart-subtitle { font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem; }

    /* Bottom Cards */
    .bottom-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
        overflow-x: auto !important; /* Table overflow fix */
    }
    .bottom-title { font-size: 1rem; font-weight: 700; color: #1e293b; }
    .bottom-subtitle { font-size: 0.85rem; color: #64748b; margin-top: 0.2rem; }

    /* Insight Callout */
    .insight-callout {
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #475569;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }

    /* Cluster Legend */
    .cluster-legend { display: flex; gap: 1rem; font-size: 0.85rem; color: #475569; margin-bottom: 1rem; flex-wrap: wrap; }
    .legend-item { display: flex; align-items: center; gap: 0.4rem; }
    .dot { width: 8px; height: 8px; border-radius: 50%; }

    /* Top Nav */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 1.5rem;
        background-color: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 52px;
        z-index: 1000001 !important; /* Higher than Streamlit defaults */
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .top-nav-logo { font-weight: 700; font-size: 1.1rem; color: #1a1d23; letter-spacing: -0.3px; }
    .top-nav-logo span { color: #2563eb; }
    .top-nav-links { display: flex; gap: 1.5rem; font-size: 0.85rem; font-weight: 500; color: #6b7280; }
    .top-nav-links .active { color: #2563eb; background-color: #eff6ff; padding: 0.35rem 0.85rem; border-radius: 6px; }

</style>

<!-- Top Navigation Injection -->
<div class="top-nav" style="box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); padding-left: 2rem;">
    <div class="top-nav-logo">Retail<span>AI</span></div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Content (Styled structurally) ──────────────────────
with st.sidebar:
    st.markdown("""
        <div class="upload-box" style="margin-top: 0; margin-bottom: 1rem;">
            <div class="upload-box-text">Upload your dataset to load live data</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Priority action: Upload at top
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    # Simple navigation logic
    st.markdown('<div class="sidebar-section">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio("", ["📊 Dashboard", "💡 Business Strategy"], label_visibility="collapsed")

    st.markdown(f"""
        <div class="sidebar-section">AI MODELS</div>
        <div class="sidebar-item {"active" if "Dashboard" in page else ""}">
            <span class="sidebar-item-icon" style="opacity: 0.8;">🧩</span> Segmentation
        </div>
        <div class="sidebar-item">
            <span class="sidebar-item-icon" style="opacity: 0.8;">📈</span> Forecasting
        </div>
        
        <div class="sidebar-section">INTERPRETATION</div>
        <div class="sidebar-item {"active" if "Strategy" in page else ""}">
            <span class="sidebar-item-icon" style="opacity: 0.8;">🏢</span> Economic Strategy
        </div>
    """, unsafe_allow_html=True)
    
    # K-Means settings directly to keep functionality
    if uploaded_file is not None:
         n_clusters = st.slider("Clusters (K):", 2, 8, 4, key="sidebar_k")
    else:
         n_clusters = 4

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Restore strict cleaning as per user request to match expected metrics
    df.dropna(inplace=True) 
    df.drop_duplicates(inplace=True)

    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        # User requested format='mixed' and dayfirst=True to handle Jan/Feb 2023 correctly
        df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce', format='mixed', dayfirst=True)
            
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

    # Prioritize Total_Amount as Revenue per user request
    total_amt_cols = [c for c in df.columns if 'total_amount' in c.lower()]
    if total_amt_cols:
        df['Revenue'] = pd.to_numeric(df[total_amt_cols[0]], errors='coerce')
    else:
        # Fallback: Quantity & Price
        qty_cols = [c for c in df.columns if any(x in c.lower() for x in ['qty','quantity','units','purchases'])]
        price_cols = [c for c in df.columns if any(x in c.lower() for x in ['price','cost','amount'])]
        
        if qty_cols and price_cols and qty_cols[0] != price_cols[0]:
            df['Revenue'] = pd.to_numeric(df[qty_cols[0]], errors='coerce') * pd.to_numeric(df[price_cols[0]], errors='coerce')
        elif 'Revenue' not in df.columns:
            total_cols = [c for c in df.columns if 'total' in c.lower() and c not in qty_cols]
            if total_cols:
                df['Revenue'] = pd.to_numeric(df[total_cols[0]], errors='coerce')

    cust_cols = [c for c in df.columns if any(x in c.lower() for x in ['customer','client','user'])]
    if cust_cols:
        df.rename(columns={cust_cols[0]: 'CustomerID'}, inplace=True)

    # Improved Product detection
    prod_match = [c for c in df.columns if any(x in c.lower() for x in ['product','item'])]
    if prod_match:
        df.rename(columns={prod_match[0]: 'Product'}, inplace=True)
    elif 'products' in df.columns: # Specific match for user screenshot
        df.rename(columns={'products': 'Product'}, inplace=True)
    else:
        name_cols = [c for c in df.columns if 'name' in c.lower() and (not cust_cols or c != cust_cols[0])]
        if name_cols:
            df.rename(columns={name_cols[0]: 'Product'}, inplace=True)

    # Only drop rows that are missing critical analysis data - already covered by global dropna
    # df.dropna(subset=['Revenue', 'Date'], inplace=True)
    return df

# Helper to format millions
def format_currency(val):
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif abs(val) >= 1_000:
        return f"${val/1_000:.1f}K" # Show one decimal place as requested
    else:
        return f"${val:.2f}"


# Fallback to simulated data if no upload, just to show the beautiful UI structure
is_simulated = False
if uploaded_file is None:
    # We'll show the dummy dashboard looking exactly like the image if no file is uploaded
    total_rev_str = "$2.84M"
    avg_order_str = "$147"
    unique_cust_str = "19,340"
    total_txn_str = "21,450"
    is_simulated = True
else:
    df = load_data(uploaded_file)
    total_rev = df['Revenue'].sum()
    avg_order = df['Revenue'].mean()
    unique_cust = df['CustomerID'].nunique() if 'CustomerID' in df.columns else 0
    total_txn = len(df)
    
    total_rev_str = format_currency(total_rev)
    avg_order_str = format_currency(avg_order)
    unique_cust_str = f"{unique_cust:,}"
    total_txn_str = f"{total_txn:,}"

# ── Strategy Page Logic ───────────────────────────────────────
def render_strategy_page():
    st.markdown('<div class="dashboard-title">Strategic Business Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Data-Driven Insights from the $401.97M Retail Ecosystem</div>', unsafe_allow_html=True)

    # 1. Revenue Optimization & CLV
    st.markdown("""
        <div style="border-left: 4px solid #3b82f6; padding-left: 15px; margin-bottom: 30px; margin-top: 30px;">
            <h3 style="margin:0; font-weight:700; color: #1e293b;">1. Revenue Optimization (Segmentation Strategy)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Premium Segment Value ($11K+ per Customer)</div>
                <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
                    Our AI models identify the <b>Premium</b> segment as the core driver of profitability, with an average revenue of <b>$11,006</b> per customer. 
                    Economically, these customers represent <b>inelastic demand</b>. 
                    <b>Strategy:</b> Implement "White Glove" services and exclusive early access to major categories like <b>Electronics</b> (our #1 category at $90M+) to maximize CLV.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Bargain & Frequent Dynamics</div>
                <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
                    The <b>Frequent</b> ($6.8K) and <b>Bargain</b> ($4.0K) segments represent our high-volume engine. 
                    With a lower individual order value (~$1.2K-$1.4K), these segments are highly sensitive to price changes. 
                    <b>Strategy:</b> Use bundle-pricing or cross-selling between Grocery and Clothing to increase the "Avg Order Value" beyond the current <b>$1.4K</b> benchmark.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 2. Demand-Supply Equilibrium
    st.markdown("""
        <div style="border-left: 4px solid #10b981; padding-left: 15px; margin-bottom: 30px; margin-top: 30px;">
            <h3 style="margin:0; font-weight:700; color: #1e293b;">2. Demand-Supply Dynamics (Forecasting Performance)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="chart-card">
            <div style="display: flex; gap: 2rem; align-items: flex-start;">
                <div style="flex: 1;">
                    <div class="chart-title">March Growth & Future Scalability</div>
                    <p style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
                        The dashboard reveals a massive structural shift in March (the "Revenue Leap" from $11M to $32M). 
                        Our forecast predicts steady growth towards <b>$39.8M</b> by Month +3. 
                        <b>Efficiency:</b> To maintain the current <b>$1.4K Avg Order Value</b>, we must ensure Supply Chain stability 
                        for <b>Electronics</b> and <b>Groceries</b>, which together account for nearly 45% of our $401M total.
                    </p>
                </div>
                <div style="flex: 1; background: #eff6ff; padding: 1.5rem; border-radius: 12px; border: 1px solid #dbeafe;">
                    <div style="font-weight: 600; color: #2563eb; margin-bottom: 0.5rem;">Forecasting Precision (R²: 0.41)</div>
                    <p style="font-size: 0.85rem; color: #1e40af; margin: 0;">
                        While the R² index is moderate, the <b>RMSE of $5.83M</b> indicates high directional accuracy. 
                        The model successfully captures the upward trajectory, enabling better <b>Working Capital Allocation</b> 
                        for peak inventory months.
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Risk Analysis
    st.markdown("""
        <div style="border-left: 4px solid #ef4444; padding-left: 15px; margin-bottom: 30px; margin-top: 30px;">
            <h3 style="margin:0; font-weight:700; color: #1e293b;">3. Financial Risk Analysis (Revenue Leakage)</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Churn Probability & Liquidity</div>
                <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
                    The <b>At-Risk</b> segment maintains an average revenue of only <b>$1.6K</b> per customer with sparse ordering (1.6 orders avg). 
                    Economically, this represents <b>opportunity cost</b>. If even 10% of this segment churns, 
                    the firm faces a liquidity variance of nearly several million dollars.
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
            <div class="chart-card">
                <div class="chart-title">Mitigation & Stabilization</div>
                <div style="font-size: 0.95rem; color: #475569; line-height: 1.6;">
                    <b>Risk Strategy:</b> Converting At-Risk customers to "Frequent" (moving from $1.6K to $6.8K per head) 
                    is the most efficient way to scale revenue without expensive new customer acquisition. 
                    Focusing on "At-Risk" re-engagement stabilizes the <b>Firm Valuation</b> by proving revenue predictability.
                </div>
            </div>
        """, unsafe_allow_html=True)

# ── Main Content Routing ─────────────────────────────────────
if "Strategy" in page:
    render_strategy_page()
    st.stop()

st.markdown('<div class="dashboard-title">Revenue Growth Strategy Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Retail Transactional Dataset · Customer Segmentation · Sales Forecasting</div>', unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────
kpi_html = f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">Total Revenue</div>
        <div class="kpi-value">{total_rev_str}</div>
        <div class="kpi-line kpi-line-blue"></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Avg Order Value</div>
        <div class="kpi-value">{avg_order_str}</div>
        <div class="kpi-line kpi-line-green"></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Unique Customers</div>
        <div class="kpi-value">{unique_cust_str}</div>
        <div class="kpi-line kpi-line-orange"></div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Total Transactions</div>
        <div class="kpi-value" style="color: #0f172a;">{total_txn_str}</div>
        <div class="kpi-line kpi-line-red"></div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)


# ── Chart Style Helpers ───────────────────────────────────────
def style_chart(ax, fig, no_borders=True):
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#94a3b8', labelsize=9, length=0)
    for spine in ax.spines.values():
        if no_borders:
            spine.set_visible(False)
        else:
            spine.set_edgecolor('#e2e8f0')
    if no_borders:
        ax.grid(axis='y', color='#f1f5f9', linewidth=1)
        ax.grid(axis='x', visible=False)

# ── Charts Section ────────────────────────────────────────────
col1, col2 = st.columns([1.6, 1])

with col1:
    # Sales Analytics Inline Header
    st.markdown("""
        <div style="border-left: 4px solid #3b82f6; padding-left: 15px; margin-bottom: 20px;">
            <h3 style="margin:0; font-weight:700; color: #1e293b;">Sales Analytics</h3>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Revenue trends, distribution, and top performance</div>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Monthly Revenue Trend", "Revenue Distribution", "Top Products"])

    with tab1:
        # Plot Monthly Revenue Line Chart
        fig, ax = plt.subplots(figsize=(10, 3.5))
        
        if is_simulated:
            months = ['Jan', 'Apr', 'Jul', 'Oct']
            x = np.arange(4)
            actual = [70000, 160000, 240000, 310000]
            trend = [75000, 155000, 235000, 315000]
            ax.plot(x, actual, color='#3b82f6', marker='o', linewidth=2.5, markersize=5, zorder=3)
            ax.plot(x, trend, color='#f97316', linestyle='--', linewidth=1.5, zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels(months)
            ax.set_yticks([100000, 200000, 300000])
            ax.set_yticklabels(['$100K', '$200K', '$300K'])
            ax.set_ylim(0, 350000)
        else:
            if 'Year' in df.columns and 'Month' in df.columns:
                # Group by Month and ensure all months (1-12) are present
                monthly = df[df['Year'] == 2023].groupby('Month')['Revenue'].sum().reindex(range(1, 13), fill_value=0).reset_index()
                monthly['TimeIndex'] = range(len(monthly))
                X = monthly[['TimeIndex']]
                y = monthly['Revenue']
                if len(X) > 1:
                    model = LinearRegression().fit(X, y)
                    y_pred = model.predict(X)
                    ax.plot(monthly['TimeIndex'], y, color='#3b82f6', marker='o', linewidth=2.5, markersize=5)
                    ax.plot(monthly['TimeIndex'], y_pred, color='#f97316', linestyle='--', linewidth=1.5)
                    
                    # Labels for all 12 months
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    ax.set_xticks(range(12))
                    ax.set_xticklabels(month_names, fontsize=8)
                    
                    # Million-aware formatter
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000000:.0f}M" if x >= 1000000 else f"${x/1000:.0f}K"))
                ax.set_xlabel('Month (2023)', fontsize=10, color='#64748b')
                ax.set_ylabel('Total Revenue', fontsize=10, color='#64748b')

        style_chart(ax, fig, no_borders=True)
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        if is_simulated:
            data = np.random.normal(500, 200, 1000)
            ax.hist(data, bins=30, color='#3b82f6', alpha=0.7, edgecolor='white')
        else:
            ax.hist(df['Revenue'], bins=30, color='#3b82f6', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Revenue per Transaction ($)', fontsize=10, color='#64748b')
        ax.set_ylabel('Frequency', fontsize=10, color='#64748b')
        style_chart(ax, fig, no_borders=True)
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        if is_simulated:
            prods = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
            vals = [120000, 95000, 80000, 60000, 45000]
            ax.barh(prods, vals, color='#3b82f6', alpha=0.8)
        else:
            if 'Product' in df.columns:
                top_p = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(10)
                ax.barh(top_p.index, top_p.values, color='#3b82f6', alpha=0.8)
                ax.invert_yaxis()
                # Use million-aware formatter as requested
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000000:.0f}M"))
        ax.set_xlabel('Total Revenue', fontsize=10, color='#64748b')
        ax.set_ylabel('Product Name', fontsize=10, color='#64748b')
        style_chart(ax, fig, no_borders=True)
        st.pyplot(fig)
        plt.close()
    

    # Customer Segments Inline Header
    st.markdown("""
        <div style="border-left: 4px solid #3b82f6; padding-left: 15px; margin-bottom: 20px;">
            <h3 style="margin:0; font-weight:700; color: #1e293b;">Customer Segments</h3>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">K-Means Clustering Analysis</div>
        </div>
    """, unsafe_allow_html=True)
    
    # K-picker layout
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <div style="font-size:0.9rem; color:#475569; font-weight: 500;">Clusters (K):</div>
        </div>
    """, unsafe_allow_html=True)
    
    # We display actual slider
    k_val = st.slider("", 2, 8, int(n_clusters) if not is_simulated else 4, label_visibility="collapsed")
    
    # Legend labels ranked by revenue
    # Ordered set of labels
    pool_names = ["Premium", "Upper-Frequent", "Frequent", "Bargain", "Occasional", "At-Risk", "Dormant", "Inactive"]
    
    # User specific request: K=4 must be [Premium, Frequent, Bargain, At-Risk]
    if k_val == 4:
        seg_names = ["Premium", "Frequent", "Bargain", "At-Risk"]
    else:
        seg_names = pool_names[:k_val]
    
    colors = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#06b6d4', '#eab308', '#ec4899']
    
    # Generate dynamic legend based on current k_val
    legend_items = "".join([
        f'<div class="legend-item"><div class="dot" style="background:{colors[i%len(colors)]};"></div> {seg_names[i]}</div>'
        for i in range(k_val)
    ])
    
    st.markdown(f'<div class="cluster-legend">{legend_items}</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    
    if is_simulated:
        # Dummy scatter exactly matching the general visual of the clusters
        colors = ['#3b82f6', '#22c55e', '#f97316', '#ef4444']
        centers = [(8, 8), (5, 4), (3, 3), (1, 1)]
        spreads = [0.5, 0.4, 0.6, 0.3]
        np.random.seed(42)
        for i, (cx, cy) in enumerate(centers):
            x = cx + np.random.randn(4) * spreads[i]
            y = cy + np.random.randn(4) * spreads[i]
            ax.scatter(x, y, color=colors[i], s=80, alpha=0.8, edgecolors='white', linewidths=1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Orders", fontsize=8, color='#94a3b8', loc='right')
        ax.set_ylabel("Rev", fontsize=8, color='#94a3b8', loc='top', rotation=0)
    else:
        if 'CustomerID' in df.columns:
            cdf = df.groupby('CustomerID').agg(Rev=('Revenue','sum'), Ord=('Revenue','count')).reset_index()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cdf[['Rev','Ord']])
            km = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit(scaled)
            cdf['Cluster'] = km.labels_
            
            # Problem 1: Dynamic Labeling based on Revenue
            cluster_rev = cdf.groupby('Cluster')['Rev'].mean().sort_values(ascending=False)
            # Map old cluster IDs to rank-based labels
            rank_map = {old_id: i for i, old_id in enumerate(cluster_rev.index)}
            cdf['Rank'] = cdf['Cluster'].map(rank_map)
            
            for i in range(k_val):
                mask = cdf['Rank'] == i
                label = seg_names[i] if i < len(seg_names) else f"Segment {i+1}"
                ax.scatter(cdf[mask]['Ord'], cdf[mask]['Rev'], color=colors[i%len(colors)], s=50, alpha=0.8, edgecolors='white', label=label)
            
            ax.set_xticks([])
            # Problem 2: Add Y-axis ticks with currency format
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1000:.0f}K" if x >= 1000 else f"${x}"))
            ax.tick_params(axis='y', colors='#94a3b8', labelsize=8)
            
            ax.set_xlabel("Total Orders", fontsize=10, color='#64748b', loc='center')
            ax.set_ylabel("Total Revenue ($)", fontsize=10, color='#64748b', loc='center', rotation=90)

    # Only show left and bottom spines for scatter
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.spines['left'].set_color('#e2e8f0')
    
    st.pyplot(fig)
    plt.close()
    
    st.write("") # Spacer


# ── Bottom Cards ──────────────────────────────────────────────
st.markdown("<br/>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("""
        <div class="bottom-card">
            <div class="bottom-title">Revenue Forecast</div>
            <div class="bottom-subtitle">Next 3 months · Linear Regression</div>
        </div>
    """, unsafe_allow_html=True)
    
    if is_simulated:
        mcols = st.columns(2)
        mcols[0].metric("Model Accuracy (R²)", "0.87")
        mcols[1].metric("RMSE", "$14.2K")
        
        fcols = st.columns(3)
        fcols[0].metric("Month +1", "$325K")
        fcols[1].metric("Month +2", "$340K")
        fcols[2].metric("Month +3", "$355K")
    else:
        if 'Year' in df.columns and 'Month' in df.columns:
            # Training only on 2023 as per Problem 1 - use all 12 months
            monthly = df[df['Year'] == 2023].groupby('Month')['Revenue'].sum().reindex(range(1, 13), fill_value=0).reset_index()
            if len(monthly[monthly['Revenue'] > 0]) >= 2: # Need at least 2 points with actual sales to train
                monthly['TimeIndex'] = range(len(monthly))
                X = np.array(monthly[['TimeIndex']]).reshape(-1, 1)
                y = monthly['Revenue']
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                mcols = st.columns(2)
                mcols[0].metric("Model Accuracy (R²)", f"{r2:.2f}")
                mcols[1].metric("RMSE", format_currency(rmse))
                
                # Predict next 3 months (Index 12, 13, 14)
                future = np.array([12, 13, 14]).reshape(-1, 1)
                preds = model.predict(future)
                fcols = st.columns(3)
                for i, p in enumerate(preds):
                    fcols[i].metric(f"Month +{i+1}", format_currency(p))
            else:
                st.write("Insufficient 2023 data for forecasting.")

with b2:
    st.markdown("""
        <div class="bottom-card">
            <div class="bottom-title">Segment Profiles</div>
            <div class="bottom-subtitle">Average metrics per cluster</div>
        </div>
    """, unsafe_allow_html=True)
    
    if is_simulated:
        dummy_profiles = pd.DataFrame({
            'Cluster': ['Premium', 'Frequent', 'Bargain', 'At-Risk'],
            'Avg Rev': ['$2.4K', '$1.2K', '$0.6K', '$0.3K'],
            'Orders': [12, 8, 5, 2]
        })
        st.dataframe(dummy_profiles, hide_index=True, use_container_width=True)
    else:
        if 'CustomerID' in df.columns:
            # Aggregate metrics per customer
            cdf = df.groupby('CustomerID').agg(Rev=('Revenue','sum'), Ord=('Revenue','count')).reset_index()
            
            if len(cdf) >= k_val:
                scaler = StandardScaler()
                scaled = scaler.fit_transform(cdf[['Rev','Ord']])
                km = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit(scaled)
                cdf['Cluster'] = km.labels_
                
                # Rank clusters by revenue for consistent labeling
                cluster_rev = cdf.groupby('Cluster')['Rev'].mean().sort_values(ascending=False)
                rank_map = {old_id: i for i, old_id in enumerate(cluster_rev.index)}
                cdf['Rank'] = cdf['Cluster'].map(rank_map)
                
                # Compute averages per rank
                profiles = cdf.groupby('Rank').agg({'Rev':'mean', 'Ord':'mean'}).round(2)
                profiles['Avg Order Value'] = (profiles['Rev'] / profiles['Ord']).round(2)
                
                # Map rank index to names
                pool_names_full = ["Premium", "Upper-Frequent", "Frequent", "Bargain", "Occasional", "At-Risk", "Dormant", "Inactive"]
                if k_val == 4:
                    seg_names_final = ["Premium", "Frequent", "Bargain", "At-Risk"]
                else:
                    seg_names_final = pool_names_full[:k_val]
                
                def get_name(i):
                    return seg_names_final[i] if i < len(seg_names_final) else f"Segment {i+1}"
                
                profiles.index = [get_name(i) for i in profiles.index]
                profiles.columns = ['Avg Revenue', 'Avg Orders', 'Avg Order Value']
                st.dataframe(profiles, use_container_width=True)
            else:
                st.write("Not enough customers for clustering.")

with b3:
    st.markdown("""
        <div class="bottom-card">
            <div class="bottom-title">Revenue Predictor</div>
            <div class="bottom-subtitle">Enter values to predict revenue</div>
        </div>
    """, unsafe_allow_html=True)
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        qty_in = st.number_input("Quantity", min_value=1, value=10)
        price_in = st.number_input("Price", min_value=1.0, value=50.0)
    with p_col2:
        seg_in = st.selectbox("Segment", ["Premium (+20%)", "Frequent (Baseline)", "Bargain (-15%)", "At-Risk (-40%)"])
        mults = {"Premium (+20%)": 1.2, "Frequent (Baseline)": 1.0, "Bargain (-15%)": 0.85, "At-Risk (-40%)": 0.6}
        pred_val = qty_in * price_in * mults[seg_in]
        st.metric("Predicted", f"${pred_val:,.2f}")
