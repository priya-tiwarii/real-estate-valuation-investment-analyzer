"""
Real Estate Valuation & Investment Analyzer
Professional Streamlit Dashboard - Production Ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PATH CONFIGURATION FOR DEPLOYMENT
# ============================================

# Get the base directory (works both locally and on Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Real Estate Valuation & Investment Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 1rem;
        padding: 1.2rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1f2937;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 0.75rem;
        border-radius: 0.75rem;
        text-align: center;
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    
    .recommendation-consider {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 0.75rem;
        border-radius: 0.75rem;
        text-align: center;
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);
    }
    
    .recommendation-hold {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 0.75rem;
        border-radius: 0.75rem;
        text-align: center;
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
    }
    
    .recommendation-avoid {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 0.75rem;
        border-radius: 0.75rem;
        text-align: center;
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-size: 0.85rem;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA AND MODELS
# ============================================

@st.cache_data
def load_data():
    """Load all processed data"""
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, 'processed_training_data.csv'))
        investment_df = pd.read_csv(os.path.join(DATA_DIR, 'investment_analysis.csv'))
        return df, investment_df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Make sure you have run all notebooks to generate the data files.")
        return None, None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'investment_model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        return None, None

# Load data
df, investment_df = load_data()
model, scaler = load_models()

if df is None:
    st.stop()

# ============================================
# SIDEBAR - PROFESSIONAL NAVIGATION
# ============================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🏠</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #667eea;">Valuation & Investment</div>
            <div style="font-size: 0.8rem; color: #6b7280;">Intelligence Platform</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation - FIXED: Added label with visibility collapsed
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Property Analyzer", "💰 Portfolio Optimizer", "📈 Market Insights", "📄 Investment Report"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats in Sidebar
    st.markdown("### 🎯 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Avg Investment Score", f"{investment_df['Investment_Score'].mean():.1f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg ROI", f"{investment_df['ROI_Potential'].mean():.1f}%")
    with col2:
        st.metric("Buy Recs", f"{(investment_df['Recommendation'] == 'BUY').sum()}")
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🔧 Filters")
    min_price = st.slider(
        "💰 Min Price",
        min_value=int(df['SalePrice'].min()),
        max_value=int(df['SalePrice'].max()),
        value=int(df['SalePrice'].min()),
        format="$%d"
    )
    
    max_price = st.slider(
        "💰 Max Price",
        min_value=int(df['SalePrice'].min()),
        max_value=int(df['SalePrice'].max()),
        value=int(df['SalePrice'].max()),
        format="$%d"
    )
    
    min_score = st.slider(
        "⭐ Min Investment Score",
        min_value=0,
        max_value=100,
        value=50
    )
    
    # Apply filters
    filtered_df = df[
        (df['SalePrice'] >= min_price) & 
        (df['SalePrice'] <= max_price)
    ].copy()
    
    filtered_investment = investment_df[
        (investment_df['SalePrice'] >= min_price) & 
        (investment_df['SalePrice'] <= max_price) &
        (investment_df['Investment_Score'] >= min_score)
    ].copy()
    
    st.markdown("---")
    st.caption("🚀 Powered by Machine Learning")

# ============================================
# DASHBOARD PAGE
# ============================================
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">Real Estate Valuation & Investment Analyzer</h1>', unsafe_allow_html=True)
    
    # Hero Section with Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🏘️</div>
                <div class="metric-label">Total Properties</div>
                <div class="metric-value">{len(filtered_df):,}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💰</div>
                <div class="metric-label">Average Price</div>
                <div class="metric-value">${filtered_df['SalePrice'].mean():,.0f}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⭐</div>
                <div class="metric-label">Avg Investment Score</div>
                <div class="metric-value">{filtered_investment['Investment_Score'].mean():.1f}/100</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-label">Avg ROI Potential</div>
                <div class="metric-value">{filtered_investment['ROI_Potential'].mean():.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">📊 Investment Score Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(
            filtered_investment,
            x='Investment_Score',
            nbins=30,
            color_discrete_sequence=['#667eea'],
            title="",
            labels={'Investment_Score': 'Investment Score', 'count': 'Number of Properties'}
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            showlegend=False,
            bargap=0.05,
            font=dict(family="Inter", size=12)
        )
        fig.add_vline(x=filtered_investment['Investment_Score'].mean(), line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: {filtered_investment['Investment_Score'].mean():.1f}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">📉 Risk vs Return Analysis</p>', unsafe_allow_html=True)
        fig = px.scatter(
            filtered_investment,
            x='Risk_Score',
            y='ROI_Potential',
            color='Investment_Score',
            size='SalePrice',
            hover_data=['Property_Index'],
            title="",
            labels={'Risk_Score': 'Risk Score (lower is better)', 'ROI_Potential': 'ROI Potential (%)'},
            color_continuous_scale='RdYlGn',
            size_max=20
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            font=dict(family="Inter", size=12)
        )
        fig.add_hline(y=filtered_investment['ROI_Potential'].mean(), line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">🏆 Top 10 Investment Opportunities</p>', unsafe_allow_html=True)
        top_props = filtered_investment.nlargest(10, 'Investment_Score')
        fig = px.bar(
            top_props,
            x='Property_Index',
            y='Investment_Score',
            color='Risk_Score',
            title="",
            labels={'Property_Index': 'Property Index', 'Investment_Score': 'Investment Score'},
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">💎 Price vs Investment Score</p>', unsafe_allow_html=True)
        fig = px.scatter(
            filtered_investment,
            x='SalePrice',
            y='Investment_Score',
            color='Risk_Score',
            size='ROI_Potential',
            hover_data=['Property_Index'],
            title="",
            labels={'SalePrice': 'Property Price ($)', 'Investment_Score': 'Investment Score'},
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation Summary Section
    st.markdown('<p class="sub-header">📊 Investment Recommendation Summary</p>', unsafe_allow_html=True)
    
    rec_counts = filtered_investment['Recommendation'].value_counts()
    
    cols = st.columns(5)
    rec_data = [
        ('STRONG BUY', rec_counts.get('STRONG BUY', 0), '#10b981'),
        ('BUY', rec_counts.get('BUY', 0), '#059669'),
        ('CONSIDER', rec_counts.get('CONSIDER', 0), '#f59e0b'),
        ('HOLD', rec_counts.get('HOLD', 0), '#3b82f6'),
        ('AVOID', rec_counts.get('AVOID', 0), '#ef4444')
    ]
    
    for col, (rec, count, color) in zip(cols, rec_data):
        with col:
            pct = (count / len(filtered_investment) * 100) if len(filtered_investment) > 0 else 0
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%); 
                            padding: 1rem; 
                            border-radius: 1rem; 
                            text-align: center;
                            color: white;
                            margin: 0.25rem;">
                    <div style="font-size: 1.5rem; font-weight: 800;">{count}</div>
                    <div style="font-size: 0.8rem; font-weight: 600;">{rec}</div>
                    <div style="font-size: 0.7rem; opacity: 0.9;">{pct:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

# ============================================
# PROPERTY ANALYZER PAGE
# ============================================
elif page == "🔍 Property Analyzer":
    st.markdown('<h1 class="main-header">Property Analyzer</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<p class="sub-header">Select Property</p>', unsafe_allow_html=True)
        property_options = filtered_df.index.tolist()
        selected_property = st.selectbox("", property_options, format_func=lambda x: f"Property #{x}", label_visibility="collapsed")
        
        if selected_property is not None:
            prop_data = filtered_df.loc[selected_property]
            inv_data = filtered_investment[filtered_investment['Property_Index'] == selected_property]
            
            if len(inv_data) > 0:
                inv_data = inv_data.iloc[0]
                
                st.markdown("---")
                st.markdown('<p class="sub-header">📋 Property Details</p>', unsafe_allow_html=True)
                
                details = {
                    "💰 Price": f"${prop_data['SalePrice']:,.0f}",
                    "📅 Year Built": f"{prop_data['YearBuilt']:.0f}",
                    "⏳ Property Age": f"{prop_data['PropertyAge']:.0f} years",
                    "📏 Lot Area": f"{prop_data['LotArea']:,.0f} sq ft",
                    "🔧 Condition": f"{prop_data['OverallCond']:.0f}/10",
                    "🏗️ Basement": f"{prop_data['TotalBsmtSF']:,.0f} sq ft"
                }
                
                for label, value in details.items():
                    st.markdown(f"**{label}:** {value}")
    
    with col2:
        if selected_property is not None and inv_data is not None:
            st.markdown('<p class="sub-header">📊 Investment Analysis</p>', unsafe_allow_html=True)
            
            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Investment Score", f"{inv_data['Investment_Score']:.1f}/100", 
                         delta="Good" if inv_data['Investment_Score'] > 70 else "Average")
            with m2:
                st.metric("Risk Score", f"{inv_data['Risk_Score']:.1f}/100",
                         delta="Low Risk" if inv_data['Risk_Score'] < 30 else "Moderate Risk",
                         delta_color="inverse")
            with m3:
                st.metric("ROI Potential", f"{inv_data['ROI_Potential']:.1f}%")
            
            # Recommendation Card
            rec = inv_data['Recommendation']
            rec_class = {
                'STRONG BUY': 'recommendation-buy',
                'BUY': 'recommendation-buy',
                'CONSIDER': 'recommendation-consider',
                'HOLD': 'recommendation-hold',
                'AVOID': 'recommendation-avoid'
            }.get(rec, 'recommendation-hold')
            
            st.markdown(f"""
                <div class="{rec_class}" style="padding: 1.2rem; margin: 1rem 0; text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 800;">📌 {rec}</div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem;">{inv_data['Recommendation_Reason']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📊 Key Metrics")
                st.metric("Estimated Rent", f"${inv_data['Estimated_Rent']:,.0f}/month")
                st.metric("Gross Yield", f"{inv_data['Gross_Yield']:.2f}%")
            with col2:
                st.markdown("### 📈 Potential")
                value_add = "High" if inv_data['ROI_Potential'] > 60 else "Medium" if inv_data['ROI_Potential'] > 40 else "Low"
                st.metric("Value Add Potential", value_add)
                market_pos = "Premium" if inv_data['Investment_Score'] > 70 else "Standard" if inv_data['Investment_Score'] > 50 else "Budget"
                st.metric("Market Position", market_pos)

# ============================================
# PORTFOLIO OPTIMIZER PAGE
# ============================================
elif page == "💰 Portfolio Optimizer":
    st.markdown('<h1 class="main-header">Portfolio Optimizer</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Investment Parameters</p>', unsafe_allow_html=True)
        
        budget = st.number_input(
            "💰 Investment Budget ($)",
            min_value=100000,
            max_value=5000000,
            value=1000000,
            step=100000,
            format="%d"
        )
        
        min_score = st.slider(
            "⭐ Minimum Investment Score",
            min_value=0,
            max_value=100,
            value=60,
            help="Only properties with score above this will be considered"
        )
        
        max_risk = st.slider(
            "⚠️ Maximum Risk Score",
            min_value=0,
            max_value=100,
            value=50,
            help="Only properties with risk below this will be considered"
        )
        
        if st.button("🎯 Optimize Portfolio", type="primary", use_container_width=True):
            with st.spinner("Optimizing portfolio..."):
                candidates = filtered_investment[
                    (filtered_investment['Investment_Score'] >= min_score) & 
                    (filtered_investment['Risk_Score'] <= max_risk) &
                    (filtered_investment['Recommendation'].isin(['STRONG BUY', 'BUY', 'CONSIDER']))
                ].copy()
                
                candidates = candidates.sort_values('Investment_Score', ascending=False)
                
                portfolio = []
                remaining = budget
                
                for _, prop in candidates.iterrows():
                    price = prop['SalePrice']
                    if price <= remaining:
                        portfolio.append(prop)
                        remaining -= price
                    if remaining < 50000:
                        break
                
                st.session_state['portfolio'] = portfolio
                st.session_state['remaining'] = remaining
                st.session_state['budget'] = budget
                
                st.success(f"✅ Portfolio optimized with {len(portfolio)} properties!")
    
    with col2:
        if 'portfolio' in st.session_state and len(st.session_state['portfolio']) > 0:
            portfolio = st.session_state['portfolio']
            remaining = st.session_state['remaining']
            
            total_invested = sum(p['SalePrice'] for p in portfolio)
            avg_score = np.mean([p['Investment_Score'] for p in portfolio])
            avg_risk = np.mean([p['Risk_Score'] for p in portfolio])
            avg_roi = np.mean([p['ROI_Potential'] for p in portfolio])
            
            st.markdown('<p class="sub-header">📊 Portfolio Summary</p>', unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Investment", f"${total_invested:,.0f}")
            with col_b:
                st.metric("Properties", len(portfolio))
            with col_c:
                st.metric("Remaining", f"${remaining:,.0f}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Avg Investment Score", f"{avg_score:.1f}")
            with col_b:
                st.metric("Avg ROI Potential", f"{avg_roi:.1f}%")
    
    if 'portfolio' in st.session_state and len(st.session_state['portfolio']) > 0:
        st.markdown("---")
        st.markdown('<p class="sub-header">🏢 Portfolio Properties</p>', unsafe_allow_html=True)
        
        portfolio_df = pd.DataFrame(st.session_state['portfolio'])
        display_df = portfolio_df[['Property_Index', 'SalePrice', 'Investment_Score', 'Risk_Score', 'ROI_Potential', 'Recommendation']].copy()
        display_df.columns = ['Property', 'Price', 'Investment Score', 'Risk Score', 'ROI Potential', 'Recommendation']
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:,.0f}")
        display_df['ROI Potential'] = display_df['ROI Potential'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df.style.background_gradient(subset=['Investment Score'], cmap='RdYlGn'), use_container_width=True)
        
        # Portfolio visualization
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Investment Score Distribution', 'ROI Potential'))
        
        fig.add_trace(
            go.Bar(x=portfolio_df['Property_Index'], y=portfolio_df['Investment_Score'], 
                   marker_color='#667eea', name='Investment Score'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=portfolio_df['Property_Index'], y=portfolio_df['ROI_Potential'], 
                   marker_color='#f59e0b', name='ROI Potential'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# MARKET INSIGHTS PAGE
# ============================================
elif page == "📈 Market Insights":
    st.markdown('<h1 class="main-header">Market Insights</h1>', unsafe_allow_html=True)
    
    # Price Trends
    st.markdown('<p class="sub-header">📅 Price Trends by Decade</p>', unsafe_allow_html=True)
    
    df['Decade'] = (df['YearBuilt'] // 10) * 10
    decade_stats = df.groupby('Decade').agg({
        'SalePrice': ['mean', 'median', 'count']
    }).round(0)
    decade_stats.columns = ['Mean Price', 'Median Price', 'Count']
    decade_stats = decade_stats.reset_index()
    
    fig = px.line(
        decade_stats,
        x='Decade',
        y=['Mean Price', 'Median Price'],
        title="",
        labels={'value': 'Price ($)', 'decade': 'Decade Built', 'variable': 'Metric'},
        color_discrete_map={'Mean Price': '#667eea', 'Median Price': '#f59e0b'}
    )
    fig.update_layout(plot_bgcolor='white', height=500, font=dict(family="Inter"))
    st.plotly_chart(fig, use_container_width=True)
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">🗺️ Investment Opportunity Heatmap</p>', unsafe_allow_html=True)
        
        df['Price_Bin'] = pd.qcut(df['SalePrice'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['Score_Bin'] = pd.cut(investment_df['Investment_Score'], bins=5, labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        
        heatmap_data = df.groupby(['Price_Bin', 'Score_Bin']).size().unstack(fill_value=0)
        
        fig = px.imshow(
            heatmap_data,
            title="",
            labels=dict(x="Investment Score", y="Price Segment", color="Number of Properties"),
            color_continuous_scale='YlOrRd',
            aspect="auto"
        )
        fig.update_layout(height=450, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<p class="sub-header">📊 Feature Correlation Heatmap</p>', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="",
            labels=dict(color="Correlation"),
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto=True,
            aspect="auto"
        )
        fig.update_layout(height=450, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# INVESTMENT REPORT PAGE
# ============================================
elif page == "📄 Investment Report":
    st.markdown('<h1 class="main-header">Investment Report</h1>', unsafe_allow_html=True)
    
    # Summary Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties Analyzed", f"{len(filtered_investment):,}")
    with col2:
        st.metric("Average Investment Score", f"{filtered_investment['Investment_Score'].mean():.1f}/100")
    with col3:
        st.metric("Total ROI Potential", f"{filtered_investment['ROI_Potential'].sum():,.0f}%")
    with col4:
        st.metric("Average Gross Yield", f"{filtered_investment['Gross_Yield'].mean():.2f}%")
    
    st.markdown("---")
    
    # Top Opportunities Table
    st.markdown('<p class="sub-header">🏆 Top 20 Investment Opportunities</p>', unsafe_allow_html=True)
    
    top_20 = filtered_investment.nlargest(20, 'Investment_Score')
    display_top = top_20[['Property_Index', 'SalePrice', 'Investment_Score', 'Risk_Score', 'ROI_Potential', 'Gross_Yield', 'Recommendation']].copy()
    display_top['SalePrice'] = display_top['SalePrice'].apply(lambda x: f"${x:,.0f}")
    display_top['ROI_Potential'] = display_top['ROI_Potential'].apply(lambda x: f"{x:.1f}%")
    display_top['Gross_Yield'] = display_top['Gross_Yield'].apply(lambda x: f"{x:.2f}%")
    display_top.columns = ['Property', 'Price', 'Investment Score', 'Risk Score', 'ROI Potential', 'Gross Yield', 'Recommendation']
    
    st.dataframe(display_top.style.background_gradient(subset=['Investment Score'], cmap='RdYlGn'), use_container_width=True)
    
    st.markdown("---")
    
    # Download Section
    st.markdown('<p class="sub-header">📥 Download Full Report</p>', unsafe_allow_html=True)
    
    csv = filtered_investment.to_csv(index=False)
    st.download_button(
        label="📊 Download Complete Investment Analysis (CSV)",
        data=csv,
        file_name="real_estate_investment_report.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.caption("The CSV includes all properties with their investment scores, risk ratings, ROI potential, and recommendations.")

# ============================================
# FOOTER
# ============================================
st.markdown("""
    <div class="footer">
        <div>🏠 Real Estate Valuation & Investment Analyzer | Powered by Machine Learning</div>
        <div style="font-size: 0.7rem; margin-top: 0.5rem;">Data-driven insights for smarter real estate investments</div>
    </div>
""", unsafe_allow_html=True)