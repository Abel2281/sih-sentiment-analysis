import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import json

from src.law_fetcher import fetch_law_summary
from src.linker import link_comments_to_law
from src.sentiment_engine import analyze_sentiment
from src.insight_engine import get_groq_insight

st.set_page_config(page_title="Policy AI Dashboard", page_icon="⚖️", layout="wide")

st.markdown("""
    <style>
    /* Import modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #1a2333 100%);
        color: #f1f5f9;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Smooth Fade-in & Slide-up Page Transition */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .block-container {
        animation: fadeInUp 0.6s ease-out forwards;
        padding-top: 3rem !important;
    }

    /* Glassmorphism Inputs & Uploader */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Sleek File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: #6366f1 !important;
    }

    /* Premium Button Design */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 54px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
    }

    /* Custom HTML Insight Cards */
    .insight-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .insight-card:hover {
        transform: translateY(-4px);
    }
    .insight-opposed { border-left: 4px solid #ef4444; }
    .insight-supported { border-left: 4px solid #10b981; }
    .insight-title { 
        font-weight: 700; 
        font-size: 1.1rem; 
        margin-bottom: 12px; 
        display: flex; 
        align-items: center; 
        gap: 8px; 
    }
    .title-opposed { color: #f87171; }
    .title-supported { color: #34d399; }
    .insight-text { color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; }

    /* Loading Pulse Animation */
    @keyframes pulseText {
        0% { opacity: 0.7; text-shadow: 0 0 10px rgba(99, 102, 241, 0.3); }
        50% { opacity: 1; text-shadow: 0 0 20px rgba(99, 102, 241, 0.8); }
        100% { opacity: 0.7; text-shadow: 0 0 10px rgba(99, 102, 241, 0.3); }
    }
    h3 {
        animation: pulseText 2s infinite ease-in-out;
        color: #e0e7ff !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def run_cached_analysis(law_name, file_bytes):
    temp_path = os.path.join("data", "raw", "temp_upload.csv")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
        
    df = pd.read_csv(temp_path)
    
    status_text.markdown("### 🔍 1/4: Fetching Law Context (Groq Llama-3)...")
    progress_bar.progress(25)
    law_data = fetch_law_summary(law_name)
    
    law_context_path = os.path.join("data", "processed", "law_context.json")
    os.makedirs(os.path.dirname(law_context_path), exist_ok=True)
    with open(law_context_path, "w") as f:
        json.dump(law_data, f)

    status_text.markdown("### 🔗 2/4: Linking Comments to Clauses (MiniLM)...")
    progress_bar.progress(50)   
    linked_df = link_comments_to_law(df)

    status_text.markdown("### 🧠 3/4: Analyzing Sentiment (RoBERTa)...")
    progress_bar.progress(75)
    final_df = analyze_sentiment(linked_df)
    
    status_text.markdown("### 💡 4/4: Generating Insights...")
    progress_bar.progress(90)
    clean_df = final_df[(final_df['Sentiment_Label'] != 'N/A') & (final_df['Linked_Clause'] != 'Irrelevant')]
    summary = clean_df.groupby(['Linked_Clause', 'Sentiment_Label']).size().unstack(fill_value=0)
    for col in ['negative', 'positive']:
        if col not in summary.columns: summary[col] = 0
        
    insights = {}
    if summary['negative'].sum() > 0:
        opp_sec = summary['negative'].idxmax()
        opp_comments = clean_df[(clean_df['Linked_Clause'] == opp_sec) & (clean_df['Sentiment_Label'] == 'negative')]['Comment'].tolist()
        insights['opposed_sec'] = opp_sec
        insights['opposed_text'] = get_groq_insight(opp_comments, law_name, opp_sec, "negative")
        
    if summary['positive'].sum() > 0:
        sup_sec = summary['positive'].idxmax()
        sup_comments = clean_df[(clean_df['Linked_Clause'] == sup_sec) & (clean_df['Sentiment_Label'] == 'positive')]['Comment'].tolist()
        insights['supported_sec'] = sup_sec
        insights['supported_text'] = get_groq_insight(sup_comments, law_name, sup_sec, "positive")

    return clean_df, insights

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'final_data' not in st.session_state:
    st.session_state.final_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = {}

def set_step(step_num):
    st.session_state.step = step_num

if st.session_state.step == 1:
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        st.write("")
        st.write("")
        st.markdown("<h1 style='text-align: center; color: white;'>⚖️ Policy Sentiment Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Upload public comments to generate a structured AI impact report.</p>", unsafe_allow_html=True)
        
        st.markdown("""<div style='background: rgba(255,255,255,0.02); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05);'>""", unsafe_allow_html=True)
        
        law_name = st.text_input("Target Law / Policy Name", value="Personal Data Protection Bill, 2019")
        st.write("")
        uploaded_file = st.file_uploader("Upload Comments (CSV)", type=['csv'])
        
        st.write("")
        if uploaded_file is not None:
            st.write("")
            if st.button("🚀 Analyze Data", type="primary"):
                st.session_state.file_bytes = uploaded_file.getvalue()
                st.session_state.law_name = law_name
                set_step(2)
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.step == 2:
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        status_text = st.empty()
        st.write("")
        progress_bar = st.progress(0)
        
        try:
            status_text.markdown("<h3 style='text-align: center;'>⚙️ Initializing AI Pipeline...</h3>", unsafe_allow_html=True)
            
            clean_df, insights = run_cached_analysis(
                st.session_state.law_name, 
                st.session_state.file_bytes
            )
            
            progress_bar.progress(100)
            status_text.markdown("<h3 style='text-align: center; color: #34d399 !important; animation: none;'>✅ Analysis Complete! Compiling Dashboard...</h3>", unsafe_allow_html=True)
            time.sleep(1)
            
            st.session_state.final_data = clean_df
            st.session_state.insights = insights
            set_step(3)
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline crashed: {e}")
            if st.button("Try Again"):
                set_step(1)
                st.rerun()

elif st.session_state.step == 3:
    df = st.session_state.final_data
    law_name = st.session_state.law_name
    
    col_head1, col_head2 = st.columns([4, 1])
    with col_head1:
        st.markdown(f"<h1 style='margin-bottom: 0px;'>📊 Impact Report</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #94a3b8; font-weight: 400; animation: none; margin-top: 5px;'>Policy: {law_name}</h3>", unsafe_allow_html=True)
    with col_head2:
        st.write("")
        if st.button("🔄 Analyze New File"):
            set_step(1)
            st.rerun()
            
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;'>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1, 1.5], gap="large")
    
    with chart_col1:
        st.markdown("<h4 style='color: #e2e8f0; font-weight: 600;'>Overall Sentiment</h4>", unsafe_allow_html=True)
        pie_data = df['Sentiment_Label'].value_counts().reset_index()
        pie_data.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(
            pie_data, names='Sentiment', values='Count', 
            color='Sentiment',
            color_discrete_map={'positive':'#10b981', 'neutral':'#6366f1', 'negative':'#ef4444'},
            hole=0.45
        )
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1')
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("<h4 style='color: #e2e8f0; font-weight: 600;'>Sentiment by Clause</h4>", unsafe_allow_html=True)
        bar_data = df.groupby(['Linked_Clause', 'Sentiment_Label']).size().reset_index(name='Count')
        fig_bar = px.bar(
            bar_data, x='Linked_Clause', y='Count', color='Sentiment_Label',
            barmode='group',
            color_discrete_map={'positive':'#10b981', 'neutral':'#6366f1', 'negative':'#ef4444'}
        )
        fig_bar.update_layout(
            margin=dict(t=20, b=20, l=0, r=0),
            xaxis_title="", yaxis_title="Volume",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;'>", unsafe_allow_html=True)

    st.markdown("<h4 style='color: #e2e8f0; font-weight: 600; margin-bottom: 1.5rem;'>💡Insights</h4>", unsafe_allow_html=True)
    insight_col1, insight_col2 = st.columns(2, gap="large")
    
    opp_sec = st.session_state.insights.get('opposed_sec', 'N/A')
    opp_text = st.session_state.insights.get('opposed_text', 'Not enough negative data to generate insight.')
    sup_sec = st.session_state.insights.get('supported_sec', 'N/A')
    sup_text = st.session_state.insights.get('supported_text', 'Not enough positive data to generate insight.')

    with insight_col1:
        st.markdown(f"""
            <div class="insight-card insight-opposed">
                <div class="insight-title title-opposed">🚨 Most Opposed: {opp_sec}</div>
                <div class="insight-text">{opp_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with insight_col2:
        st.markdown(f"""
            <div class="insight-card insight-supported">
                <div class="insight-title title-supported">✅ Most Supported: {sup_sec}</div>
                <div class="insight-text">{sup_text}</div>
            </div>
        """, unsafe_allow_html=True)