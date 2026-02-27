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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stButton>button {width: 100%; border-radius: 8px; height: 50px; font-weight: bold;}
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
        st.title("⚖️ Policy Sentiment Intelligence")
        st.markdown("<p style='color: gray; font-size: 18px;'>Upload public comments to generate a structured AI impact report.</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        law_name = st.text_input("Target Law / Policy Name", value="Personal Data Protection Bill, 2019")
        uploaded_file = st.file_uploader("Upload Comments (CSV)", type=['csv'])
        
        st.write("")
        if uploaded_file is not None:
            _, btn_col, _ = st.columns([1, 2, 1])
            with btn_col:
                if st.button("🚀 Analyze Data", type="primary"):
                    st.session_state.file_bytes = uploaded_file.getvalue()
                    st.session_state.law_name = law_name
                    set_step(2)
                    st.rerun()

elif st.session_state.step == 2:
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        st.write("")
        st.write("")
        st.write("")
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_text.markdown("### ⚙️ Running AI Pipeline (or fetching from Cache)...")
            
            clean_df, insights = run_cached_analysis(
                st.session_state.law_name, 
                st.session_state.file_bytes
            )
            
            progress_bar.progress(100)
            status_text.markdown("### ✅ Analysis Complete! Compiling Dashboard...")
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
        st.title("📊 Impact Report")
        st.subheader(f"Policy: {law_name}")
    with col_head2:
        st.write("")
        if st.button("🔄 Analyze New File"):
            set_step(1)
            st.rerun()
            
    st.divider()

    chart_col1, chart_col2 = st.columns([1, 1.5])
    
    with chart_col1:
        st.markdown("#### Overall Sentiment")
        pie_data = df['Sentiment_Label'].value_counts().reset_index()
        pie_data.columns = ['Sentiment', 'Count']
        fig_pie = px.pie(
            pie_data, names='Sentiment', values='Count', 
            color='Sentiment',
            color_discrete_map={'positive':'#00CC96', 'neutral':'#636EFA', 'negative':'#EF553B'},
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("#### Sentiment by Clause")
        bar_data = df.groupby(['Linked_Clause', 'Sentiment_Label']).size().reset_index(name='Count')
        fig_bar = px.bar(
            bar_data, x='Linked_Clause', y='Count', color='Sentiment_Label',
            barmode='group',
            color_discrete_map={'positive':'#00CC96', 'neutral':'#636EFA', 'negative':'#EF553B'}
        )
        fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), xaxis_title="", yaxis_title="Volume")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    st.markdown("#### 💡 Insights")
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.error(f"🚨 **Most Opposed: {st.session_state.insights.get('opposed_sec', 'N/A')}**")
        st.write(st.session_state.insights.get('opposed_text', 'Not enough negative data to generate insight.'))
        
    with insight_col2:
        st.success(f"✅ **Most Supported: {st.session_state.insights.get('supported_sec', 'N/A')}**")
        st.write(st.session_state.insights.get('supported_text', 'Not enough positive data to generate insight.'))