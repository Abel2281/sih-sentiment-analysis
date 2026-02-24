import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

client = Groq(api_key=api_key)

CONFIDENCE_THRESHOLD = 0.65 

def get_groq_insight(comments, law_name, section_name, sentiment_type):
    """
    Sends the comments to Groq (Llama-3) for an instant 1-sentence summary.
    """
    if not comments:
        return "No comments available to analyze."
    
    comments_text = "\n- ".join(comments[:15])
    
    prompt = f"""
    You are a legal analyst. Here are some user comments regarding '{section_name}' of the law: "{law_name}".
    The general sentiment is {sentiment_type.upper()}.
    
    USER COMMENTS:
    {comments_text}
    
    TASK:
    Summarize the MAIN REASON for this {sentiment_type} sentiment in exactly one clear, professional sentence.
    Start with "Citizens feel..." or "The opposition is due to..." or "Support is driven by..."
    Do not mention individual users.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a concise legal data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Could not generate insight due to API Error: {e}"

def analyze_insights(csv_path, law_name):
    print(f"Groq Insight Engine Running for: {law_name}...")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df = df[df['Sentiment_Label'] != "N/A"]
    if df.empty: 
        print("No valid data found in CSV.")
        return

    summary = df.groupby(['Linked_Clause', 'Sentiment_Label']).size().unstack(fill_value=0)
    for col in ['negative', 'positive']:
        if col not in summary.columns: summary[col] = 0

    print("\n --- GENERATIVE INSIGHT REPORT ---")

    if summary['negative'].sum() > 0:
        target_section = summary['negative'].idxmax()
        count = summary['negative'].max()
        
        angry_comments = df[
            (df['Linked_Clause'] == target_section) & 
            (df['Sentiment_Label'] == 'negative') &
            (df['Sentiment_Score'] > CONFIDENCE_THRESHOLD)
        ]['Comment'].tolist()
        
        if not angry_comments:
            angry_comments = df[
                (df['Linked_Clause'] == target_section) & 
                (df['Sentiment_Label'] == 'negative')
            ]['Comment'].tolist()
        
        insight = get_groq_insight(angry_comments, law_name, target_section, "negative")
        
        print(f"   • Volume: {count} negative comments")
        print(f"   • Insight: {insight}")

    if summary['positive'].sum() > 0:
        target_section = summary['positive'].idxmax()
        count = summary['positive'].max()
        
        happy_comments = df[
            (df['Linked_Clause'] == target_section) & 
            (df['Sentiment_Label'] == 'positive') &
            (df['Sentiment_Score'] > CONFIDENCE_THRESHOLD)
        ]['Comment'].tolist()
        
        if not happy_comments:
            happy_comments = df[
                (df['Linked_Clause'] == target_section) & 
                (df['Sentiment_Label'] == 'positive')
            ]['Comment'].tolist()
        
        insight = get_groq_insight(happy_comments, law_name, target_section, "positive")
        
        print(f"   • Volume: {count} positive comments")
        print(f"   • Insight: {insight}")

if __name__ == "__main__":
    input_file = os.path.join("data", "processed", "sentiment_analysis_result.csv")
    TEST_LAW_NAME = "Personal Data Protection Bill, 2019"
    analyze_insights(input_file, TEST_LAW_NAME)