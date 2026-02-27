import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import streamlit as st
import numpy as np
from scipy.special import softmax


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32

@st.cache_resource
def get_sentiment_model():
    print(f"Loading Sentiment Model ({MODEL_NAME}) into RAM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def analyze_sentiment(df):
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return None
    
    if 'Linked_Clause' not in df.columns:
        print("Error: DataFrame is missing 'Linked_Clause'. Run linker.py first!")
        return None
    
    relevant_mask = df['Linked_Clause'] != "Irrelevant"
    relevant_comments = df.loc[relevant_mask, 'Comment'].tolist()
    
    if not relevant_comments:
        print("No relevant comments found. Exiting.")
        return df
    
    df['Sentiment_Label'] = "N/A"
    df['Sentiment_Score'] = 0.0

    tokenizer, model, device = get_sentiment_model()

    id2label = model.config.id2label 
    processed_labels = []
    processed_scores = []

    print("Starting Batch Analysis")
    
    # tqdm creates the progress bar
    for i in tqdm(range(0, len(relevant_comments), BATCH_SIZE), desc="Processing Batches"):
        batch_texts = relevant_comments[i : i + BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        # Inference No Gradients needed = Faster
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-processing
        logits = outputs.logits.detach().numpy()
        probs = softmax(logits, axis=1)
        
        # Extract best label for each in batch
        for j in range(len(probs)):
            top_id = np.argmax(probs[j])
            processed_labels.append(id2label[top_id])
            processed_scores.append(round(float(probs[j][top_id]), 4))

    df.loc[relevant_mask, 'Sentiment_Label'] = processed_labels
    df.loc[relevant_mask, 'Sentiment_Score'] = processed_scores
    
    return df

def generate_final_report(df):
    print("\nLAW IMPACT REPORT (3-Class Analysis)")
    
    # Filter for report (Exclude N/A and Irrelevant)
    relevant_df = df[df['Sentiment_Label'] != "N/A"]
    
    if relevant_df.empty:
        return

    # Pivot Table: Clauses vs Sentiments
    summary = relevant_df.groupby(['Linked_Clause', 'Sentiment_Label']).size().unstack(fill_value=0)
    
    # Ensure all columns exist (Negative, Neutral, Positive)
    for col in ['negative', 'neutral', 'positive']:
        if col not in summary.columns: summary[col] = 0
    
    # Calculate Most Opposed
    if summary['negative'].sum() > 0:
        most_opposed = summary['negative'].idxmax()
        count = summary['negative'].max()
        print(f"MOST OPPOSED:  {most_opposed} ({count} negative comments)")
        
    # Calculate Most Supported
    if summary['positive'].sum() > 0:
        most_supported = summary['positive'].idxmax()
        count = summary['positive'].max()
        print(f"MOST SUPPORTED: {most_supported} ({count} positive comments)")

    print("\nDetailed Breakdown (Top 5):")
    print(summary.head())

# def store_sentiment_results():
#     input_file = os.path.join("data", "processed", "linked_dataset.csv")
#     final_df = analyze_sentiment(input_file)
#     if final_df is not None:
#         generate_final_report(final_df)

#     output_path = os.path.join("data", "processed", "sentiment_analysis_result.csv")
#     if final_df is not None:
#         final_df.to_csv(output_path, index=False)
#         print(f"Final sentiment analysis saved to: {output_path}")
#     else:
#         print("Error: Could not generate sentiment analysis results.")

# if __name__ == "__main__":
#     store_sentiment_results()