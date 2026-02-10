import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import numpy as np
from scipy.special import softmax


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32

def analyze_sentiment(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    if 'Linked_Clause' not in df.columns:
        print("Error: CSV is missing 'Linked_Clause'. Run linker.py first!")
        return None
    
    relevant_mask = df['Linked_Clause'] != "Irrelevant"
    relevant_comments = df.loc[relevant_mask, 'Comment'].tolist()
    
    print(f"Total Comments: {len(df)}")
    print(f"Relevant Comments to Process: {len(relevant_comments)}")
    print(f"   (Skipping {len(df) - len(relevant_comments)} irrelevant rows...)")
    
    if not relevant_comments:
        print("No relevant comments found. Exiting.")
        return df

    print(f"Loading Sentiment Model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    device = torch.device("cpu")
    model.to(device)
    model.eval() # Disable training mode for speed
    
    df['Sentiment_Label'] = "N/A"
    df['Sentiment_Score'] = 0.0

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
            max_length=512, 
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

if __name__ == "__main__":
    input_file = os.path.join("data", "processed", "test_linked_output.csv")
    
    final_df = analyze_sentiment(input_file)
    
    if final_df is not None:
        generate_final_report(final_df)
        
        output_path = os.path.join("data", "processed", "final_analysis_result.csv")
        final_df.to_csv(output_path, index=False)
        print(f"\nSaved final analysis to {output_path}")