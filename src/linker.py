import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

@st.cache_resource
def get_linker_model():
    print("Loading Linker Model into RAM...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_law_context():
    path = os.path.join("data", "processed", "law_context.json")
    if not os.path.exists(path):
        print(f"Error: Law file not found at {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def link_comments_to_law(input_df):
    """
    Input: A Pandas DataFrame containing a 'Comment' column.
    Output: The same DataFrame with new columns added.
    """
    linker_model = get_linker_model()

    law_clauses = load_law_context()
    if not law_clauses: 
        return None

    clause_summaries = [c['summary'] for c in law_clauses]
    clause_ids = [c['clause_id'] for c in law_clauses]
    clause_vectors = linker_model.encode(clause_summaries, normalize_embeddings=True)
    
    print(f"Vectorizing {len(input_df)} user comments...") 
    if 'Comment' not in input_df.columns:
        print("Error: Your CSV must have a column named 'Comment'")
        return None

    comments_list = input_df['Comment'].tolist()
    comment_vectors = linker_model.encode(comments_list, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    
    print("Running Matrix Multiplication...")
    similarity_matrix = np.matmul(comment_vectors, clause_vectors.T)
    
    linked_clauses = []
    confidence_scores = []
    
    for i in range(len(comments_list)):
        best_match_idx = similarity_matrix[i].argmax()
        best_score = similarity_matrix[i][best_match_idx]
        
        if best_score < 0.25:
            linked_clauses.append("Irrelevant")
        else:
            linked_clauses.append(clause_ids[best_match_idx])
            
        confidence_scores.append(round(float(best_score), 2))
        
    input_df['Linked_Clause'] = linked_clauses
    input_df['Match_Confidence'] = confidence_scores

    return input_df

# def store_linked_results():
#     csv_path = os.path.join("data", "raw", "datasetv1.csv")
#     output_path = os.path.join("data", "processed", "linked_dataset.csv")
#     if os.path.exists(csv_path):
#         print(f"Found CSV at: {csv_path}")
#         input_df = pd.read_csv(csv_path)
#         enriched_df = link_comments_to_law(input_df)
#         if enriched_df is not None:
#             enriched_df.to_csv(output_path, index=False)
#             print(f"Linked dataset saved to: {output_path}")
#         else:
#             print("Error: Could not link comments to law.")
#     else:
#         print(f"Error: CSV file not found at {csv_path}")

# if __name__ == "__main__":  
#     store_linked_results()