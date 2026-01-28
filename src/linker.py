import json
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

linker_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Linker Model Loaded!")

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
    law_clauses = load_law_context()
    if not law_clauses: 
        return None

    print(f"Vectorizing {len(law_clauses)} law clauses...")
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


if __name__ == "__main__":
    csv_path = os.path.join("data", "raw", "test_comments.csv")
    
    if os.path.exists(csv_path):
        print(f"Found CSV at: {csv_path}")
        input_df = pd.read_csv(csv_path)
        enriched_df = link_comments_to_law(input_df)
        
        if enriched_df is not None:
            print("\nSUCCESS! Preview of results:")
            print(enriched_df[['Comment', 'Linked_Clause', 'Match_Confidence']].head())
            
            enriched_df.to_csv("data/processed/test_linked_output.csv", index=False)
            print("Saved result to data/processed/test_linked_output.csv")
            
    else:
        print(f"Test CSV not found at {csv_path}")