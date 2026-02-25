import os
import pandas as pd
from src.law_fetcher import store_law_summary
from src.linker import link_comments_to_law
from src.sentiment_engine import analyze_sentiment
from src.insight_engine import analyze_insights

def run_full_pipeline(law_name, raw_csv_path):
    if not os.path.exists(raw_csv_path):
        return f"Error: Could not find input file at {raw_csv_path}"

    df = pd.read_csv(raw_csv_path)
    # Step 1: Fetch Law Context
    print("\n[1/4] Fetching law summary...")
    store_law_summary(law_name)

    # Step 2: Link Comments to Law
    print("\n[2/4] Linking comments to law...")
    linked_df = link_comments_to_law(df)
    if linked_df is None:
        return "Pipeline Failed: Linker returned None."

    # Step 3: Sentiment Analysis
    print("\n[3/4] Performing sentiment analysis...")
    final_df = analyze_sentiment(linked_df)
    if final_df is None:
        return "Pipeline Failed: Sentiment Engine returned None."

    final_output_path = "data/processed/final_analysis_result.csv"
    final_df.to_csv(final_output_path, index=False)
    print(f"Final data saved to {final_output_path}")

    # Step 4: Insight Generation
    print("\n[4/4] Generating insights...")
    analyze_insights(final_output_path, law_name)
    
    return "Pipeline Completed Successfully!"

if __name__ == "__main__":
    input_file = os.path.join("data", "raw", "datasetv1.csv")
    run_full_pipeline("Personal Data Protection Bill, 2019", input_file)