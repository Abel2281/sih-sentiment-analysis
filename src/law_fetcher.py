from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

load_dotenv("../.env") 
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("GENAI_API_KEY not found! Check your .env file.")

client = genai.Client(api_key=api_key)

def fetch_law_summary(law: str):
    print(f"Fetching summary for law: {law}")

    prompt = f"""
    I need a structured summary of the law: "{law}".
    
    Please provide a response strictly in valid JSON format. 
    Do not add markdown formatting like ```json ... ```. Just the raw JSON string.
    
    The JSON should be a list of objects, where each object represents a key clause/section.
    Structure:
    [
        {{
            "clause_id": "Section 1",
            "title": "Short Title",
            "simple_summary": "One sentence explanation for a layman."
        }},
        {{
            "clause_id": "Section 2",
            ...
        }}
    ]
    
    Focus on the 5-10 most important or controversial sections of this law.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4000,
                response_mime_type="application/json"
            )
        )
        if not response.text:
            print("No response text received from Gemini.")
            return None
        law_data = json.loads(response.text)
        
        print(f"Success! Retrieved {len(law_data)} clauses.")
        return law_data

    except Exception as e:
        print(f"Error talking to Gemini: {e}")
        return None


if __name__ == "__main__":
    print("Unit Test for fetch_law_summary")
    summary = fetch_law_summary("Forest Conservation Act")
    if summary:
        print(json.dumps(summary, indent=2))