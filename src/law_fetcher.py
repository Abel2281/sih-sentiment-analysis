from google import genai
from google.genai import types
import os
import json
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path) 
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("GENAI_API_KEY not found! Check your .env file.")

client = genai.Client(api_key=api_key)

def fetch_law_summary(law: str):
    print(f"Fetching summary for law: {law}")

    prompt = f"""
    I need a structured database of the "Sections" of the law: "{law}".
    
    Rules:
    1. EXTRACT the actual Section Numbers (e.g., "Section 3", "Section 17").
    2. Provide the specific Title of that section.
    3. For the summary, provide a detailed 2-sentence explanation of what that specific section governs.
    4. Focus on the Top 20 most important sections (including Penalties, User Rights, Exemptions, and Data Fiduciary obligations).
    5. Output strictly as a JSON list of objects.

    Output Format:
    [
      {{
        "clause_id": "Section X",
        "title": "Official Title of Section",
        "summary": "Detailed explanation of this section's rules..."
      }}
    ]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
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