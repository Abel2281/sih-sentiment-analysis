import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path) 

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)

def fetch_law_summary(law: str):
    print(f"Fetching summary for law: {law} using Groq...")

    prompt = f"""
    I need a structured database of the "Sections" of the law: "{law}".
    
    Rules:
    1. EXTRACT the actual Section Numbers (e.g., "Section 3", "Section 17").
    2. Provide the specific Title of that section.
    3. For the summary, provide a detailed 2-sentence explanation of what that specific section governs.
    4. Focus on the Top 20 most important sections (including Penalties, User Rights, Exemptions, and Data Fiduciary obligations).
    5. Output strictly as a JSON object with a single key "sections" containing a list of objects.

    Output Format:
    {{
      "sections": [
        {{
          "clause_id": "Section X",
          "title": "Official Title of Section",
          "summary": "Detailed explanation of this section's rules..."
        }}
      ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": "You are a legal data extraction assistant. You must output strictly in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=8000
        )
        
        response_text = response.choices[0].message.content
        if not response_text:
            print("No response text received from Groq.")
            return None
            
        raw_data = json.loads(response_text)
        law_data = raw_data.get("sections", [])
        
        print(f"Retrieved {len(law_data)} clauses.")
        return law_data

    except Exception as e:
        print(f"Error talking to Groq: {e}")
        return None

if __name__ == "__main__":
    summary = fetch_law_summary("Digital Personal Data Protection Act, 2023")
    if summary:
        print(json.dumps(summary, indent=2))