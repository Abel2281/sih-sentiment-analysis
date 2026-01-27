import os
import json
from src.law_fetcher import fetch_law_summary

def main():
    law_name = input("Enter law name: ")
    law_data = fetch_law_summary(law_name)

    if law_data:
        save_path = os.path.join("data", "laws_json", f"{law_name}_context.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(law_data, f, indent=4)
            
        print(f"Context saved to: {save_path}")

if __name__ == "__main__":
    main()