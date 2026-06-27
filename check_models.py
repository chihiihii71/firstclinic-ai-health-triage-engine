# check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Configuration (Must match app.py) ---
load_dotenv() 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Function to List Models ---
def list_available_gemini_models():
    """Lists all models accessible by the current API key that support generateContent."""
    
    print("\n=======================================================")
    print("      AVAILABLE MODELS FOR generateContent()           ")
    print("=======================================================")
    
    found_flash = False
    
    # Iterate through all models provided by the API
    try:
        for model in genai.list_models():
            # Filter for models that can generate content (the ones you need)
            if 'generateContent' in model.supported_generation_methods:
                print(f"| Model Name: **{model.name}**")
                
                # Check for Gemini 1.5 Flash variants
                if "gemini-1.5-flash" in model.name:
                    found_flash = True

        if not found_flash:
            print("\n| WARNING: Gemini 1.5 Flash variants were not found.")
            print("| Your library might still be outdated, or access might be restricted.")
    
    except Exception as e:
        print(f"\n| FATAL ERROR during API Call: {e}")
        print("| Check your GOOGLE_API_KEY in the .env file.")
        
    print("=======================================================\n")


# Execute the function
list_available_gemini_models()