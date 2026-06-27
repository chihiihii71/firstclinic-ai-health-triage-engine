# app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Use the correct import path for the SDK
import google.generativeai as genai

# NEW IMPORT: Necessary to catch the specific "Quota Exceeded" error
from google.api_core.exceptions import ResourceExhausted

# -----------------------------
# 1. Load environment variables and configure Gemini
# -----------------------------
load_dotenv()  # Load .env variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# 2. Load trained model bundle
# -----------------------------
MODEL_PATH = "health_risk_deep_model.joblib"
model_bundle = joblib.load(MODEL_PATH)

model_pipeline = model_bundle["pipeline"]
label_encoder = model_bundle["label_encoder"]
feature_columns = model_bundle["feature_columns"]

# -----------------------------
# 3. Initialize FastAPI app
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 4. Request schema
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    vitals: dict | None = None

# -----------------------------
# 5. Predict risk function
# -----------------------------
def predict_risk(vitals: dict) -> str:
    df = pd.DataFrame([vitals], columns=feature_columns)
    pred_encoded = model_pipeline.predict(df)
    pred_label = label_encoder.inverse_transform(pred_encoded.reshape(-1))
    return pred_label[0]

# -----------------------------
# 6. Health chatbot endpoint
# -----------------------------
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message
    vitals = req.vitals

    prompt = f"The patient says: '{user_message}'. "

    if vitals:
        try:
            risk = predict_risk(vitals)
            prompt += (
                f"Their vital signs are {vitals}. "
                f"The predicted health risk level is '{risk}'. "
                "Provide empathetic, clear, and helpful medical guidance based on this information."
            )
        except Exception as e:
            prompt += (
                f"Error predicting risk from vitals. "
                f"Provide general health guidance anyway. [Error: {str(e)}]"
            )
    else:
        prompt += (
            "No vital signs provided. "
            "Provide general health or lifestyle suggestions with empathy."
        )

    model_llm = genai.GenerativeModel("models/gemini-2.5-flash")

    # --- ERROR HANDLING FIX STARTS HERE ---
    try:
        # Try to generate content
        response = model_llm.generate_content(prompt)
        return {"reply": response.text}

    except ResourceExhausted:
        # This block runs ONLY if the quota is exceeded
        print("Error: Gemini API Quota Exceeded")
        return {
            "reply": "I apologize, but I am currently overloaded with requests (Quota Exceeded). Please try again in a few minutes."
        }

    except Exception as e:
        # This catches any other unexpected errors
        print(f"Error calling Gemini API: {e}")
        return {
            "reply": "I'm having trouble connecting to the AI service right now. Please try again later."
        }
    # --- ERROR HANDLING FIX ENDS HERE ---

# -----------------------------
# 7. Root endpoint for testing
# -----------------------------
@app.get("/")
def root():
    return {"message": "Chatbot backend is running successfully!"}