import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from google import genai
from huggingface_hub import hf_hub_download

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# -----------------------------
# 2. Load trained model bundle
# -----------------------------
model_path = hf_hub_download(
    repo_id="Jaoooooo9/firstclinic-triage-dl-model",
    filename="health_risk_deep_model.joblib",
    token=os.getenv("HF_TOKEN")
)
model_bundle = joblib.load(model_path)
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
class HistoryMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    vitals: Optional[dict] = None
    history: Optional[List[HistoryMessage]] = []

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
    history = req.history or []

    # Build conversation history context
    conversation_context = ""
    if history:
        conversation_context = "Previous conversation:\n"
        for msg in history[-6:]:  # last 6 messages for context
            role = "Patient" if msg.role == "user" else "Assistant"
            conversation_context += f"{role}: {msg.content}\n"
        conversation_context += "\n"

    # Build current prompt
    if vitals:
        try:
            risk = predict_risk(vitals)
            prompt = (
                f"{conversation_context}"
                f"The patient's current message: '{user_message}'. "
                f"Their vital signs are {vitals}. "
                f"The predicted health risk level is '{risk}'. "
                "You are a medical AI assistant. Provide empathetic, clear, and helpful "
                "medical guidance based on the risk level and vital signs. "
                "If risk is High, strongly advise immediate medical attention. "
                "If Medium, advise scheduling a doctor visit soon. "
                "If Low, provide reassurance and healthy lifestyle tips. "
                "Always remind the patient this is AI guidance, not a substitute for professional medical advice."
            )
        except Exception as e:
            prompt = (
                f"{conversation_context}"
                f"The patient says: '{user_message}'. "
                f"There was an error processing vital signs: {str(e)}. "
                "Provide general health guidance with empathy."
            )
    else:
        prompt = (
            f"{conversation_context}"
            f"The patient says: '{user_message}'. "
            "You are a compassionate medical AI assistant. "
            "Provide helpful general health information, lifestyle suggestions, "
            "or answer health-related questions with empathy and clarity. "
            "Always remind the patient to consult a real doctor for medical decisions."
        )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return {"reply": response.text}

    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return {
                "reply": "I am currently experiencing high demand. Please try again in a few minutes."
            }
        print(f"Gemini API Error: {e}")
        return {
            "reply": "I am having trouble connecting to the AI service. Please check your connection and try again."
        }

# -----------------------------
# 7. Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "FirstClinic API is running successfully!"}

