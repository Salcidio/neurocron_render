import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mistral 7B Instruct model from Hugging Face
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
API_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query_huggingface(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return [{"generated_text": f"Error contacting model: {e}"}]

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: list = []

@app.post("/chat")
def chat(request: ChatRequest):
    # Format prompt
    persona_prompt = f"You are a helpful assistant with the persona: {request.persona}.\n"
    history_str = "\n".join(request.history)
    full_prompt = f"{persona_prompt}{history_str}\nUser: {request.message}\nAssistant:"

    # Send to Hugging Face model
    output = query_huggingface({
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95
        }
    })

    # Parse response
    generated_text = output[0].get("generated_text", "")
    response_text = generated_text.split("Assistant:")[-1].strip()

    return {"response": response_text}
