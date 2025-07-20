# main.py
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests  # <-- Add this import

app = FastAPI()

# Allow CORS for your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace the local model pipeline with an API call function
API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status() # Raise an exception for bad status codes
    return response.json()

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: list = [] # To hold conversation history for context

@app.post("/chat")
def chat(request: ChatRequest):
    # Create a prompt combining persona, history, and the new message
    persona_prompt = f"You are a {request.persona}. "
    history_str = "\n".join(request.history)
    full_prompt = f"{persona_prompt}\n{history_str}\nUser: {request.message}\nBot:"

    # Generate a response using the API
    output = query({
        "inputs": full_prompt,
        "options": {"use_cache": False},
        "parameters": {"max_new_tokens": 100}
    })
    
    # Extract the generated text from the API response
    response_text = output[0]['generated_text'].split("Bot:")[-1].strip()
    
    return {"response": response_text}