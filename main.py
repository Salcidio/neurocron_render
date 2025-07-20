import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192" #"mixtral-8x7b-32768"  Also try "gemma-7b-it"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: list = []

@app.post("/chat")
def chat(request: ChatRequest):
    # Build the chat history
    messages = [
        {"role": "system", "content": f"You are a helpful assistant with the persona of a {request.persona}."}
    ]
    
    for turn in request.history:
        messages.append({"role": "user", "content": turn})
    
    messages.append({"role": "user", "content": request.message})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return {"response": reply}
    except Exception as e:
        return {"response": f"Error contacting Groq API: {e}"}
