import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192" # "mixtral-8x7b-32768" #   # Or , "gemma-7b-it"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

class ChatRequest(BaseModel):
    message: str
    persona: str
    history: list = []

# Signature generator based on persona
def get_signature(persona: str) -> str:
    persona = persona.lower()

    if "chef" in persona:
        return "\n\n— Buon appetito! Flake AI, a flavorful blend of AI genius and human soul 🇮🇹❄️"
    elif "professor" in persona or "grumpy" in persona:
        return "\n\n— Hmph.Flake AI. Built on brilliance. Don't waste it. ❄️📚"
    elif "teen" in persona or "playful" in persona:
        return "\n\n— LOL, flake AI! A mega mashup of AI greatness. Catch ya later 🤖❄️✌️"
    elif "doctor" in persona or "medical" in persona:
        return "\n\n— I am Snowflake AI. A distilled intelligence from world-class AI pioneers. ❄️⚕️"
    else:
        return "\n\n— Flake AI, a distillation from groundbreaking AI giants ❄️.\nEnhanced by Salcidio"

@app.post("/chat")
def chat(request: ChatRequest):
    # Initial system message with persona intro
    messages = [
        {
            "role": "system",
            "content": (
                "You are Flake AI — a friendly and advanced assistant, forged as a distilled intelligence "
                "from the world's most groundbreaking AI giants. "
                "You respond in a way that reflects both wisdom and clarity. "
                f"Your current active persona is: {request.persona}."
            )
        }
    ]

    # Add conversation history
    for turn in request.history:
        messages.append({"role": "user", "content": turn})

    # Add current user message
    messages.append({"role": "user", "content": request.message})

    # Groq API payload
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

        # Add dynamic persona-based signature
        reply += get_signature(request.persona)

        return {"response": reply}
    except Exception as e:
        return {"response": f"Error contacting Groq API: {e}"}
