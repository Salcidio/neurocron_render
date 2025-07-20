# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, set_seed
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for your frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a lightweight conversational model
generator = pipeline('text-generation', model='distilgpt2')
set_seed(42)

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

    # Generate a response
    output = generator(full_prompt, max_length=100, num_return_sequences=1)
    response_text = output[0]['generated_text'].split("Bot:")[-1].strip()
    
    return {"response": response_text}