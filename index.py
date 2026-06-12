import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

# We try to load dotenv safely for your local machine, but bypass it in Vercel production
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="CoreLLM Model Hub",
    description="Multi-Model AI backend workspace running on Groq",
    version="1.0.0"
)

# Initialise Groq safely
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

class ChatMessage(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "CoreLLM Backend is active!"}

@app.post("/api/chat")
async def chat_with_llm(user_input: ChatMessage):
    global client
    # Re-verify key checks at runtime if instance built cleanly without token
    if not client:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY variable is missing on Vercel.")
        client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are CoreLLM, a brilliant AI workspace assistant."},
                {"role": "user", "content": user_input.message}
            ],
            temperature=0.7
        )
        ai_response = completion.choices[0].message.content
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Safely mount static folder assets if visible inside server environment containers
if os.path.exists("public"):
    # Mount the public directory cleanly
    app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
async def read_index():
    # A simple, bulletproof path rule for Vercel's main dashboard folder
    return FileResponse("public/index.html")

