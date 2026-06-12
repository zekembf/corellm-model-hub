import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load local .env variables if present (Vercel will bypass this and use its Dashboard variables)
load_dotenv()

app = FastAPI(
    title="CoreLLM Model Hub",
    description="Multi-Model AI backend workspace running on Groq",
    version="1.0.0"
)

# Initialise Groq safely (prevents crashes during Vercel build time if env keys load late)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback placeholder so the file imports without crashing during build verification
        return None
    return Groq(api_key=api_key)

class ChatMessage(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "CoreLLM Backend is active!"}

@app.post("/api/chat")
async def chat_with_llm(user_input: ChatMessage):
    client = get_groq_client()
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is missing on the server.")
        
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

# Check if public folder exists before mounting to prevent deployment import issues
if os.path.exists("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
async def read_index():
    # Production-safe absolute path structure for Vercel's execution containers
    base_dir = os.path.dirname(os.path.dirname(__file__)) if __name__ == "api.index" else os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "public", "index.html")
    
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Frontend files not found, but backend is fully operational!"}
