import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles  # 👈 Add this line
from fastapi.responses import FileResponse  # 👈 Add this line
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="CoreLLM Model Hub",
    description="Multi-Model AI backend workspace running on Groq",
    version="1.0.0"
)

client = Groq()

class ChatMessage(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "CoreLLM Backend is active!"}

@app.post("/api/chat")
async def chat_with_llm(user_input: ChatMessage):
    try:
                # 🔄 Update the model string right here!
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 👈 Change to this active model name
            messages=[
                {"role": "system", "content": "You are CoreLLM, a brilliant AI workspace assistant."},
                {"role": "user", "content": user_input.message}
            ],
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content
        return {"response": ai_response}
    except Exception as e:
        print("🔴 BACKEND CRASH ERROR:", str(e))  # 👈 Add this line right here!
        raise HTTPException(status_code=500, detail=str(e))


# 👈 Add these final lines at the absolute bottom of the file
# Mount the public directory to serve your static assets (CSS, Icons, etc.)
# 🔄 Update the final lines at the bottom of your file to use index.html cleanly:
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
async def read_index():
    # Vercel needs a relative path check to locate your template file safely in serverless environments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "public", "index.html")
    return FileResponse(html_path)
