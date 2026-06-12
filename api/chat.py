import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# 🔄 Vercel strictly requires this variable name to be 'handler'
handler = FastAPI()

class ChatMessage(BaseModel):
    message: str

# 🔄 Update the decorator to use @handler instead of @app
@handler.post("/api/chat")
async def chat_with_llm(chat_data: ChatMessage):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is missing on Vercel.")

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are CoreLLM, a brilliant AI workspace assistant."},
                {"role": "user", "content": chat_data.message}
            ],
            temperature=0.7
        )
        ai_response = completion.choices.message.content
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
