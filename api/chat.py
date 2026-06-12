import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

class ChatMessage(BaseModel):
    message: str

@app.post("/api/chat")
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
        ai_response = completion.choices[0].message.content
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
