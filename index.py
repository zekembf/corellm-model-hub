import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="CoreLLM Model Hub",
    description="Multi-Model AI backend workspace running on Groq",
    version="1.0.0"
)

print("CoreLLM starting...")
print("GROQ_API_KEY found:", bool(os.getenv("GROQ_API_KEY")))

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


class ChatMessage(BaseModel):
    message: str


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "CoreLLM Backend is active!"
    }


@app.post("/api/chat")
def chat_with_llm(chat_data: ChatMessage):
    global client

    try:
        if not client:
            current_key = os.getenv("GROQ_API_KEY")

            if not current_key:
                raise HTTPException(
                    status_code=500,
                    detail="GROQ_API_KEY is missing on Vercel."
                )

            client = Groq(api_key=current_key)

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are CoreLLM, a brilliant AI workspace assistant."
                },
                {
                    "role": "user",
                    "content": chat_data.message
                }
            ],
            temperature=0.7
        )

        # Correct Groq response access
        ai_response = completion.choices[0].message.content

        return {
            "response": ai_response
        }

    except HTTPException:
        raise

    except Exception as e:
        print("CHAT ERROR:")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/")
def read_root():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoreLLM - AI Workspace</title>

<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
    margin: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
}

.chat-container {
    width: 100%;
    max-width: 700px;
    height: 85vh;
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}

.chat-header {
    padding: 1rem;
    background: #1f2937;
    font-weight: bold;
    border-bottom: 1px solid #374151;
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.message {
    max-width: 80%;
    padding: 12px;
    border-radius: 10px;
    line-height: 1.5;
    word-wrap: break-word;
}

.user-message {
    background: #2563eb;
    color: white;
    align-self: flex-end;
}

.ai-message {
    background: #374151;
    color: white;
    align-self: flex-start;
}

.chat-input-area {
    display: flex;
    gap: 10px;
    padding: 1rem;
    background: #1f2937;
    border-top: 1px solid #374151;
}

input {
    flex: 1;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #374151;
    background: #0b0f19;
    color: white;
    outline: none;
}

input:focus {
    border-color: #2563eb;
}

button {
    padding: 12px 20px;
    border: none;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    cursor: pointer;
    font-weight: bold;
}

button:hover {
    background: #1d4ed8;
}
</style>
</head>

<body>

<div class="chat-container">

    <div class="chat-header">
        <div class="status-dot"></div>
        CoreLLM Workspace (Llama 3.1 via Groq)
    </div>

    <div id="chatBox" class="chat-messages">
        <div class="message ai-message">
            Hello! I'm CoreLLM. Ask me anything.
        </div>
    </div>

    <div class="chat-input-area">
        <input
            id="userInput"
            type="text"
            placeholder="Ask anything..."
            onkeypress="handleKeyPress(event)"
        />
        <button onclick="sendMessage()">Send</button>
    </div>

</div>

<script>

async function sendMessage() {

    const inputEl = document.getElementById("userInput");
    const chatBox = document.getElementById("chatBox");

    const text = inputEl.value.trim();

    if (!text) return;

    chatBox.innerHTML += `
        <div class="message user-message">${text}</div>
    `;

    inputEl.value = "";

    const aiMessageEl = document.createElement("div");
    aiMessageEl.className = "message ai-message";
    aiMessageEl.innerText = "Thinking...";

    chatBox.appendChild(aiMessageEl);

    chatBox.scrollTop = chatBox.scrollHeight;

    try {

        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: text
            })
        });

        const data = await response.json();

        if (!response.ok) {
            aiMessageEl.innerText =
                data.detail || "Server error occurred.";
        } else {
            aiMessageEl.innerText =
                data.response || "No response received.";
        }

    } catch (error) {

        console.error(error);

        aiMessageEl.innerText =
            "Failed to connect to backend.";

    }

    chatBox.scrollTop = chatBox.scrollHeight;
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}

</script>

</body>
</html>
"""

    return HTMLResponse(
        content=html_content,
        status_code=200
    )