import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse  # 👈 Make sure this is imported at the top
from pydantic import BaseModel

app = FastAPI(
    title="CoreLLM Model Hub",
    description="Multi-Model AI backend workspace running on Groq",
    version="1.0.0"
)

class ChatMessage(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "CoreLLM Backend is active!"}

@app.post("/api/chat")
async def chat_with_llm(user_input: ChatMessage):
    # We import groq dynamically INSIDE the function to bypass Vercel's build-time module validation bug
    try:
        from groq import Groq
    except ImportError:
        raise HTTPException(status_code=500, detail="The 'groq' package is failing to load on the server container.")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is missing on Vercel.")

    try:
        client = Groq(api_key=api_key)
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

# Static assets serving
if os.path.exists("public"):
    app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/")
async def read_index():
    # We bake your beautiful frontend UI directly into the Python file so Vercel never loses it!
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
                max-width: 600px;
                height: 80vh;
                background: #111827;
                border: 1px solid #1f2937;
                border-radius: 12px;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            }
            .chat-header {
                padding: 1rem;
                background: #1f2937;
                font-weight: bold;
                border-bottom: 1px solid #374151;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background: #10b981;
                border-radius: 50%;
            }
            .chat-messages {
                flex: 1;
                padding: 1rem;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .message {
                max-width: 80%;
                padding: 10px 14px;
                border-radius: 8px;
                line-height: 1.4;
                word-wrap: break-word;
            }
            .user-message {
                background: #2563eb;
                color: white;
                align-self: flex-end;
            }
            .ai-message {
                background: #374151;
                color: #f3f4f6;
                align-self: flex-start;
            }
            .chat-input-area {
                padding: 1rem;
                background: #1f2937;
                display: flex;
                gap: 8px;
                border-top: 1px solid #374151;
            }
            input {
                flex: 1;
                padding: 12px;
                background: #0b0f19;
                border: 1px solid #374151;
                border-radius: 6px;
                color: white;
                outline: none;
            }
            input:focus {
                border-color: #2563eb;
            }
            button {
                padding: 12px 20px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                cursor: pointer;
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
            CoreLLM Workspace (Llama 3 Active)
        </div>
        <div class="chat-messages" id="chatBox">
            <div class="message ai-message">Hello! I am CoreLLM. Type a prompt below to chat with me.</div>
        </div>
        <div class="chat-input-area">
            <input type="text" id="userInput" placeholder="Ask anything..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const inputEl = document.getElementById('userInput');
            const chatBox = document.getElementById('chatBox');
            const text = inputEl.value.trim();
            
            if (!text) return;

            chatBox.innerHTML += `<div class="message user-message">${text}</div>`;
            inputEl.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;

            const aiMessageEl = document.createElement('div');
            aiMessageEl.className = 'message ai-message';
            aiMessageEl.innerText = 'Thinking...';
            chatBox.appendChild(aiMessageEl);
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                // Hit the local or production cloud endpoint cleanly relative to origin
                            //  The perfect format:
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: text }) // 👈 Change "message" to "user_input"
            });

                
                const data = await response.json();
                aiMessageEl.innerText = data.response || "Error fetching AI response.";
            } catch (error) {
                aiMessageEl.innerText = "Failed to connect to the backend server.";
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }
    </script>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)