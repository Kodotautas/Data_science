import uvicorn
from fastapi import FastAPI, Request, Form
from models.dialoGPT import ChatBot

app = FastAPI()

@app.post("/")
async def conversation(request:Request, message: str = Form(...)):
  
  # gets a response of the AI bot
  bot_reply = chatbot.get_reply(message)
  
  # returns the final HTML
  return {'message':bot_reply}

# initialises the chatbot model and starts the uvicorn app
if __name__ == "__main__":
  chatbot = ChatBot()
  uvicorn.run(app, host="0.0.0.0", port=8000)