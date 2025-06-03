from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# Enable CORS for all origins (you can restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming JSON
class Question(BaseModel):
    data: list[str]

# Set your OpenAI API key as an environment variable in Render dashboard!
openai.api_key = os.getenv("AIzaSyCSOtZ-6ixp4fIGpoB7IWKqjiAeZx28xZ0")

@app.get("/")
async def root():
    return {"message": "College chatbot API is running."}

@app.post("/api/predict")
async def predict(question: Question):
    user_question = question.data[0]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful college assistant."},
                {"role": "user", "content": user_question}
            ],
            max_tokens=150,
        )
        answer = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        answer = f"Sorry, an error occurred: {str(e)}"

    return {"data": [answer]}
