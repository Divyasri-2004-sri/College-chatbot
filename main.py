from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class Question(BaseModel):
    data: list[str]

# Load models
gen_model = "google/flan-t5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(gen_model)
model = AutoModelForSeq2SeqLM.from_pretrained(gen_model).to(device)
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

knowledge_base = [
    "Our college offers B.Tech, MBA, and BBA courses.",
    "Hostel fees are 50,000 INR per year.",
    "Admissions start every June.",
    "Scholarships are available for the top 10% of students.",
    "The campus has modern labs and sports facilities.",
]

kb_embeddings = embed_model.encode(knowledge_base, convert_to_tensor=True)

simple_responses = {
    "hi": "Hello! How can I help you with your college queries today?",
    "hello": "Hi there! What do you want to know about the college?",
    "thanks": "You're welcome!",
    "bye": "Goodbye! Have a great day.",
}

def get_relevant_info(question, top_k=2):
    q_embed = embed_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_embed, kb_embeddings, top_k=top_k)[0]
    return "\n".join(knowledge_base[hit['corpus_id']] for hit in hits)

def ask_bot(question: str) -> str:
    q_lower = question.strip().lower()
    if q_lower in simple_responses:
        return simple_responses[q_lower]

    info = get_relevant_info(question)
    prompt = f"""
You are a helpful college assistant. Use ONLY the information below to answer the question.
If the information does not contain the answer, say "Sorry, I don't know."

Information:
{info}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer

@app.get("/")
async def root():
    return {"message": "College chatbot API is running."}

@app.post("/api/predict")
async def predict(question: Question):
    user_question = question.data[0]
    answer = ask_bot(user_question)
    return {"data": [answer]}
