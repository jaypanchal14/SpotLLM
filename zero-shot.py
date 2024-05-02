from transformers import pipeline

candidate_labels = ['GPT-GENERATED', 'HUMAN-GENERATED']

bert_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence_to_classify = "A spelling checker in an NLP (Natural Language Processing) ML model typically works by leveraging statistical language models or neural networks to identify and correct misspelled words in text. Here's a general outline of how it can be implemented using machine learning"

# result = bert_classifier(sequence_to_classify, candidate_labels)
# print(result)
# labels = result['labels']
# scores = result['scores']

def predict(text : str):
    result = bert_classifier(text, candidate_labels)
    labels = result['labels']
    scores = result['scores']
    # print(result)
    return { "labels" : labels, "result" : scores}

# API part is below

from fastapi import FastAPI
from pydantic import BaseModel

class UserInput(BaseModel):
    text : str

app = FastAPI()

# pip install "uvicorn[standard]"  -> for server
# pip install fastapi
# command to run : uvicorn api:app 
# option to pass while running (if you want automatic reloading):  --reload

#swagger api : 
# http://127.0.0.1:8000/docs
# or
# http://127.0.0.1:8000/redoc

# you can try endpoints from the swagger api doc

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def mainMLFunction(user_input: UserInput):
    # print(user_input.text)
    return predict(user_input.text)

