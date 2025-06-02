import pandas as pd
from rag_utils import rag
import requests

df = pd.read_csv("../data/ground-truth-retrieval.csv")
question = df.sample(n=1).iloc[0]['question']

print("question: ", question)

url = "http://localhost:5000/question"
print(url)

data = {"question": question}

response = requests.post(url, json=data)
#print(response.content)

print(response.json())

#print(requests.get("https://api.openai.com/v1/models").json())

#answer_data = rag(question)
#print(answer_data["answer"])