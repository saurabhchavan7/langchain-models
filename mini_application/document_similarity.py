from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv('config/.env')

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300
)

documents = [
    "Elon Musk is a technology entrepreneur known for leading companies like Tesla and SpaceX.",
    "Tim Cook is the CEO of Apple and is known for his focus on operations and supply chain excellence.",
    "Jeff Bezos founded Amazon and played a major role in the growth of e-commerce and cloud computing.",
    "Satya Nadella is the CEO of Microsoft and is known for driving the company's cloud-first strategy.",
    "Sundar Pichai is the CEO of Google and oversees products like Search, Android, and Chrome."
]

query = "tell me about microsoft ceo"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
#print (cosine_similarity([query_embedding], doc_embeddings))

index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

print(query)
print(documents[index])
print("similarity score is:", score)
