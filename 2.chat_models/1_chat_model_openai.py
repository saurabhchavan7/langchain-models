from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv('config/.env')

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("what is the capital of india?")

print(result)