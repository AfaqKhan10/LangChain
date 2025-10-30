from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of synonyms for the following word.Retrun the result as a comma seperated list."),
        ("human", "{input}")
    ]
)

chain = prompt | llm

response = chain.invoke({"input":"happy"})
print(response)