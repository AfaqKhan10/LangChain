from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser




model = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,)

def str_output_parsers():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tell me a joke about the following subject."),
            ("human", "{input}")
        ]
    )
    parser  = StrOutputParser()
    chain = prompt | model | parser

    return chain.invoke({"input":"dog"})

print(str_output_parsers())


def list_output_parser():
    prompt  = ChatPromptTemplate.from_messages([
        ("system", "Generate the list of 10 synonyms for the following word. Return the result as a comma seperated list."),
        ("human", "{input}")
    ])
    parser = CommaSeparatedListOutputParser()
    chain = prompt | model | parser
    return chain.invoke({"input": "happy"})

print(list_output_parser())