from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

load_dotenv()


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(docs)
    return split_docs



def create_db(docs):
    # Use open-source embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore


def create_chain(vectorstore):
    model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",  # or mixtral-8x7b-32768
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based only on the following context.
    Context: {context}
    Question: {input}
    """)


    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

 
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )
    return retrieval_chain


docs = get_documents_from_web("https://datics.ai/")
vectorstore = create_db(docs)
chain = create_chain(vectorstore)

response = chain.invoke({"input": "Tell something about datics"})
print(response["context"])


