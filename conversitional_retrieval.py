from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever


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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore


def create_chain(vectorstore):
    model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.3
    )
    
    document_prompt = ChatPromptTemplate.from_template("""
    You are a helpful and friendly AI assistant. Use only the retrieved context to answer the user’s questions.
    If the context doesn’t contain relevant information, politely say you’re unsure. Respond in a natural, human-like,
    and conversational way while keeping your answers short, clear, and to the point. Remember the previous chat history
    to maintain context and a personalized tone.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=document_prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Based on the above conversation, generate a concise search query to find the most relevant information.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=document_chain
    )

    return retrieval_chain



def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]



if __name__ == '__main__':
    print("Loading website content, please wait...")
    docs = get_documents_from_web("https://datics.ai/")
    vectorstore = create_db(docs)
    chain = create_chain(vectorstore)

    chat_history = []
    print("\n Chatbot ready! Start chatting (type 'exit' 'bye''quit' to stop)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye", "quit"]:
            print("Assistant: Goodbye! Have a great day!")
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)
