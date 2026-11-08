from httpx2 import query
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.title("RAG gpt chatbot Implementation")

# using OpenAI embeddings for the documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# loading the vector store from the local file
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# creating a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# creating a chat model for GPT
gpt_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

# prompt template structure to inject context into the gpt_llm
gpt_rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. If the question can't be answered based on the context, say "I don't know."
Add emojis to  your answer to make it more engaging. Keep your answer concise and clear.
Context: {context}
Question: {question}
""")

query = st.text_input("Enter your question:")
if query:
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    chain = gpt_rag_prompt | gpt_llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": query})
    st.write(response)