import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

# Load groq API Key from environment variable
groq_api_key = "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"

st.set_page_config(page_title="Wine Chatbot", page_icon="🍷")

st.title("🍷Jessup Cellars Wine Chatbot")
st.image("idealwine.jpeg", width=250 )
st.header("Ask me anything about wines! 🍾")

# Sidebar with Company Logo and Details
st.sidebar.image("idealwines.jpg", use_column_width=True)  # Replace with your actual logo path
st.sidebar.header("Jessup Cellars")
st.sidebar.write("Welcome to Jessup Cellars Wines. We provide authentic Wines and customers are our god.")
st.sidebar.write("Address: 123 AI Drive, Tech City, Innovation State, 12345")
st.sidebar.write("Contact: contact@companyname.com")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt_text = """
Based on the context provided, answer the following questions accurately and thoroughly. If the question pertains to the PDF content, provide a detailed explanation. If the question is unrelated to the PDF, respond with "Please contact the business directly for more information."

<context>
{context}
<context>
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_text)

# Specify the path of the PDF file
file_path = "Corpus.pdf"  # Replace with your actual file path

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.loader = PyPDFLoader(file_path)  # Data Injection
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Document splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector Huggingface Embedding

# Initialize embeddings when the Streamlit app starts
vector_embeddings()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt1 = st.text_input("Enter the Question from your Mind:")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)  # Pass `prompt` instance, not `prompt_text`
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input': prompt1})
    
    # Save question and response to chat history
    st.session_state.chat_history.append((prompt1, response['answer']))

# Display chat history
st.header("Chat History")
for i, (question, answer) in enumerate(st.session_state.chat_history):
    st.write(f"**Q{i+1}:** {question}")
    st.write(f"**A{i+1}:** {answer}")

st.header('', divider='rainbow')
st.markdown('''
    Developed by KARTAVYA MASTER :8ball:
''')

link = 'PORTFOLIO : [CLICK ME](https://mydawjbhdas.my.canva.site/aiwithkartavya)'
st.markdown(link, unsafe_allow_html=True)
