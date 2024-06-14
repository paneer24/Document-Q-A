import os
import pymongo
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as gen_ai
import atexit
import hashlib
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Get environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GOOGLE_API_KEY is None:
    st.error("Google API key not found. Please make sure it's set in your .env file.")
    st.stop()

if MONGODB_URI is None:
    st.error("MongoDB URI not found. Please make sure it's set in your .env file.")
    st.stop()

if GROQ_API_KEY is None:
    st.error("GROQ API key not found. Please make sure it's set in your .env file.")
    st.stop()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with my AI!",
    page_icon=":alien:",
    layout="wide",
)

# Set up MongoDB connection
try:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["chatbot_db"]
    collection = db["chat_history"]
    user_collection = db["users"]  # Collection for user credentials

    # Ensure text index on 'text' field for full-text search
    collection.create_index([("text", pymongo.TEXT)])
except pymongo.errors.ConfigurationError as e:
    st.error(f"Configuration error: {e}")
    st.stop()
except pymongo.errors.ServerSelectionTimeoutError as e:
    st.error(f"Server selection timeout error: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Ensure MongoDB client is closed when Streamlit app stops
def close_mongo_client():
    client.close()

atexit.register(close_mongo_client)

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to register a new user
def register_user(username, password):
    hashed_password = hash_password(password)
    user_data = {"username": username, "password": hashed_password}
    user_collection.insert_one(user_data)

# Function to authenticate user
def authenticate_user(username, password):
    hashed_password = hash_password(password)
    user = user_collection.find_one({"username": username, "password": hashed_password})
    return user is not None

# Function to save chat history
def save_chat_history(username, question, answer):
    chat_data = {"username": username, "question": question, "answer": answer, "timestamp": datetime.utcnow()}
    collection.insert_one(chat_data)

# Function to load chat history
def load_chat_history(username):
    return collection.find({"username": username}).sort("timestamp", pymongo.ASCENDING)

# User authentication interface
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.sidebar.title("User Login")

    register = st.sidebar.checkbox("Register")
    if register:
        username = st.sidebar.text_input("Username", key="register_username")
        password = st.sidebar.text_input("Password", type="password", key="register_password")
        if st.sidebar.button("Register"):
            if user_collection.find_one({"username": username}):
                st.sidebar.error("Username already exists")
            else:
                register_user(username, password)
                st.sidebar.success("Registration successful. Please log in.")
    else:
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success("Login successful.")
            else:
                st.sidebar.error("Invalid username or password")
else:
    st.sidebar.write(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

    st.title("Gemma Model Document Q&A")

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

    prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
    )

    def vector_embedding():
        if "vectors" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader("./Documents")  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # vector OpenAI embeddings

    if st.button("Documents Embedding"):
        vector_embedding()
        st.write("Vector Store DB is ready")

    prompt1 = st.text_input("Enter Your Question From Documents")

    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        save_chat_history(st.session_state.username, prompt1, response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

    if st.button("Load Chat History"):
        chat_history = load_chat_history(st.session_state.username)
        for chat in chat_history:
            st.write(f"Q: {chat['question']}")
            st.write(f"A: {chat['answer']}")
            st.write("--------------------------------")
