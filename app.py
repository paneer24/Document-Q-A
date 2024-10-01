import atexit
import datetime
import hashlib
import os
import time
from dotenv import load_dotenv
import streamlit as st
import pymongo
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from googlesearch import search

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(page_title="Chat with PDF using Gemini")

# Error handling for missing API keys or MongoDB URI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if GOOGLE_API_KEY is None:
    st.error("Google API key not found. Please make sure it's set in your .env file.")
    st.stop()

if MONGODB_URI is None:
    st.error("MongoDB URI not found. Please make sure it's set in your .env file.")
    st.stop()

# MongoDB connection and setup
try:
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["chatbot_db"]
    collection = db["chat_history"]
    user_collection = db["users"]
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Ensure MongoDB client is closed when Streamlit app stops
def close_mongo_client():
    client.close()

atexit.register(close_mongo_client)

# Set up Google Gemini-Pro AI model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    hashed_password = hash_password(password)
    user_data = {"username": username, "password": hashed_password}
    user_collection.insert_one(user_data)

def authenticate_user(username, password):
    hashed_password = hash_password(password)
    user = user_collection.find_one({"username": username, "password": hashed_password})
    return user is not None

# Chat history functions
def save_chat_history(username, question, answer, response_time=None):
    chat_data = {
        "username": username,
        "question": question,
        "answer": answer,
        "response_time": response_time,
        "timestamp": datetime.datetime.utcnow()
    }
    collection.insert_one(chat_data)

def load_chat_history(username):
    return collection.find({"username": username}).sort("timestamp", pymongo.ASCENDING)

# PDF processing functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    
    start_time = time.process_time()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    response_time = time.process_time() - start_time
    
    answer = response["output_text"]
    
    # Display answer and response time
    st.write(f"**Answer:** {answer}")
    st.write(f"**Response time:** {response_time:.2f} seconds")
    
    # Display document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write("--------------------------------")
    
    # Google Search results
    st.write("### For More Information:")
    try:
        google_results = list(search(user_question, num_results=4))
        for result in google_results:
            st.write(result)
    except Exception as e:
        st.error(f"Error during Google search: {e}")
    
    return answer, response_time, docs

# Main Streamlit app
def main():
    st.header("Chat with PDF using Gemini")

    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.conversation = []

    # Sidebar for login/logout
    with st.sidebar:
        if not st.session_state.logged_in:
            st.title("User Login")
            login_tab, register_tab = st.tabs(["Login", "Register"])
            
            with login_tab:
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Login"):
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("Login successful.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with register_tab:
                new_username = st.text_input("New Username", key="register_username")
                new_password = st.text_input("New Password", type="password", key="register_password")
                if st.button("Register"):
                    if user_collection.find_one({"username": new_username}):
                        st.error("Username already exists")
                    else:
                        register_user(new_username, new_password)
                        st.success("Registration successful. Please log in.")

    # Main application (only shown when logged in)
    if st.session_state.logged_in:
        st.write(f"Logged in as {st.session_state.username}")
        
        # File uploader
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if pdf_docs:
            if st.button("Process Documents"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")

        # Q&A interface
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            if os.path.exists("faiss_index"):
                answer, response_time, docs = user_input(user_question)
                
                # Save to conversation history
                st.session_state.conversation.append(f"Q: {user_question}")
                st.session_state.conversation.append(f"A: {answer}")
                
                # Save to database
                save_chat_history(st.session_state.username, user_question, answer, response_time)
            else:
                st.warning("Please process documents first.")

        # Show conversation history
        if st.button("Show History"):
            st.write("### Conversation History")
            chat_history = load_chat_history(st.session_state.username)
            for chat in chat_history:
                st.write(f"Q: {chat['question']}")
                st.write(f"A: {chat['answer']}")
                if 'response_time' in chat:
                    st.write(f"Response time: {chat['response_time']:.2f} seconds")
                st.write("---")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()

if __name__ == "__main__":
    main()
