import streamlit as st
import os
from dotenv import load_dotenv
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from llama_index.readers.database import DatabaseReader
import sqlalchemy

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="SQL Data ChatBot", page_icon="🤖")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner=False)
def load_data():
    # Create database connection
    db_connection = sqlalchemy.create_engine("sqlite:///your_database.db")

    # Initialize database reader
    reader = DatabaseReader(
        sqlalchemy_engine=db_connection
    )

    # Load documents from database
    documents = reader.load_data()

    # Create LLM
    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

    # Create service context
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context
    )

    return index

# Main app
st.title("💬 SQL Data ChatBot")
st.caption("🚀 A chatbot powered by OpenAI and LlamaIndex")

# Load the index
with st.spinner("Loading the database..."):
    index = load_data()
    chat_engine = index.as_chat_engine(chat_mode="condense_question")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your data"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

# Add sidebar with instructions
with st.sidebar:
    st.markdown("""
    ## How to use
    1. Make sure your SQL database is properly connected
    2. Ask questions about your data in natural language
    3. The bot will analyze your database and provide relevant answers

    ## About
    This chatbot uses:
    - LlamaIndex for RAG (Retrieval-Augmented Generation)
    - OpenAI's GPT-3.5 for natural language processing
    - Streamlit for the user interface
    """)
