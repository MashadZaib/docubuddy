import warnings
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile
import os
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="Local PDF Chatbot", layout="centered")

@st.cache_resource(show_spinner=False)
def load_embeddings():
    
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)

def create_vector_store(file_bytes, _embeddings):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        if not pages:
            st.error("Could not extract text from the PDF. It might be empty or a scanned image.")
            return None
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        chunks = splitter.split_documents(pages)

        vectorstore = FAISS.from_documents(chunks, _embeddings)
        os.remove(temp_file_path)
        return vectorstore
    
    except Exception as e:
        error_message = str(e)
        if "PDF could not be read" in error_message or "not a PDF" in error_message:
            st.error("This PDF is corrupted or not a valid PDF file.")
        elif "password" in error_message.lower():
            st.error("This PDF is password protected and cannot be read.")
        else:
            st.error(f"An error occurred while processing the PDF: {error_message}")
        return None

def get_conversational_chain(_vector_store, model_name="mistral"):
    llm = Ollama(model=model_name)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return chain

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "source_documents" in message:
                with st.expander("View sources"):
                    for i, doc in enumerate(message["source_documents"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "file_key" not in st.session_state:
        st.session_state.file_key = None
    if "chain" not in st.session_state:
        st.session_state.chain = None

    with st.sidebar:
        st.title("📄 PDF Chat Expert")
        st.markdown("""
        **Instructions:**
        1. Upload a PDF file
        2. Ask questions about its content
        3. Click 'Clear Chat' to start new conversation
        """)
        
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        model_name = 'mistral'
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
            
        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            st.info(f"File size: {len(uploaded_file.getvalue())//1024} KB")

    st.title("Chat with Your PDF")
    st.info("Your chat history is saved for this session. Use the sidebar to clear it.")

    # Process PDF if uploaded
    if uploaded_file:
        file_key = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state.get("file_key") != file_key:
            with st.spinner("Processing PDF... This may take a moment."):
                file_bytes = uploaded_file.getvalue()
                embeddings = load_embeddings()
                vector_store = create_vector_store(file_bytes, embeddings)
                
                if vector_store:
                    st.session_state.chain = get_conversational_chain(vector_store, model_name)
                    st.session_state.file_key = file_key
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                    st.success("Document processed successfully! You can now start chatting.")
                else:
                    st.session_state.chain = None

    # Display chat messages
    display_chat()

    # Handle user input
    user_input = st.chat_input(
        "Ask a question about your document...",
        disabled=(not st.session_state.chain)
    )
    
    if user_input and st.session_state.chain:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain({
                        "question": user_input, 
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response.get("answer", "Sorry, I couldn't find an answer.")
                    st.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_documents": response.get("source_documents", [])
                    })
                    
                    st.session_state.chat_history = response.get("chat_history", [])
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Sorry, something went wrong. Please try again."
                    })

if __name__ == "__main__":
    main()