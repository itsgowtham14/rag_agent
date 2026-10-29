import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Setup ---
st.set_page_config(page_title="RAG Document Chatbot", page_icon="📄")
st.title("📄 Chat with your Document (Google GenAI RAG)")

# Retrieve Gemini API key from Streamlit secrets or environment variable
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Please set the GEMINI_API_KEY in Streamlit Secrets or as an environment variable.")
    st.stop()

# Initialize session states
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing and indexing document..."):
            # Save uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # 1. Load document
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 2. Chunk text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # 3. Embed & store in ChromaDB vector store
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=api_key
            )
            st.session_state.vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            
            # Clean up temporary file
            os.remove(tmp_path)
            st.success("Document processed successfully!")

# --- Main Interface: Chat ---
# Render past messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
user_query = st.chat_input("Ask a question about your document...")

if user_query:
    if not st.session_state.vector_store:
        st.warning("Please upload and process a PDF document first.")
    else:
        # Append and display user prompt
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching document & thinking..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.3
                )

                # Prompt formulation
                system_prompt = (
                    "You are an expert assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know based on the provided document.\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # Execute RAG execution loop
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]

                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
              
