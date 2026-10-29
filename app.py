import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Setup ---
st.set_page_config(page_title="Gemini AI Assistant & RAG Agent", page_icon="🤖")
st.title("🤖 Dual-Mode Gemini Chatbot")

# Retrieve Gemini API key from Streamlit secrets or environment variable
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Please set the GEMINI_API_KEY in Streamlit Secrets or as an environment variable.")
    st.stop()

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

# --- Sidebar: Document Management & Controls ---
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload a PDF file to enable RAG", type=["pdf"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # 1. Extract text from PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                # 2. Split document into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # 3. Create Vector Store with Google Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-2-preview", 
                    google_api_key=api_key
                )
                st.session_state.vector_store = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
                st.session_state.document_processed = True
                st.success("Document loaded successfully! Switching to RAG mode.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.markdown("---")
    
    # Mode indicator/selector
    if st.session_state.document_processed:
        st.info("📌 **Status**: RAG Mode (Document Loaded)")
        if st.button("Clear Document & Switch to General Chat"):
            st.session_state.vector_store = None
            st.session_state.document_processed = False
            st.rerun()
    else:
        st.info("💬 **Status**: General Chat Mode (No Document Uploaded)")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main Interface: Chat Render ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input Box
user_query = st.chat_input("Ask a question or start a conversation...")

if user_query:
    # 1. Display user input
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Build model instance
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.document_processed and st.session_state.vector_store:
                # --- RAG Mode Execution ---
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are a helpful assistant. Use the following context from an uploaded document "
                     "to answer the question. If the user asks something unrelated to the document, "
                     "you can answer using your general knowledge while mentioning that it isn't in the document.\n\n"
                     "Document Context:\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])

                chain = rag_prompt | llm
                response = chain.invoke({
                    "context": context_text,
                    "chat_history": st.session_state.chat_history,
                    "input": user_query
                })
            else:
                # --- Standard Conversational Mode Execution ---
                general_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a friendly, intelligent conversational AI chatbot. Help the user with general questions, task planning, or friendly chat."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])

                chain = general_prompt | llm
                response = chain.invoke({
                    "chat_history": st.session_state.chat_history,
                    "input": user_query
                })

            answer = response.content
            st.markdown(answer)

    # Update conversation history memory state
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=answer))
                
