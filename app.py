import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# -------------------------
# Load environment variables
# -------------------------
# -------------------------
# Load API keys (local + Streamlit Cloud)
# -------------------------
load_dotenv()

OPENAI_API_KEY = st.secrets.get(
    "OPENAI_API_KEY",
    os.getenv("OPENAI_API_KEY")
)

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -------------------------
# Streamlit page setup
# -------------------------
st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .chat-bubble-user {
        background: #1e3a5f;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
        color: #dce8ff;
    }
    .chat-bubble-bot {
        background: #1a1f2e;
        border: 1px solid #2a3555;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 80%;
        color: #c8d8ff;
    }
    .chat-label {
        font-size: 0.72rem;
        color: #7a8aaa;
        margin-bottom: 2px;
    }
    .stTextInput > div > div > input {
        background-color: #1a1f2e;
        color: #e0e0e0;
        border: 1px solid #2a3555;
        border-radius: 8px;
    }
    .stButton > button {
        background-color: #1e3a5f;
        color: #dce8ff;
        border: 1px solid #2a5090;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2a5090;
        border-color: #4a80d0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Initialize session state
# -------------------------
for key, default in {
    "vectors": None,
    "chat_history": [],
    "last_chunks": [],
    "embed_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# Validate API keys
# -------------------------
missing_keys = []

if not OPENAI_API_KEY:
    missing_keys.append("OPENAI_API_KEY")

if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if missing_keys:
    st.error(
        f"Missing API keys: {', '.join(missing_keys)}. "
        "Please check Streamlit Secrets."
    )
    st.stop()

# -------------------------
# Load LLM (cached)
# -------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

llm = load_llm()

# -------------------------
# Prompt template
# -------------------------
prompt = ChatPromptTemplate.from_template("""
You are an intelligent AI assistant. Answer the user's question ONLY
from the provided context. Be concise and accurate.

If the answer is not available in the context, say:
"I could not find this information in the uploaded document."

Context:
{context}

Question: {question}
""")

# -------------------------
# Create vector database
# -------------------------
def create_vector_embeddings(uploaded_files):
    documents = []
    temp_paths = []

    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_paths.append(tmp.name)

            loader = PyPDFLoader(temp_paths[-1])
            try:
                docs = loader.load()
                if docs:
                    documents.extend(docs)
                else:
                    st.warning(f"No content found in {uploaded_file.name}. Skipping.")
            except Exception as e:
                st.warning(f"Could not read {uploaded_file.name}: {e}")

        if not documents:
            raise ValueError("No readable content found in the uploaded PDFs.")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        final_documents = text_splitter.split_documents(documents)
        vectors = FAISS.from_documents(final_documents, embeddings)
        return vectors

    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

# -------------------------
# Build RAG chain using LCEL
# -------------------------
def build_rag_chain(retriever):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("## Document Setup")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents to query."
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.markdown(f"- {f.name}")

        if st.button("Build Knowledge Base"):
            with st.spinner("Processing PDFs and creating embeddings..."):
                try:
                    st.session_state.vectors = create_vector_embeddings(uploaded_files)
                    st.session_state.embed_done = True
                    st.session_state.chat_history = []
                    st.session_state.last_chunks = []
                    st.success("Knowledge base ready!")
                except Exception as e:
                    st.error(f"Failed to create embeddings: {e}")
                    st.session_state.vectors = None
                    st.session_state.embed_done = False
    else:
        st.info("Upload PDFs above to get started.")

    st.markdown("---")

    if st.session_state.embed_done:
        st.markdown("🟢 **Status:** Knowledge base active")
    else:
        st.markdown("🔴 **Status:** No knowledge base loaded")

    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.last_chunks = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>Powered by LangChain · Groq · OpenAI · FAISS</small>",
        unsafe_allow_html=True
    )

# -------------------------
# Main Chat UI
# -------------------------
st.markdown("# AI RAG Document Chatbot")
st.markdown("Ask questions from your uploaded documents.")
st.markdown("---")

if st.session_state.chat_history:
    st.markdown("### Conversation")
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f"<div class='chat-label'>You</div>"
                f"<div class='chat-bubble-user'>{message}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='chat-label'>Assistant</div>"
                f"<div class='chat-bubble-bot'>{message}</div>",
                unsafe_allow_html=True
            )
    st.markdown("---")

col1, col2 = st.columns([5, 1])
with col1:
    user_prompt = st.text_input(
        "Your question",
        placeholder="e.g. What is the main topic of the document?",
        label_visibility="collapsed",
        key="user_input"
    )
with col2:
    ask_btn = st.button("Ask", use_container_width=True)

# -------------------------
# Query processing
# -------------------------
if ask_btn and user_prompt.strip():
    if st.session_state.vectors is None:
        st.warning("Please upload PDF(s) and build the knowledge base first (see sidebar).")
    else:
        with st.spinner("Searching documents..."):
            try:
                retriever = st.session_state.vectors.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                chunks = retriever.invoke(user_prompt.strip())
                rag_chain = build_rag_chain(retriever)

                start = time.time()
                answer = rag_chain.invoke(user_prompt.strip())
                elapsed = time.time() - start

                st.session_state.chat_history.append(("You", user_prompt.strip()))
                st.session_state.chat_history.append(("Bot", answer))
                st.session_state.last_chunks = chunks

                st.markdown(
                    f"<small style='color:#555'>Response time: {elapsed:.2f}s</small>",
                    unsafe_allow_html=True
                )
                st.rerun()

            except Exception as e:
                st.error(f"Error during retrieval: {e}")

# -------------------------
# Retrieved chunks
# -------------------------
if st.session_state.last_chunks:
    with st.expander("Retrieved Document Chunks", expanded=False):
        for i, doc in enumerate(st.session_state.last_chunks):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Chunk {i+1}** — Source: `{os.path.basename(source)}` | Page: {page}")
            st.markdown(
                f"<div style='background:#1a1f2e;padding:10px;border-radius:8px;"
                f"font-size:0.85rem;color:#aab8d8'>{doc.page_content}</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")