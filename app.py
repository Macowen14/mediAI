import streamlit as st
import os
import logging
import time
import subprocess
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import OllamaEmbeddings

from src.rag_pipeline import MedicalRAGPipeline
from src.vector_utils import VectorStore
from src.logger import LoggerSetup
from src.enums import ModelType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# Page config
st.set_page_config(
    page_title="MediAI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .stChatMessage[data-testid="stChatMessageAvatarUser"] {
        background-color: #f0f2f6;
    }
    .metric-card {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .source-citation {
        font-size: 0.8em;
        color: #666;
        border-left: 2px solid #ccc;
        padding-left: 0.5rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "models" not in st.session_state:
    st.session_state.models = []

def get_ollama_models():
    """Fetch available models from Ollama CLI."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:] # Skip header
            models = [line.split()[0] for line in lines]
            return models
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
    return ["ministral-3:8b", "deepseek-v3.1:671b-cloud"] # Fallback

# Initialize Resources
@st.cache_resource
def get_pipeline():
    """Initialize the RAG pipeline."""
    load_dotenv()
    
    # Check API key
    if not os.getenv("PINECONE_API_KEY"):
        st.error("PINECONE_API_KEY not found in environment variables!")
        st.stop()
        
    embeddings = OllamaEmbeddings(model=ModelType.EMBEDDING.value)
    index_name = "mediai-bot"
    
    # Connect to vectorstore
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    
    return MedicalRAGPipeline(vectorstore)

# Sidebar
with st.sidebar:
    st.title("🏥 MediAI Settings")
    
    # Model Selection
    if not st.session_state.models:
        st.session_state.models = get_ollama_models()
        
    selected_model = st.selectbox(
        "Select Model", 
        st.session_state.models,
        index=0 if st.session_state.models else 0
    )
    
    # Update ModelType enum dynamically if needed (or just rely on default for now)
    # TODO: Make ModelManager accept dynamic model names based on selection
    
    st.divider()
    
    # RAG Parameters
    search_k = st.slider("Context Documents (K)", 1, 10, 3)
    relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.5)
    
    st.divider()
    st.markdown("### 📊 Capabilities")
    st.info("""
    - **Theme Detection**: Identifies 10+ medical themes
    - **RAG Pipeline**: Retrieves from Pinecone
    - **Evidence-Based**: Cites sources
    - **Safety**: Fallback mechanisms included
    """)
    
    # Scalability Note
    with st.expander("Scalability Notes"):
        st.caption("""
        **Future Improvements:**
        - **Async Processing**: Switch to `AsyncLLM` for concurrent user support.
        - **Caching**: Implement Redis/Memcached for frequent queries.
        - **User Session**: Persist history in database (Postgres).
        - **Queueing**: Use Celery/Redis Queue for batch processing.
        """)

# Main Interface
st.title("Medical AI Assistant")
st.caption("Powered by RAG & Ollama | Evidence-based Answers")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Reference Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

class StreamHandler(BaseCallbackHandler):
    """Custom callback handler for streaming to Streamlit."""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

class StepCallback:
    """Callback to update streamlit status context."""
    def __init__(self, container):
        self.container = container
        
    def __call__(self, step_msg):
        self.container.write(step_msg)
        time.sleep(0.1) # Small visual delay for effect

# Input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant"):
        pipeline = get_pipeline()
        
        # Thinking Process Visualization
        with st.status("Thinking...", expanded=True) as status:
            st.write("Initializing pipeline...")
            step_updater = StepCallback(status)
            
            # Streaming Container
            response_placeholder = st.empty()
            # Use custom StreamHandler instead of StreamlitCallbackHandler to avoid RecursionError
            stream_handler = StreamHandler(response_placeholder)
            
            try:
                # Process Question
                answer = pipeline.process_question(
                    question=prompt,
                    search_k=search_k,
                    relevance_threshold=relevance_threshold,
                    status_callback=step_updater,
                    streaming_callback=[stream_handler]
                )
                
                status.update(label="Complete!", state="complete", expanded=False)
                
                # If streaming didn't fill the placeholder (e.g. short answer or fallback), ensure it's shown
                response_placeholder.markdown(answer.answer)
                
                # Show sources
                if answer.sources:
                    with st.expander("Reference Sources"):
                        for src in answer.sources:
                            st.markdown(f"- {src}")
                
                # Build message history object
                msg_data = {
                    "role": "assistant",
                    "content": answer.answer,
                    "sources": answer.sources,
                    "theme": answer.theme
                }
                
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Streamlit App Error: {e}", exc_info=True)
