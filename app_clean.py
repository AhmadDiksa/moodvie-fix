"""
Streamlit Movie Therapist Agent
A conversational AI agent that analyzes user moods and recommends movies using an agentic architecture.
"""

import streamlit as st
import os
import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Qdrant imports
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_COLLECTION = "moodviedb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
SIMILARITY_K = 3

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="🎬 Movie Therapist Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.title("⚙️ Konfigurasi API")
    st.markdown("---")
    
    # Google API Configuration
    st.subheader("Google Generative AI")
    google_api_key = st.text_input(
        "Google API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Dapatkan dari https://aistudio.google.com/app/apikeys"
    )
    
    # Google Custom Search Configuration
    st.subheader("Google Custom Search")
    search_engine_id = st.text_input(
        "Search Engine ID",
        value=os.getenv("GOOGLE_SEARCH_ENGINE_ID", ""),
        type="password",
    )
    search_api_key = st.text_input(
        "Search API Key",
        value=os.getenv("GOOGLE_SEARCH_API_KEY", ""),
        type="password",
    )
    
    # Qdrant Configuration
    st.subheader("Qdrant Vector Database")
    qdrant_url = st.text_input(
        "Qdrant URL",
        value=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    qdrant_api_key = st.text_input(
        "Qdrant API Key (jika perlu)",
        value=os.getenv("QDRANT_API_KEY", ""),
        type="password"
    )
    
    st.markdown("---")
    st.info("💡 Simpan API keys di file `.env` untuk keamanan lebih baik.")


# =============================================================================
# MOVIETHERAPIST AGENT CLASS
# =============================================================================

class MovieTherapistAgent:
    """
    Agentic Architecture for Movie Recommendations with Mood Analysis.
    """
    
    def __init__(
        self,
        google_api_key: str,
        search_engine_id: str,
        search_api_key: str,
        qdrant_url: str,
        qdrant_api_key: Optional[str] = None,
    ):
        """Initialize the Movie Therapist Agent."""
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=google_api_key,
            temperature=0.7,
        )
        
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize Qdrant Client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
        )
        
        # Store Search API credentials
        self.search_engine_id = search_engine_id
        self.search_api_key = search_api_key
    
    # ==========================================================================
    # TOOL 1: MOOD_MOVIE_RETRIEVER
    # ==========================================================================
    
    def get_recommendations(self, mood_query: str) -> str:
        """
        TOOL: Mood Movie Retriever
        
        Searches the Qdrant vector database for movies based on emotional state.
        Returns movie details including title, plot, poster, and reviews.
        """
        try:
            with st.spinner("🔍 Sedang mencari film yang cocok untuk suasana hati Anda..."):
                # Embed the mood query
                query_embedding = self.embeddings.embed_query(mood_query)
                
                # Search in Qdrant
                search_results = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_embedding,
                    limit=SIMILARITY_K,
                    with_payload=True,
                )
                
                # Format results
                recommendations = []
                for result in search_results:
                    movie_data = {
                        "title": result.payload.get("title", "N/A"),
                        "plot": result.payload.get("plot", "N/A"),
                        "poster": result.payload.get("poster_url", ""),
                        "raw_reviews": result.payload.get("raw_reviews", ""),
                        "genre": result.payload.get("genre", "N/A"),
                        "rating": result.payload.get("rating", "N/A"),
                    }
                    recommendations.append(movie_data)
                
                return str(recommendations)
        
        except Exception as e:
            error_msg = f"Error searching Qdrant: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    # ==========================================================================
    # TOOL 2: STREAMING_LINK_FINDER
    # ==========================================================================
    
    def get_streaming_links(self, movie_title: str) -> str:
        """
        TOOL: Streaming Link Finder
        
        Uses Google Custom Search to find legal streaming links.
        """
        try:
            with st.spinner(f"🔗 Sedang mencari link streaming untuk '{movie_title}'..."):
                search_query = f"{movie_title} streaming link netflix disney+ prime video"
                
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "q": search_query,
                    "cx": self.search_engine_id,
                    "key": self.search_api_key,
                    "num": 5,
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                results = response.json()
                streaming_info = []
                
                if "items" in results:
                    for item in results["items"][:3]:
                        streaming_info.append({
                            "title": item.get("title", "N/A"),
                            "link": item.get("link", "N/A"),
                            "snippet": item.get("snippet", "N/A"),
                        })
                
                return str(streaming_info) if streaming_info else "Tidak menemukan link streaming."
        
        except Exception as e:
            error_msg = f"Error searching for streaming links: {str(e)}"
            st.warning(error_msg)
            return error_msg
    
    # ==========================================================================
    # AGENT LOGIC
    # ==========================================================================
    
    def run_chat_turn(self, user_input: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Run one turn of conversation with automatic tool calling.
        """
        try:
            system_prompt = """Anda adalah 'Movie Therapist' - AI yang memahami emosi dan merekomendasikan film sempurna.

PENTING: Ketika user minta rekomendasi film atau menceritakan mood mereka:
1. Pahami suasana hati mereka
2. LANGSUNG gunakan frasa "cari film" atau "rekomendasi film" dalam respons Anda
3. Ini akan trigger pencarian otomatis di database kami

Jika user bertanya tentang streaming:
- Gunakan frasa "link streaming" atau "di mana nonton"

Gaya: Empatik, santai, ramah, Bahasa Indonesia, gunakan emoji."""
            
            # Build messages
            messages = [HumanMessage(content=system_prompt)]
            
            # Add chat history (last 2 messages for context)
            for msg in chat_history[-2:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current input
            messages.append(HumanMessage(content=user_input))
            
            # Get initial response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Check if tools should be called
            response_lower = response_text.lower()
            
            # Tool 1: Recommendations
            if any(keyword in response_lower for keyword in ["cari film", "rekomendasi", "film yang cocok", "pencarian"]):
                tool_result = self.get_recommendations(user_input)
                
                # Ask LLM to summarize
                messages.append(AIMessage(content=response_text))
                summary_prompt = f"""Berdasarkan hasil pencarian film ini:
{tool_result}

Buatkan ringkasan yang ramah dan membantu untuk user. Sertakan:
- Judul film yang cocok
- Plot singkat
- Mengapa film ini cocok untuk moodnya
- Genre dan rating
Gunakan Bahasa Indonesia, santai, dan empatik dengan emoji."""
                messages.append(HumanMessage(content=summary_prompt))
                
                response = self.llm.invoke(messages)
                response_text = response.content
            
            # Tool 2: Streaming links
            elif any(keyword in response_lower for keyword in ["link streaming", "di mana nonton", "where to watch"]):
                # Try to extract movie title
                movie_title = user_input if len(user_input) < 50 else response_text
                tool_result = self.get_streaming_links(movie_title)
                
                # Ask LLM to summarize
                messages.append(AIMessage(content=response_text))
                summary_prompt = f"""Hasil pencarian streaming link:
{tool_result}

Format ulang dalam Bahasa Indonesia yang ramah. Tunjukkan platform mana yang tersedia dan link-nya."""
                messages.append(HumanMessage(content=summary_prompt))
                
                response = self.llm.invoke(messages)
                response_text = response.content
            
            return response_text
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            return error_msg


# =============================================================================
# STREAMLIT APP MAIN
# =============================================================================

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    
    # Main title
    st.title("🎬 Movie Therapist Agent")
    st.markdown("*Temukan film yang sempurna berdasarkan suasana hati Anda*")
    st.markdown("---")
    
    # Initialize agent if APIs are provided
    if not st.session_state.agent_initialized:
        if not google_api_key or not search_engine_id or not search_api_key:
            st.error("❌ Mohon isi semua API key di sidebar untuk memulai!")
            st.stop()
        
        try:
            with st.spinner("🚀 Menginisialisasi Movie Therapist Agent..."):
                st.session_state.agent = MovieTherapistAgent(
                    google_api_key=google_api_key,
                    search_engine_id=search_engine_id,
                    search_api_key=search_api_key,
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key if qdrant_api_key else None,
                )
                st.session_state.agent_initialized = True
            st.success("✅ Agent siap! Mulai bercerita tentang suasana hati Anda...")
        except Exception as e:
            st.error(f"❌ Gagal menginisialisasi agent: {str(e)}")
            st.stop()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ceritakan tentang suasana hati Anda atau minta rekomendasi film..."):
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get agent response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                response = st.session_state.agent.run_chat_turn(
                    user_input,
                    st.session_state.chat_history[:-1],
                )
                
                message_placeholder.markdown(response)
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                })
            
            except Exception as e:
                error_response = f"❌ Error: {str(e)}"
                message_placeholder.error(error_response)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_response,
                })


if __name__ == "__main__":
    main()
