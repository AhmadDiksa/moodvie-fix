"""
Streamlit Movie Therapist Agent
A conversational AI agent that analyzes user moods and recommends movies using an agentic architecture.
"""

import streamlit as st
import os
import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

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
    .streaming-link {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
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
        help="Dapatkan dari https://programmablesearchengine.google.com/"
    )
    search_api_key = st.text_input(
        "Search API Key",
        value=os.getenv("GOOGLE_SEARCH_API_KEY", ""),
        type="password",
        help="Dapatkan dari Google Cloud Console"
    )
    
    # Qdrant Configuration
    st.subheader("Qdrant Vector Database")
    qdrant_url = st.text_input(
        "Qdrant URL",
        value=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="URL ke Qdrant instance (Cloud atau local)"
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
    
    This agent coordinates between:
    1. Mood Analysis (LLM)
    2. Movie Retrieval Tool (Qdrant Vector DB)
    3. Streaming Link Finder Tool (Google Custom Search)
    """
    
    def __init__(
        self,
        google_api_key: str,
        search_engine_id: str,
        search_api_key: str,
        qdrant_url: str,
        qdrant_api_key: Optional[str] = None,
    ):
        """Initialize the Movie Therapist Agent with all required API clients."""
        
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
        
        # Store credentials for later use
        self.google_api_key = google_api_key
        
        # Initialize tools
        self.tools = [self.get_recommendations_tool, self.get_streaming_links_tool]
        
        # Create agent
        self._setup_agent()
    
    # ==========================================================================
    # TOOL 1: MOOD_MOVIE_RETRIEVER
    # ==========================================================================
    
    @tool
    def get_recommendations_tool(self, mood_query: str) -> str:
        """
        TOOL: Mood Movie Retriever
        
        Searches the Qdrant vector database for movies based on the user's 
        emotional state or plot keywords. Returns movie details including title, 
        plot, poster, and raw reviews.
        
        Args:
            mood_query: User's emotional state or movie preferences (e.g., "sedih", "action", "komedi")
            
        Returns:
            JSON string with recommended movies and their details
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
                        "similarity_score": result.score,
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
    
    @tool
    def get_streaming_links_tool(self, movie_title: str) -> str:
        """
        TOOL: Streaming Link Finder
        
        Uses Google Custom Search to find legal streaming links (Netflix, Disney+, etc.) 
        for a specific movie title.
        
        Args:
            movie_title: The movie title to search for
            
        Returns:
            Formatted string with streaming links or availability info
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
                    for item in results["items"][:3]:  # Top 3 results
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
    # AGENT SETUP
    # ==========================================================================
    
    def _setup_agent(self):
        """Set up the agent by binding tools to the LLM."""
        
        # Convert tools to a dictionary for easy lookup
        self.tools_dict = {tool.name: tool for tool in self.tools}
    
    def _should_use_tool(self, response_text: str) -> tuple[bool, str, str]:
        """
        Parse LLM response to check if it wants to use a tool.
        Returns (should_use_tool, tool_name, tool_input)
        """
        import json
        
        # Check for tool usage patterns
        if "TOOL:" in response_text and "ACTION:" in response_text:
            try:
                # Simple pattern matching for tool calls
                lines = response_text.split('\n')
                tool_name = None
                tool_input = None
                
                for i, line in enumerate(lines):
                    if "TOOL:" in line:
                        tool_name = line.split("TOOL:")[1].strip()
                    elif "ACTION:" in line or "ARGUMENT:" in line:
                        tool_input = line.split(":", 1)[1].strip()
                
                if tool_name and tool_input:
                    return True, tool_name, tool_input
            except:
                pass
        
        return False, None, None
    
    def run_chat_turn(self, user_input: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Run one turn of the conversation with the agent.
        
        Args:
            user_input: User's message
            chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Agent's response
        """
        try:
            # Build system prompt
            system_prompt = """Anda adalah 'Movie Therapist' - seorang AI yang memahami emosi manusia dan merekomendasikan film yang sempurna untuk suasana hati mereka.

Proses Anda:
1. Dengarkan dan pahami mood/keadaan emosional user
2. Jika perlu mencari film: tuliskan "TOOL: get_recommendations_tool" dan "ACTION: [mood/genre user]"
3. Jika perlu cari link: tuliskan "TOOL: get_streaming_links_tool" dan "ACTION: [movie title]"
4. Berikan ringkasan review film dengan bahasa santai dan dalam Bahasa Indonesia

Gaya komunikasi Anda:
- Empatik dan memahami
- Santai dan ramah
- Memberikan insight tentang mengapa film ini cocok untuk mood mereka
- Gunakan emoji yang sesuai

Available tools:
1. get_recommendations_tool - Cari film berdasarkan mood
2. get_streaming_links_tool - Cari link streaming film"""
            
            # Build conversation messages
            messages = [
                HumanMessage(content=system_prompt),
            ]
            
            # Add chat history
            for msg in chat_history[-3:]:  # Last 3 messages
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current user input
            messages.append(HumanMessage(content=user_input))
            
            # Get LLM response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Check if LLM wants to use a tool
            max_iterations = 3
            iteration = 0
            tool_results = []
            
            while iteration < max_iterations:
                should_use_tool, tool_name, tool_input = self._should_use_tool(response_text)
                
                if not should_use_tool:
                    break
                
                # Execute tool if found
                if tool_name and tool_name in self.tools_dict:
                    try:
                        tool_func = self.tools_dict[tool_name]
                        # Call tool - handle both sync and methods
                        if tool_name == "get_recommendations_tool":
                            tool_result = self.get_recommendations_tool(tool_input)
                        elif tool_name == "get_streaming_links_tool":
                            tool_result = self.get_streaming_links_tool(tool_input)
                        else:
                            tool_result = str(tool_func(tool_input))
                        
                        tool_results.append(f"\nTool Result for {tool_name}:\n{tool_result}")
                        
                        # Ask LLM to continue with tool result
                        messages.append(AIMessage(content=response_text))
                        messages.append(HumanMessage(content=f"Tool result:\n{tool_result}\n\nPlease provide your response based on this information."))
                        
                        response = self.llm.invoke(messages)
                        response_text = response.content
                    except Exception as e:
                        tool_results.append(f"Error executing {tool_name}: {str(e)}")
                        break
                else:
                    break
                
                iteration += 1
            
            return response_text
        
        except Exception as e:
            error_msg = f"Error in chat turn: {str(e)}"
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
                    st.session_state.chat_history[:-1],  # Exclude the latest user message
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
