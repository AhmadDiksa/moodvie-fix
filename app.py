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
    
    /* Netflix-style Movie Cards */
    .movie-cards-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 0.75rem;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        color: white;
        position: relative;
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
    }
    
    .movie-poster {
        width: 100%;
        height: 280px;
        object-fit: cover;
        background: linear-gradient(45deg, #333, #555);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
    }
    
    .movie-info {
        padding: 1rem;
    }
    
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .movie-rating {
        color: #e50914;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .movie-genre {
        font-size: 0.85rem;
        color: #b3b3b3;
        margin-bottom: 0.75rem;
    }
    
    .movie-plot {
        font-size: 0.85rem;
        color: #e5e5e5;
        line-height: 1.4;
        height: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
    }
    
    .movie-reviews {
        font-size: 0.8rem;
        color: #ffb000;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid #333;
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
# HELPER FUNCTIONS FOR DISPLAY
# =============================================================================

def display_movie_cards(movies_data: List[Dict[str, Any]]) -> None:
    """
    Display movies in Netflix-style card grid.
    
    Args:
        movies_data: List of movie dictionaries with keys: title, plot, poster_url, 
                    raw_reviews, genre, rating, etc.
    """
    if not movies_data:
        st.warning("Tidak ada film yang ditemukan")
        return
    
    # Create columns for cards
    cols = st.columns(min(3, len(movies_data)))
    
    for idx, movie in enumerate(movies_data):
        col = cols[idx % len(cols)]
        
        with col:
            # Create card container
            card_html = f"""
            <div class="movie-card">
                <div class="movie-poster">
            """
            
            # Add poster if available
            poster_url = movie.get("poster_url", "") or movie.get("poster", "")
            if poster_url and poster_url.startswith("http"):
                card_html = f"""
                <div class="movie-card">
                    <img src="{poster_url}" class="movie-poster" onerror="this.parentElement.style.backgroundColor='#333'">
                """
            else:
                # Fallback emoji based on genre
                genre = movie.get("genre", "Film").lower()
                emoji = "🎬"
                if "horror" in genre or "seram" in genre:
                    emoji = "😱"
                elif "action" in genre or "aksi" in genre:
                    emoji = "💥"
                elif "comedy" in genre or "komedi" in genre:
                    emoji = "😂"
                elif "romance" in genre or "romantis" in genre:
                    emoji = "💕"
                elif "drama" in genre:
                    emoji = "😢"
                elif "animation" in genre or "animasi" in genre:
                    emoji = "🎨"
                
                card_html = f"""
                <div class="movie-card">
                    <div class="movie-poster" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 60px;">
                        {emoji}
                    </div>
                """
            
            # Movie info
            title = movie.get("title", "Unknown Title")
            rating = movie.get("rating", "N/A")
            genre = movie.get("genre", "N/A")
            plot = movie.get("plot", "No description available")[:100]
            reviews = movie.get("raw_reviews", "")[:50]
            
            card_html += f"""
                    <div class="movie-info">
                        <div class="movie-title">{title}</div>
                        <div class="movie-rating">⭐ {rating}</div>
                        <div class="movie-genre">{genre}</div>
                        <div class="movie-plot">{plot}...</div>
                        <div class="movie-reviews">💬 {reviews}</div>
                    </div>
                </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)

def parse_movie_data(recommendations_str: str) -> List[Dict[str, Any]]:
    """
    Parse movie recommendations from string format to dict.
    
    Args:
        recommendations_str: String representation of recommendations list
        
    Returns:
        List of parsed movie dictionaries
    """
    try:
        import ast
        # Try to parse as Python literal
        movies = ast.literal_eval(recommendations_str)
        if isinstance(movies, list):
            return movies
    except:
        pass
    
    return []

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
        
        # Initialize Embeddings with proper device settings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={
                    "device": "cpu",  # Force CPU to avoid meta tensor issues
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                }
            )
        except Exception as e:
            # Fallback without device specification
            st.warning(f"⚠️ Embeddings initialization with device failed, using default: {str(e)}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={
                    "normalize_embeddings": True,
                }
            )
        
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
                        "poster_url": result.payload.get("poster_url", ""),
                        "raw_reviews": result.payload.get("raw_reviews", ""),
                        "genre": result.payload.get("genre", "N/A"),
                        "rating": result.payload.get("rating", "N/A"),
                        "similarity_score": result.score,
                    }
                    recommendations.append(movie_data)
                
                # Store in session state for display
                if recommendations:
                    st.session_state.last_recommendations = recommendations
                
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
        self.tools_dict = {
            "get_recommendations_tool": self.get_recommendations_tool,
            "get_streaming_links_tool": self.get_streaming_links_tool,
        }
    
    def _should_use_tool(self, response_text: str) -> tuple[bool, str, str]:
        """
        Parse LLM response to check if it wants to use a tool.
        Returns (should_use_tool, tool_name, tool_input)
        """
        response_lower = response_text.lower()
        
        # Pattern 1: Check for recommendation keywords
        if any(keyword in response_lower for keyword in ["cari film", "rekomendasi", "film yang cocok", "pencarian film", "search movie"]):
            return True, "get_recommendations_tool", response_text
        
        # Pattern 2: Check for streaming keywords
        if any(keyword in response_lower for keyword in ["link streaming", "di mana nonton", "watch", "dimana nonton", "streaming"]):
            # Extract movie title
            lines = response_text.split('\n')
            for line in lines:
                if '"' in line or "'" in line:
                    return True, "get_streaming_links_tool", line
            return True, "get_streaming_links_tool", response_text
        
        # Pattern 3: Explicit tool markers
        if "TOOL:" in response_text:
            try:
                lines = response_text.split('\n')
                tool_name = None
                tool_input = None
                
                for line in lines:
                    if "TOOL:" in line:
                        tool_name = line.split("TOOL:")[1].strip()
                    elif "ACTION:" in line or "INPUT:" in line:
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
            chat_history: List of previous messages
            
        Returns:
            Agent's response
        """
        try:
            # System prompt
            system_prompt = """Anda adalah 'Movie Therapist' - seorang AI yang memahami emosi manusia dan merekomendasikan film yang sempurna untuk suasana hati mereka.

Instruksi:
1. Pahami suasana hati/mood user
2. Jika user minta rekomendasi: Selalu gunakan frasa "cari film berdasarkan mood" atau "film yang cocok"
3. Jika user minta tahu di mana nonton: Selalu gunakan frasa "link streaming" atau "di mana nonton"
4. Jelaskan dengan bahasa santai dan empatik dalam Bahasa Indonesia
5. Gunakan emoji yang sesuai

Prioritas: SELALU carikan rekomendasi film jika user menunjukkan emosi/mood apapun!"""
            
            # Build messages
            messages = [HumanMessage(content=system_prompt)]
            
            # Add recent chat history (max 2 messages)
            for msg in chat_history[-2:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current input
            messages.append(HumanMessage(content=user_input))
            
            # Get initial response from LLM
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Check if we need to use tools
            max_iterations = 2
            iteration = 0
            
            while iteration < max_iterations:
                should_use_tool, tool_name, tool_input = self._should_use_tool(response_text)
                
                if not should_use_tool:
                    break
                
                # Execute the appropriate tool
                try:
                    if tool_name == "get_recommendations_tool":
                        # Use user input or extracted mood
                        tool_result = self.get_recommendations_tool(user_input)
                        
                        # Display movie cards immediately
                        st.markdown("### 🎬 Rekomendasi Film Untukmu:")
                        movies = parse_movie_data(tool_result)
                        if movies:
                            display_movie_cards(movies)
                    
                    elif tool_name == "get_streaming_links_tool":
                        # Extract movie title from response or input
                        movie_title = tool_input if tool_input != response_text else "unknown"
                        tool_result = self.get_streaming_links_tool(movie_title)
                    else:
                        break
                    
                    # Ask LLM to summarize tool result
                    messages.append(AIMessage(content=response_text))
                    summary_prompt = f"""Tool result untuk {tool_name}:
{tool_result}

Sekarang buatkan ringkasan yang ramah dan membantu untuk user. Dalam Bahasa Indonesia, santai, dan empatik."""
                    messages.append(HumanMessage(content=summary_prompt))
                    
                    response = self.llm.invoke(messages)
                    response_text = response.content
                    
                except Exception as e:
                    st.error(f"Error executing tool: {str(e)}")
                    break
                
                iteration += 1
            
            return response_text
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            import traceback
            st.write(traceback.format_exc())
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
    
    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = []
    
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
