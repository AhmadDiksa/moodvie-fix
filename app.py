import streamlit as st
import os
import requests
from typing import List
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# ==========================================
# 1. CONFIG & CSS INJECTION (CRITICAL)
# ==========================================

st.set_page_config(
    page_title="Movie Therapist",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Netflix-inspired Dark Theme & Custom Card CSS
custom_css = """
<style>
    /* Global Reset & Dark Mode */
    .stApp {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Hide Streamlit Boilerplate */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    /* Chat Input Styling */
    .stChatInput {
        position: fixed;
        bottom: 30px;
        z-index: 100;
    }
    
    .stChatInput input {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    /* MOVIE CARD STYLING (Flexbox) */
    .movie-card {
        display: flex;
        background-color: #1f1f1f;
        border-radius: 8px;
        margin-bottom: 25px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        border: 1px solid #333;
    }

    .movie-card:hover {
        transform: scale(1.01);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.4); /* Netflix Red Glow */
        border-color: #E50914;
    }

    .movie-poster-container {
        flex: 0 0 200px;
        position: relative;
    }

    .movie-poster {
        width: 100%;
        height: 300px;
        object-fit: cover;
        display: block;
    }

    .movie-details {
        flex: 1;
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .movie-header h2 {
        margin: 0;
        font-size: 1.8rem;
        color: white;
        font-weight: 700;
    }

    .movie-meta {
        color: #a3a3a3;
        font-size: 0.9rem;
        margin-bottom: 15px;
        display: flex;
        gap: 15px;
        align-items: center;
    }

    .meta-badge {
        border: 1px solid #a3a3a3;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
    }

    /* Therapist/Netizen Note Box */
    .netizen-box {
        background: rgba(229, 9, 20, 0.1);
        border-left: 4px solid #E50914;
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 0 4px 4px 0;
    }

    .netizen-label {
        color: #E50914;
        font-weight: bold;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-bottom: 5px;
        display: block;
    }

    .netizen-text {
        font-style: italic;
        color: #e5e5e5;
        font-size: 0.95rem;
        line-height: 1.4;
    }

    /* Buttons */
    .action-row {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }

    .btn-stream {
        background-color: #ffffff;
        color: #000000;
        padding: 8px 20px;
        text-decoration: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
        transition: background 0.2s;
    }

    .btn-stream:hover {
        background-color: #e6e6e6;
        color: #000000;
    }

    .btn-details {
        background-color: rgba(109, 109, 110, 0.7);
        color: white;
        padding: 8px 20px;
        text-decoration: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    
    .btn-details:hover {
        background-color: rgba(109, 109, 110, 0.4);
        color: white;
    }

    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .movie-card {
            flex-direction: column;
        }
        .movie-poster-container {
            flex: 0 0 auto;
        }
        .movie-poster {
            height: 200px;
            width: 100%;
        }
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==========================================
# 2. DATA MODELS (Pydantic)
# ==========================================

class MoodAnalysis(BaseModel):
    detected_moods: List[str] = Field(description="List of detected emotions")
    summary: str = Field(description="Short empathetic summary of user's feeling")
    search_keywords: str = Field(description="A detailed English plot description for Qdrant vector search")


# ==========================================
# 3. CLASS `MovieChatbot` & LOGIC WITH QDRANT
# ==========================================

class MovieChatbot:
    def __init__(self):
        """Initialize MovieChatbot dengan Qdrant dan Google Embeddings"""
        print("\n🤖 MENYIAPKAN CHATBOT...")

        try:
            qdrant_config = {
                'url': st.secrets.get("QDRANT_URL"),
                'api_key': st.secrets.get("QDRANT_API_KEY"),
                'collection': st.secrets.get("QDRANT_COLLECTION", "moodviedb")
            }
        except Exception as e:
            st.error(f"❌ Konfigurasi Qdrant belum lengkap: {e}")
            qdrant_config = None

        # --- TENTUKAN MODE: QDRANT ONLY ---
        self.mode = "QDRANT"
        print("🟠 Mode: QDRANT DATABASE (Vector Search dengan Google Embeddings)")

        if not qdrant_config or not qdrant_config.get('url'):
            raise ValueError("❌ Konfigurasi Qdrant belum lengkap!")

        # --- INISIALISASI GOOGLE EMBEDDINGS (768 DIMENSI) ---
        print("🔌 Menggunakan Google Embeddings (models/text-embedding-004)...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        # --- INISIALISASI QDRANT VECTOR STORE ---
        client = QdrantClient(
            url=qdrant_config['url'],
            api_key=qdrant_config['api_key'] if qdrant_config['api_key'] else None
        )
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=qdrant_config['collection'],
            embedding=self.embeddings
        )

        # --- SETUP GEMINI AI ---
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        self.parser = JsonOutputParser(pydantic_object=MoodAnalysis)

        # --- SETUP PROMPT UNTUK ANALISIS MOOD ---
        self.prompt = PromptTemplate(
            template="""You are a Movie Therapist.
User Input: "{user_input}"
History: "{chat_history}"

Task: Analyze the user's mood and generate search parameters for finding the perfect movie.
IMPORTANT: For 'search_keywords', write a detailed paragraph describing the plot atmosphere and emotional tone to match vector embeddings. Be specific about the emotional journey and themes.

{format_instructions}""",
            input_variables=["user_input", "chat_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser
        self.chat_history = []

    def get_streaming_link(self, movie_title):
        """Tool: Link_Finder (Google Custom Search)"""
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
            cse_id = st.secrets.get("GOOGLE_CSE_ID")
        except:
            return "https://www.google.com/search?q=" + movie_title.replace(" ", "+") + "+streaming"

        if not api_key or not cse_id:
            return "https://www.google.com/search?q=" + movie_title.replace(" ", "+")

        search_query = f'watch "{movie_title}" movie streaming legal'
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cse_id, "q": search_query, "num": 5}

        try:
            res = requests.get(url, params=params).json()
            links = []
            allowed_domains = ["netflix", "disneyplus", "hotstar", "vidio", "primevideo", 
                             "viu", "wetv", "iq", "catchplay", "hbo", "apple"]
            
            if "items" in res:
                for item in res["items"]:
                    link = item.get("link", "").lower()
                    if any(dom in link for dom in allowed_domains) and "/browse" not in link:
                        links.append({"title": item.get("title", "Streaming Link"), "link": item["link"]})
            
            return links[:2] if links else None
        except Exception as e:
            print(f"Search Error: {e}")
            return None

    def generate_kata_netizen(self, title, overview):
        """Feature: Dynamic Summarization in Bahasa Indonesia"""
        prompt_template = PromptTemplate.from_template(
            """Act as a witty Indonesian social media movie reviewer.
Summarize this movie description for '{title}' into a single, punchy paragraph (max 3 sentences).
Use slang (bahasa gaul), be expressive (e.g., "Gila sih", "Wajib tonton"), and convincing.
            
Movie Description: {overview}
            
Output (Bahasa Indonesia only):"""
        )
        chain = prompt_template | self.llm | StrOutputParser()
        try:
            truncated_overview = str(overview)[:1000]
            return chain.invoke({"title": title, "overview": truncated_overview})
        except Exception:
            return "Film ini rame banget dibahas, coba cek sendiri deh!"

    def retrieve_movies(self, query):
        """Retrieve movies dari Qdrant berdasarkan mood query"""
        try:
            docs = self.vector_store.similarity_search(query, k=3)
            results = []
            for doc in docs:
                results.append({
                    "title": doc.metadata.get("title", "Unknown"),
                    "year": str(doc.metadata.get("release_date", ""))[:4],
                    "rating": doc.metadata.get("vote_average", 0),
                    "overview": doc.metadata.get("overview", ""),
                    "poster_url": doc.metadata.get("poster_url", "https://via.placeholder.com/200x300?text=No+Poster"),
                    "genre": doc.metadata.get("genre", ""),
                    "vote_count": doc.metadata.get("vote_count", 0)
                })
            return results
        except Exception as e:
            print(f"Qdrant Retrieval Error: {e}")
            return []

    def process_query(self, user_input):
        """Orchestrator: Analyze mood dan retrieve movies"""
        try:
            # 1. Analyze mood
            hist_str = "\n".join([f"{r}: {m}" for r, m in self.chat_history[-3:]])
            analysis = self.chain.invoke({"user_input": user_input, "chat_history": hist_str})
            
            # Update chat history
            self.chat_history.append(("User", user_input))
            self.chat_history.append(("System", analysis['summary']))
            
            # 2. Retrieve movies
            search_results = self.retrieve_movies(analysis['search_keywords'])
            
            if not search_results:
                return {
                    "mood_summary": analysis['summary'],
                    "detected_moods": analysis['detected_moods'],
                    "movies": [],
                    "error": "Tidak ada film yang cocok ditemukan"
                }
            
            # 3. Process each movie
            processed_movies = []
            for movie in search_results:
                kata_netizen = self.generate_kata_netizen(movie['title'], movie['overview'])
                streaming_links = self.get_streaming_link(movie['title'])
                
                processed_movies.append({
                    "title": movie['title'],
                    "year": movie['year'],
                    "rating": movie['rating'],
                    "overview": movie['overview'],
                    "poster_url": movie['poster_url'],
                    "genre": movie['genre'],
                    "vote_count": movie['vote_count'],
                    "kata_netizen": kata_netizen,
                    "streaming_links": streaming_links
                })
            
            return {
                "mood_summary": analysis['summary'],
                "detected_moods": analysis['detected_moods'],
                "movies": processed_movies,
                "error": None
            }
            
        except Exception as e:
            print(f"Process Query Error: {e}")
            return {
                "mood_summary": "Terjadi kesalahan dalam menganalisis mood Anda.",
                "detected_moods": [],
                "movies": [],
                "error": str(e)
            }

# ==========================================
# 4. STREAMLIT SESSION STATE MANAGEMENT
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I understand life can be overwhelming. Tell me how you're feeling, or what you're doing right now (e.g., 'aku gabut', 'heartbroken', 'need inspiration'), and I'll prescribe the perfect cinema therapy."}
    ]

# Initialize Chatbot Agent
@st.cache_resource
def get_agent():
    return MovieChatbot()

agent = get_agent()

# ==========================================
# 5. MAIN UI LOOP
# ==========================================

# Display Header (Minimalist)
st.markdown("<h1 style='color: #E50914; font-weight: 900; letter-spacing: -2px; margin-bottom: 30px;'>MOVIE <span style='color: white;'>THERAPIST</span></h1>", unsafe_allow_html=True)

# Render Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        # User Bubble
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        # Assistant Bubble
        with st.chat_message("assistant", avatar="🤖"):
            content = message["content"]
            
            # Render based on content type
            if isinstance(content, dict):
                # Rich response with mood analysis and movie cards
                if content.get("error"):
                    st.error(f"❌ {content['error']}")
                else:
                    # Show mood summary
                    st.markdown(f"<p style='font-size: 1.1rem; color: #ccc; margin-bottom: 20px;'>{content['mood_summary']}</p>", unsafe_allow_html=True)
                    
                    # Show detected moods
                    mood_badges = " ".join([f"🎭 {mood}" for mood in content.get('detected_moods', [])])
                    if mood_badges:
                        st.markdown(f"<p style='font-size: 0.9rem; color: #E50914; margin-bottom: 20px;'>{mood_badges}</p>", unsafe_allow_html=True)
                    
                    # Show movie cards
                    for movie in content.get('movies', []):
                        rating_stars = "⭐" * int(movie['rating'] / 2) if movie['rating'] else "N/A"
                        
                        streaming_html = ""
                        if movie.get('streaming_links'):
                            streaming_html = "<div style='margin-top: 10px;'><p style='color: #E50914; font-weight: bold; font-size: 0.9rem;'>🌐 Streaming:</p>"
                            for link in movie['streaming_links']:
                                streaming_html += f"<p style='font-size: 0.85rem; margin: 5px 0;'><a href='{link['link']}' target='_blank' style='color: #4da6ff; text-decoration: none;'>{link['title']}</a></p>"
                            streaming_html += "</div>"
                        
                        card_html = f"""
                        <div class="movie-card">
                            <div class="movie-poster-container">
                                <img src="{movie['poster_url']}" class="movie-poster" alt="{movie['title']}">
                            </div>
                            <div class="movie-details">
                                <div class="movie-header">
                                    <h2>{movie['title']}</h2>
                                    <div class="movie-meta">
                                        <span class="meta-badge">{movie['year']}</span>
                                        <span>{movie['genre']}</span>
                                        <span>{rating_stars}</span>
                                    </div>
                                </div>
                                
                                <div class="netizen-box">
                                    <span class="netizen-label">Kata Netizen</span>
                                    <span class="netizen-text">{movie['kata_netizen']}</span>
                                </div>
                                
                                <p style='color: #a3a3a3; font-size: 0.9rem; margin: 10px 0;'>{movie['overview'][:200]}...</p>
                                
                                <div class="action-row">
                                    <a href="https://www.imdb.com/find?q={movie['title'].replace(' ', '+')}" target="_blank" class="btn-details">
                                        📖 Details
                                    </a>
                                </div>
                                {streaming_html}
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
            else:
                # Simple text response
                st.write(content)

# Chat Input
if prompt := st.chat_input("How are you feeling today?"):
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Menganalisis mood Anda..."):
            # Process via Agent
            response_payload = agent.process_query(prompt)
        
        # Render response
        if response_payload.get("error"):
            st.error(f"❌ {response_payload['error']}")
        else:
            # Show mood summary
            st.markdown(f"<p style='font-size: 1.1rem; color: #ccc; margin-bottom: 20px;'>{response_payload['mood_summary']}</p>", unsafe_allow_html=True)
            
            # Show detected moods
            mood_badges = " ".join([f"🎭 {mood}" for mood in response_payload.get('detected_moods', [])])
            if mood_badges:
                st.markdown(f"<p style='font-size: 0.9rem; color: #E50914; margin-bottom: 20px;'>{mood_badges}</p>", unsafe_allow_html=True)
            
            # Show movie cards
            if response_payload.get('movies'):
                for movie in response_payload['movies']:
                    rating_stars = "⭐" * int(movie['rating'] / 2) if movie['rating'] else "N/A"
                    
                    streaming_html = ""
                    if movie.get('streaming_links'):
                        streaming_html = "<div style='margin-top: 10px;'><p style='color: #E50914; font-weight: bold; font-size: 0.9rem;'>🌐 Streaming:</p>"
                        for link in movie['streaming_links']:
                            streaming_html += f"<p style='font-size: 0.85rem; margin: 5px 0;'><a href='{link['link']}' target='_blank' style='color: #4da6ff; text-decoration: none;'>{link['title']}</a></p>"
                        streaming_html += "</div>"
                    
                    card_html = f"""
                    <div class="movie-card">
                        <div class="movie-poster-container">
                            <img src="{movie['poster_url']}" class="movie-poster" alt="{movie['title']}">
                        </div>
                        <div class="movie-details">
                            <div class="movie-header">
                                <h2>{movie['title']}</h2>
                                <div class="movie-meta">
                                    <span class="meta-badge">{movie['year']}</span>
                                    <span>{movie['genre']}</span>
                                    <span>{rating_stars}</span>
                                </div>
                            </div>
                            
                            <div class="netizen-box">
                                <span class="netizen-label">Kata Netizen</span>
                                <span class="netizen-text">{movie['kata_netizen']}</span>
                            </div>
                            
                            <p style='color: #a3a3a3; font-size: 0.9rem; margin: 10px 0;'>{movie['overview'][:200]}...</p>
                            
                            <div class="action-row">
                                <a href="https://www.imdb.com/find?q={movie['title'].replace(' ', '+')}" target="_blank" class="btn-details">
                                    📖 Details
                                </a>
                            </div>
                            {streaming_html}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.info("❌ Tidak ada film yang cocok ditemukan. Coba deskripsi mood yang berbeda!")
    
    # 3. Save to history
    st.session_state.messages.append({"role": "assistant", "content": response_payload})

# Add a spacer at the bottom so the last card isn't hidden by the input box
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)