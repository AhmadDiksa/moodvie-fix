import streamlit as st
import os
import requests
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
# 2. CLASS `MovieTherapist` & LOGIC
# ==========================================

class MovieTherapist:
    def __init__(self):
        # 1. Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        
        # 2. Initialize LLM (Gemini)
        # Ensure GOOGLE_API_KEY is in env
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.7,
                convert_system_message_to_human=True
            )
        except Exception as e:
            st.error(f"LLM Initialization Error: {e}")

        # 3. Initialize Qdrant
        try:
            self.qdrant = QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"]
            )
            self.collection_name = "moodviedb"
        except Exception as e:
            st.error(f"Qdrant Connection Error: {e}")

    def get_streaming_link(self, movie_title):
        """Tool: Link_Finder (Google Search)"""
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
            cse_id = st.secrets["GOOGLE_CSE_ID"]
        except KeyError:
            return "https://www.google.com/search?q=" + movie_title.replace(" ", "+") + "+streaming"

        search_query = f"watch {movie_title} online streaming"
        url = f"https://www.googleapis.com/customsearch/v1?q={search_query}&key={api_key}&cx={cse_id}&num=1"

        try:
            response = requests.get(url)
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                return data["items"][0]["link"]
        except:
            pass

        # Fallback to generic search
        return "https://www.google.com/search?q=" + movie_title.replace(" ", "+")

    def generate_kata_netizen(self, title, raw_reviews):
        """Feature: Dynamic Summarization in Bahasa Indonesia"""
        prompt = PromptTemplate.from_template(
            """
            Act as a witty Indonesian social media movie reviewer.
            Summarize these raw reviews for the movie '{title}' into a single, punchy paragraph (max 3 sentences).
            Use slang (bahasa gaul), be expressive (e.g., "Gila sih", "Wajib tonton"), and convincing.
            Label it as "KATA NETIZEN".
            
            Raw Reviews: {reviews}
            
            Output (Bahasa Indonesia only):
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            # Truncate reviews to fit context window if necessary
            truncated_reviews = str(raw_reviews)[:2000]
            return chain.invoke({"title": title, "reviews": truncated_reviews})
        except Exception:
            return "Kata Netizen: Film ini rame banget dibahas, coba cek sendiri deh!"

    def retrieve_movies(self, mood_query):
        """Tool: Mood_Retriever (Qdrant)"""
        vector = self.embeddings.embed_query(mood_query)

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=3
        )
        return results.points

    def process_query(self, user_input):
        """Orchestrator"""
        
        # 1. Check intent (Simple heuristic or LLM call - keeping simple for speed here)
        # In a full agent, we'd use an AgentExecutor, but for UI control, we build the response manually.
        
        with st.spinner("Analyzing your mood..."):
            # Retrieve movies
            search_results = self.retrieve_movies(user_input)
            
            if not search_results:
                return [{"type": "text", "content": "I couldn't find any movies matching that specific mood. Try something like 'feeling adventurous' or 'need a good laugh'."}]

            response_data = []
            
            # Therapist Intro
            intro_prompt = PromptTemplate.from_template(
                "You are a movie therapist. The user feels: '{query}'. Write a very short, empathetic introductory sentence (1 sentence) acknowledging their feeling before presenting the movies."
            )
            intro_chain = intro_prompt | self.llm | StrOutputParser()
            intro_text = intro_chain.invoke({"query": user_input})
            
            response_data.append({"type": "text", "content": intro_text})

            # Process Cards
            for hit in search_results:
                payload = hit.payload
                title = payload.get('title', 'Unknown Movie')
                poster = payload.get('poster_url', 'https://via.placeholder.com/200x300?text=No+Poster')
                year = payload.get('year', 'N/A')
                genre = payload.get('genre', 'General')
                raw_reviews = payload.get('raw_reviews', '')
                
                # Parallelizable in production, sequential here for simplicity
                kata_netizen = self.generate_kata_netizen(title, raw_reviews)
                link = self.get_streaming_link(title)
                
                card_html = f"""
                <div class="movie-card">
                    <div class="movie-poster-container">
                        <img src="{poster}" class="movie-poster" alt="{title}">
                    </div>
                    <div class="movie-details">
                        <div class="movie-header">
                            <h2>{title}</h2>
                            <div class="movie-meta">
                                <span class="meta-badge">{year}</span>
                                <span>{genre}</span>
                            </div>
                        </div>
                        
                        <div class="netizen-box">
                            <span class="netizen-label">Kata Netizen</span>
                            <span class="netizen-text">{kata_netizen}</span>
                        </div>
                        
                        <div class="action-row">
                            <a href="{link}" target="_blank" class="btn-stream">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M8 5V19L19 12L8 5Z" fill="black"/>
                                </svg>
                                Play
                            </a>
                            <a href="https://www.imdb.com/find?q={title}" target="_blank" class="btn-details">
                                Details
                            </a>
                        </div>
                    </div>
                </div>
                """
                response_data.append({"type": "html", "content": card_html})
                
            return response_data

# ==========================================
# 3. STREAMLIT SESSION STATE MANAGEMENT
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I understand life can be overwhelming. Tell me how you're feeling, or what you're doing right now (e.g., 'aku gabut', 'heartbroken', 'need inspiration'), and I'll prescribe the perfect cinema therapy."}
    ]

# Initialize Therapist Agent
@st.cache_resource
def get_agent():
    return MovieTherapist()

agent = get_agent()

# ==========================================
# 4. MAIN UI LOOP
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
        # Assistant Bubble (Handle Text vs HTML)
        with st.chat_message("assistant", avatar="🤖"):
            if isinstance(message["content"], list):
                # It's a rich response with cards
                for item in message["content"]:
                    if item["type"] == "text":
                        st.markdown(f"<p style='font-size: 1.1rem; color: #ccc; margin-bottom: 20px;'>{item['content']}</p>", unsafe_allow_html=True)
                    elif item["type"] == "html":
                        st.markdown(item["content"], unsafe_allow_html=True)
            else:
                # Simple text response
                st.write(message["content"])

# Chat Input
if prompt := st.chat_input("How are you feeling today?"):
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Generate response
    with st.chat_message("assistant", avatar="🤖"):
        # Process via Agent
        response_payload = agent.process_query(prompt)
        
        # Render immediately
        for item in response_payload:
            if item["type"] == "text":
                st.markdown(f"<p style='font-size: 1.1rem; color: #ccc; margin-bottom: 20px;'>{item['content']}</p>", unsafe_allow_html=True)
            elif item["type"] == "html":
                st.markdown(item["content"], unsafe_allow_html=True)
    
    # 3. Save to history
    st.session_state.messages.append({"role": "assistant", "content": response_payload})

# Add a spacer at the bottom so the last card isn't hidden by the input box
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)