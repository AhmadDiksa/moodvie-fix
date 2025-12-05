import streamlit as st
import time
from core.backend import Config, MoodAnalyzerTool, QdrantSearcher, ReviewSummarizer, StreamingFinder, MovieRecommenderAgent

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Movie Therapist",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INJECT CSS CUSTOM ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css("assets/style.css")

# --- INJECT CSS TAMBAHAN UNTUK CONSISTENCY KARTU NATIVE ---
st.markdown("""
<style>
    /* Paksa tinggi minimum kartu native agar sejajar */
    [data-testid="stVerticalBlockBorderWrapper"] {
        height: 520px; /* Tinggi FIXED */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 0 !important; /* Hilangkan padding default container */
        overflow: hidden;
    }
    
    /* Area Konten Atas (Poster + Judul) */
    [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        gap: 0px;
    }

    /* Padding manual untuk konten teks di dalam kartu native */
    .native-card-content {
        padding: 15px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }

    /* Styling Gambar di Native Card */
    [data-testid="stVerticalBlockBorderWrapper"] img {
        width: 100%;
        height: 250px; /* Tinggi poster fix */
        object-fit: cover; /* Crop agar rapi */
        border-radius: 0 !important; /* Hilangkan radius default gambar */
    }

    /* Tombol Full Width di Bawah */
    .stButton {
        padding: 10px 15px;
        margin-top: auto;
    }
    .stButton button {
        width: 100%;
        border-radius: 4px;
        font-weight: 600;
        background-color: #222;
        color: #fff;
        border: 1px solid #333;
    }
    .stButton button:hover {
        border-color: #D32F2F;
        color: #D32F2F;
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. FUNGSI POP-UP DETAIL (st.dialog) ---
@st.dialog("🎬 Detail Film & Analisis")
def show_movie_details(movie):
    st.header(movie.get('title', 'Unknown'))
    
    col_img, col_info = st.columns([1.5, 3])
    
    with col_img:
        payload = movie.get('payload', {})
        poster_url = movie.get('poster_url') or payload.get('poster_url')
        img_src = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
        st.image(img_src, use_container_width=True)
        
        score = movie.get('score', 0)
        release_date = movie.get('release_date', '????')
        year = release_date.split('-')[0] if release_date else '????'
        st.caption(f"⭐ **{score:.1f}/10** | 📅 **{year}**")

        trailer_url = payload.get('trailer_url') or payload.get('trailer')
        if trailer_url and trailer_url != '#':
            st.link_button("▶ Tonton Trailer", trailer_url, use_container_width=True)
    
    with col_info:
        genres = movie.get('genres', [])
        st.markdown(f"**Genre:** {', '.join(genres)}")
        st.divider()

        st.markdown("##### 🩺 Therapist's Note")
        overview = movie.get('overview', 'No description available.')
        st.write(overview)
        
        if 'review_snippet' in movie:
            st.markdown("##### 🗣️ Kata Netizen (Gemini Summary)")
            st.info(f"_{movie['review_snippet']}_")
        
        st.markdown("##### 📺 Streaming")
        platform_name = movie.get('platform_name', 'Not Checked')
        if platform_name and platform_name != "Not Available":
            st.success(f"Tersedia di: **{platform_name}**")
        else:
            st.warning("Tidak ditemukan di layanan streaming legal utama region ID.")


# --- 3. FUNGSI RENDER KARTU FILM (NATIVE GRID) ---
def render_movie_grid(movies):
    if not movies:
        return

    # Spacer agar tidak nempel header
    st.write("") 

    cols = st.columns(3)
    
    for idx, movie in enumerate(movies):
        with cols[idx % 3]:
            # Container Native (Styled via CSS di atas)
            with st.container(border=True):
                # A. Poster (Full Width via CSS)
                payload = movie.get('payload', {})
                poster_url = movie.get('poster_url') or payload.get('poster_url')
                img_src = poster_url if poster_url else "https://via.placeholder.com/500x750?text=No+Poster"
                
                # Render Gambar Native
                st.image(img_src, use_container_width=True)
                
                # B. Konten Teks (Judul & Info)
                # Judul dipotong agar tinggi konsisten
                title = movie.get('title', 'Unknown')
                if len(title) > 40:
                    title = title[:37] + "..."
                
                genres = movie.get('genres', [])[:2]
                score = movie.get('score', 0)
                
                # Menggunakan Markdown native untuk isi kartu
                st.markdown(f"##### {title}")
                st.caption(f"⭐ {score:.1f} | 🎭 {', '.join(genres)}")
                
                # Spacer
                st.markdown('<div style="flex-grow:1"></div>', unsafe_allow_html=True)
                
                # C. Tombol Detail Native
                if st.button("📄 LIHAT DETAIL", key=f"btn_{idx}", use_container_width=True):
                    show_movie_details(movie)


# --- 4. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. CUSTOM HEADER (HTML) ---
# Header menggunakan HTML agar persis desain
st.markdown("""
    <div class="custom-header">
        <div class="brand-logo">
            <span>📼</span> MOVIE THERAPIST
        </div>
        <div class="powered-by">
            POWERED BY GEMINI 2.5
        </div>
    </div>
""", unsafe_allow_html=True)


# --- 6. HERO SECTION VS CHAT MODE ---
if not st.session_state.messages:
    # Tampilan Awal: Judul Besar di Tengah (HTML)
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">MOVIE THERAPIST</div>
            <div class="hero-subtitle">
                Tell me how you're feeling, and I'll prescribe the perfect cinema therapy.<br>
                Try: <i>"I need motivation"</i> or <i>"I want a good cry"</i>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Mode Chat: Spacer kecil
    st.write("")


# --- 7. SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    if "QDRANT_URL" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
        st.error("Missing Qdrant Secrets")
        st.stop()
    
    with st.expander("🔐 API Keys Settings", expanded=True):
        gemini_key = st.text_input("Gemini API Key", type="password")
        default_tmdb = st.secrets.get("TMDB_API_KEY", "")
        tmdb_key = st.text_input("TMDB API Key", value=default_tmdb, type="password")

    if st.button("Reconnect System", use_container_width=True):
        if gemini_key:
            try:
                Config.initialize(
                    gemini_key=gemini_key,
                    qdrant_url=st.secrets["QDRANT_URL"],
                    qdrant_key=st.secrets["QDRANT_API_KEY"],
                    tmdb_key=tmdb_key if tmdb_key else None
                )
                st.session_state.agent = MovieRecommenderAgent(
                    mood_tool=MoodAnalyzerTool(),
                    qdrant_searcher=QdrantSearcher(),
                    review_summarizer=ReviewSummarizer(),
                    streaming_finder=StreamingFinder
                )
                st.session_state.is_ready = True
                st.success("Connected!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# --- 8. CHAT HISTORY (RENDER NATIVE) ---
for message in st.session_state.messages:
    if message["role"] == "user":
        # Pesan user simpel
        pass 
    else:
        # Pesan AI
        if "content" in message:
            # Render HTML text untuk judul "Prescription for..."
            st.markdown(message["content"], unsafe_allow_html=True)
        
        # Render Kartu Native
        if "movies" in message:
            render_movie_grid(message["movies"])

# --- 9. INPUT AREA ---
is_ready = st.session_state.get("is_ready", False)

if not is_ready:
    st.markdown("<div style='text-align: center; color: #666; margin-bottom: 20px;'>⚠️ Please configure API Keys in the sidebar to start.</div>", unsafe_allow_html=True)

prompt = st.chat_input("How are you feeling today?")

if prompt:
    if not is_ready:
        st.sidebar.error("Please enter API Key first.")
    else:
        # Simpan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Tampilkan sementara pesan user (native)
        # st.chat_message("user").write(prompt) # Opsional, di desain referensi user chat tidak terlalu menonjol

        try:
            agent = st.session_state.agent
            
            # --- AGENT LOGIC ---
            with st.status("🧠 Analyzing mood...", expanded=True) as status:
                plan = agent.plan(prompt)
                mood_info = plan["mood_analysis"]
                status.update(label="Mood analyzed!", state="complete", expanded=False)
            
            with st.spinner("🔍 Curating movies..."):
                movies = agent.execute(plan, limit_per_genre=3)
            
            # --- RESPONSE PREPARATION ---
            if not movies:
                response_html = "<p style='color:#aaa; text-align:center;'>I couldn't find a specific prescription for that. Could you elaborate?</p>"
                st.markdown(response_html, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_html})
            else:
                mood_title = mood_info.get('detected_moods', ['Unknown'])[0].title()
                
                # HTML HEADER untuk bagian resep
                response_html = f"""
                <div style="margin-top: 20px; margin-bottom: 10px; border-left: 3px solid #D32F2F; padding-left: 15px;">
                    <h3 style="margin:0; color:#eee;">Prescription for: <span style="color:#D32F2F;">{mood_title}</span></h3>
                    <p style="color:#888; font-size: 0.9rem; margin:0;">Based on your input: "<i>{prompt}</i>"</p>
                </div>
                """
                st.markdown(response_html, unsafe_allow_html=True)

                # --- ENRICH DATA ---
                enriched_movies = []
                my_bar = st.progress(0, text="Gathering details...")
                
                for idx, movie in enumerate(movies):
                    if 'payload' in movie:
                        snippet = agent.tools['review_summarizer'].summarize_from_payload(movie['payload'], max_sentences=3)
                        movie['review_snippet'] = snippet
                    
                    movie['platform_name'] = "Not Checked"
                    if Config.TMDB_ENABLED:
                        s_data = agent.tools['streaming_finder'].find_streaming(movie.get('title', ''))
                        platforms = s_data.get("available_on", [])
                        if platforms:
                            movie['platform_name'] = platforms[0].upper()
                        elif s_data.get("rent"):
                            movie['platform_name'] = "Rent/Buy"
                        else:
                            movie['platform_name'] = "Not Available"
                    else:
                            movie['platform_name'] = "Enable TMDB Key"
                    
                    enriched_movies.append(movie)
                    my_bar.progress((idx + 1) / len(movies))
                
                my_bar.empty()

                # --- RENDER NATIVE GRID ---
                render_movie_grid(enriched_movies)

                # Simpan state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_html,
                    "movies": enriched_movies
                })

        except Exception as e:
            st.error(f"System Error: {e}")