import streamlit as st
import time
import textwrap # Import library untuk merapikan string
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

# --- INJECT CSS TAMBAHAN UNTUK CONSISTENCY KARTU NATIVE & DIALOG WIDTH ---
st.markdown("""
<style>
    /* 1. Paksa tinggi minimum kartu native agar sejajar */
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

    /* 2. CUSTOM WIDTH UNTUK DIALOG (1080px) */
    div[data-testid="stDialog"] div[role="dialog"] {
        width: 1080px !important;
        max-width: 95vw !important; /* Agar aman di layar mobile */
    }
</style>
""", unsafe_allow_html=True)


# --- 2. FUNGSI POP-UP DETAIL (st.dialog) ---
@st.dialog("🎬 Detail Film & Analisis", width="large")
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


# --- 4. SESSION STATE & INIT CHECK ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check readiness flag
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# --- 5. MODAL INPUT API KEY (FORCE POP-UP) ---
@st.dialog("🔐 Konfigurasi Wajib", width="small")
def setup_api_keys():
    st.write("Aplikasi ini membutuhkan API Key untuk dapat bekerja.")
    st.warning("Pop-up ini akan terus muncul sampai Anda memasukkan kunci yang valid.")
    
    # Input field di dalam dialog
    gemini_key = st.text_input("Gemini API Key (Wajib)", type="password")
    
    default_tmdb = st.secrets.get("TMDB_API_KEY", "")
    tmdb_key = st.text_input("TMDB API Key (Opsional)", value=default_tmdb, type="password")
    
    if st.button("🚀 Hubungkan & Mulai", use_container_width=True):
        if not gemini_key:
            st.error("Mohon isi Gemini API Key.")
        else:
            with st.spinner("Memverifikasi koneksi..."):
                try:
                    # Cek Secrets Qdrant
                    if "QDRANT_URL" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
                        st.error("Secrets Qdrant tidak ditemukan di .streamlit/secrets.toml")
                        st.stop()

                    # Inisialisasi Config
                    Config.initialize(
                        gemini_key=gemini_key,
                        qdrant_url=st.secrets["QDRANT_URL"],
                        qdrant_key=st.secrets["QDRANT_API_KEY"],
                        tmdb_key=tmdb_key if tmdb_key else None
                    )
                    
                    # Inisialisasi Agent
                    st.session_state.agent = MovieRecommenderAgent(
                        mood_tool=MoodAnalyzerTool(),
                        qdrant_searcher=QdrantSearcher(),
                        review_summarizer=ReviewSummarizer(),
                        streaming_finder=StreamingFinder
                    )
                    
                    # Set status Ready & Rerun
                    st.session_state.is_ready = True
                    st.success("Terhubung!")
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Gagal menghubungkan: {e}")

# TRIGGER POP-UP JIKA BELUM READY
if not st.session_state.is_ready:
    setup_api_keys()


# --- 6. CUSTOM HEADER (HTML) ---
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


# --- 7. HERO SECTION VS CHAT MODE ---
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


# --- 8. SIDEBAR CONFIG (FALLBACK/EDIT) ---
# Sidebar tetap ada untuk mengganti key jika sudah login
if st.session_state.is_ready:
    with st.sidebar:
        st.header("⚙️ Configuration")
        if "QDRANT_URL" not in st.secrets or "QDRANT_API_KEY" not in st.secrets:
            st.error("Missing Qdrant Secrets")
        
        with st.expander("🔐 Update API Keys", expanded=False):
            st.info("Sistem sudah terhubung. Gunakan ini jika ingin mengganti Key.")
            new_gemini_key = st.text_input("New Gemini API Key", type="password")
            if st.button("Update Key"):
                if new_gemini_key:
                    # Re-init logic sederhana
                    try:
                        Config.initialize(
                            gemini_key=new_gemini_key,
                            qdrant_url=st.secrets["QDRANT_URL"],
                            qdrant_key=st.secrets["QDRANT_API_KEY"]
                        )
                        # Re-create agent
                        st.session_state.agent = MovieRecommenderAgent(
                            mood_tool=MoodAnalyzerTool(),
                            qdrant_searcher=QdrantSearcher(),
                            review_summarizer=ReviewSummarizer(),
                            streaming_finder=StreamingFinder
                        )
                        st.success("Key Updated!")
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- 9. CHAT HISTORY (RENDER NATIVE) ---
for message in st.session_state.messages:
    if message["role"] == "user":
        # Pesan User: Tampilkan bubble chat user (sebelumnya pass)
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        # Pesan AI
        with st.chat_message("assistant"):
            if "content" in message:
                # Render HTML/Markdown text (termasuk kata pembuka)
                # PERBAIKAN: textwrap.dedent digunakan untuk menghapus spasi
                st.markdown(textwrap.dedent(message["content"]), unsafe_allow_html=True)
            
            # Render Kartu Native
            if "movies" in message:
                render_movie_grid(message["movies"])

# --- 10. INPUT AREA ---
# Kunci input jika belum ready (meskipun pop-up muncul, ini double safety)
prompt = st.chat_input("How are you feeling today?", disabled=not st.session_state.is_ready)

if prompt:
    if not st.session_state.is_ready:
        st.error("Please configure API Keys first.")
        setup_api_keys() # Trigger pop-up lagi
    else:
        # Simpan pesan user
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Tampilkan bubble chat user
        with st.chat_message("user"):
            st.write(prompt)

        try:
            agent = st.session_state.agent
            
            # --- AGENT LOGIC ---
            with st.chat_message("assistant"):
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
                    # Ambil data mood
                    mood_title = mood_info.get('detected_moods', ['Unknown'])[0].title()
                    mood_summary = mood_info.get('summary', 'Here is a list of movies tailored to your current mood.')
                    
                    # 1. Kata Pembuka (Conversational)
                    opening_text = f"_{mood_summary}_"
                    
                    # 2. Header Resep (HTML)
                    # PERBAIKAN: Menggunakan textwrap.dedent di sini juga
                    header_html = textwrap.dedent(f"""
                    <div style="margin-top: 15px; margin-bottom: 20px; border-left: 3px solid #D32F2F; padding-left: 15px;">
                        <h3 style="margin:0; color:#eee;">Prescription for: <span style="color:#D32F2F;">{mood_title}</span></h3>
                    </div>
                    """)
                    
                    # Gabungkan Konten Teks
                    full_content = f"{opening_text}\n\n{header_html}"
                    st.markdown(full_content, unsafe_allow_html=True)

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

                    # Simpan state (Gabungan teks pembuka + header)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_content,
                        "movies": enriched_movies
                    })

        except Exception as e:
            st.error(f"System Error: {e}")