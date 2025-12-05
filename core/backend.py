import logging
import time
import re
import json
import requests
import streamlit as st  # Ditambahkan untuk menggantikan print()
from typing import Any, Dict, List, Optional, Sequence
from functools import lru_cache
from collections import defaultdict

# Eksternal dependencies
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("google.generativeai not available")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchAny
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("qdrant_client not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------
# Configuration
# -------------------------
class Config:
    """Centralized configuration dengan validasi"""

    # Gemini API (Required untuk mood analysis)
    GEMINI_API_KEY: Optional[str] = None

    # Qdrant (Primary source - Required)
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION: str = "moodviedb"

    # TMDB (Optional - only if user provides)
    TMDB_API_KEY: Optional[str] = None
    TMDB_ENABLED: bool = False

    @classmethod
    def initialize(cls, gemini_key: str, qdrant_url: str, qdrant_key: str,
                   tmdb_key: Optional[str] = None):
        """Initialize configuration"""
        cls.GEMINI_API_KEY = gemini_key
        cls.QDRANT_URL = qdrant_url
        cls.QDRANT_API_KEY = qdrant_key

        if tmdb_key:
            cls.TMDB_API_KEY = tmdb_key
            cls.TMDB_ENABLED = True
            logger.info("TMDB API enabled (optional)")
        else:
            logger.info("TMDB API not provided - using Qdrant only mode")

        # Initialize Gemini
        if GENAI_AVAILABLE and cls.GEMINI_API_KEY:
            genai.configure(api_key=cls.GEMINI_API_KEY)

    @classmethod
    def validate(cls) -> bool:
        """Validate required configurations"""
        if not cls.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is required")
            return False
        if not cls.QDRANT_URL or not cls.QDRANT_API_KEY:
            logger.error("Qdrant configuration is required")
            return False
        return True


# -------------------------
# Utility Functions
# -------------------------
def clean_json_blocks(text: str) -> str:
    """Remove markdown JSON code blocks"""
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def safe_json_loads(text: str) -> Optional[Dict]:
    """Safely parse JSON with fallback"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return None


# -------------------------
# Genre Mapping (Shared)
# -------------------------
class GenreMapper:
    """Centralized genre mapping untuk konsistensi"""

    GENRE_MAP = {
        "action": 28,
        "adventure": 12,
        "animation": 16,
        "comedy": 35,
        "crime": 80,
        "documentary": 99,
        "drama": 18,
        "family": 10751,
        "fantasy": 14,
        "history": 36,
        "horror": 27,
        "music": 10402,
        "mystery": 9648,
        "romance": 10749,
        "science fiction": 878,
        "sci-fi": 878,
        "thriller": 53,
        "war": 10752,
        "western": 37
    }

    @classmethod
    def get_id(cls, genre_name: str) -> Optional[int]:
        """Get genre ID from name"""
        return cls.GENRE_MAP.get(genre_name.lower())

    @classmethod
    def get_ids(cls, genre_names: List[str]) -> List[int]:
        """Get multiple genre IDs"""
        return [cls.get_id(g) for g in genre_names if cls.get_id(g) is not None]

    @classmethod
    def get_name(cls, genre_id: int) -> Optional[str]:
        """Get genre name from ID (reverse mapping)"""
        for name, id_ in cls.GENRE_MAP.items():
            if id_ == genre_id:
                return name.title()
        return None

    @classmethod
    def get_names(cls, genre_ids: List[int]) -> List[str]:
        """Get multiple genre names from IDs"""
        names = []
        for gid in genre_ids:
            name = cls.get_name(gid)
            if name and name not in names:  # Avoid duplicates (sci-fi/science fiction)
                names.append(name)
        return names


# -------------------------
# 1. Mood Analyzer Tool (Enhanced)
# -------------------------
class MoodAnalyzerTool:
    """Enhanced mood analyzer dengan better prompt engineering"""

    PROMPT_TEMPLATE = """Analyze the user's emotional state and provide movie recommendations.

User input: "{text}"

Output ONLY valid JSON (no markdown, no extra text) with this EXACT structure:
{{
    "detected_moods": ["mood1", "mood2"],
    "intensity_score": 50,
    "thematic_keywords": ["keyword1", "keyword2", "keyword3"],
    "emotion_type": "neutral",
    "summary": "Brief empathetic summary in Indonesian",
    "recommended_feel_good_genres": ["Comedy", "Animation"]
}}

Rules:
1. intensity_score: integer 0-100 (0=very low, 100=very high)
2. emotion_type: MUST be one of: "positive", "neutral", "negative"
3. If emotion_type is "negative": recommended_feel_good_genres MUST include ["Comedy", "Animation", "Family"]
4. detected_moods: 1-3 specific emotions (e.g., "happy", "sad", "anxious", "excited")
5. thematic_keywords: relevant themes for movie search
6. summary: 1-2 empathetic sentences in Indonesian
7. recommended_feel_good_genres: 2-4 genre names matching GENRE_MAP

Output ONLY the JSON object, nothing else."""

    def __init__(self, model_name: str = "gemini-flash-latest"):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google.generativeai not available")

        self.model_name = model_name
        try:
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"MoodAnalyzer initialized with {model_name}")
        except Exception as e:
            logger.exception(f"Failed to initialize Gemini model: {e}")
            raise

    def analyze(self, text: str, retries: int = 3) -> Dict[str, Any]:
        """Analyze user mood with retry logic"""
        prompt = self.PROMPT_TEMPLATE.format(text=text.replace('"', '\\"'))

        for attempt in range(1, retries + 1):
            try:
                gen_cfg = {
                    "temperature": 0.1,
                    "max_output_tokens": 1024,
                    "top_p": 0.95
                }

                resp = self._model.generate_content(prompt, generation_config=gen_cfg)
                raw = clean_json_blocks(resp.text.strip())

                # Extract JSON
                json_str = self._extract_json(raw)
                parsed = safe_json_loads(json_str)

                if parsed and self._validate_response(parsed):
                    # Sanitize and return
                    return self._sanitize_response(parsed)
                else:
                    logger.warning(f"Invalid mood analysis response (attempt {attempt})")

            except Exception as e:
                logger.warning(f"MoodAnalyzer attempt {attempt} failed: {e}")

            if attempt < retries:
                time.sleep(0.5 * attempt)

        # Ultimate fallback
        logger.error("All mood analysis attempts failed, using fallback")
        return self._get_fallback_response()

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text"""
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def _validate_response(self, parsed: Dict) -> bool:
        """Validate required fields"""
        required = ["detected_moods", "intensity_score", "emotion_type",
                   "thematic_keywords", "summary", "recommended_feel_good_genres"]
        return all(k in parsed for k in required)

    def _sanitize_response(self, parsed: Dict) -> Dict[str, Any]:
        """Sanitize and normalize response"""
        try:
            parsed["intensity_score"] = max(0, min(100, int(parsed["intensity_score"])))
        except:
            parsed["intensity_score"] = 50

        if parsed["emotion_type"] not in ["positive", "neutral", "negative"]:
            parsed["emotion_type"] = "neutral"

        # Ensure feel-good genres for negative emotions
        if parsed["emotion_type"] == "negative":
            base_genres = ["Comedy", "Animation", "Family"]
            current = parsed.get("recommended_feel_good_genres", [])
            parsed["recommended_feel_good_genres"] = list(set(base_genres + current))

        return parsed

    def _get_fallback_response(self) -> Dict[str, Any]:
        """Fallback response if all attempts fail"""
        return {
            "detected_moods": ["neutral"],
            "intensity_score": 50,
            "thematic_keywords": ["movie", "entertainment"],
            "emotion_type": "neutral",
            "summary": "Saya siap membantu merekomendasikan film untuk Anda.",
            "recommended_feel_good_genres": ["Comedy", "Animation", "Family"]
        }


# -------------------------
# 2. Qdrant Searcher (Primary - Enhanced)
# -------------------------
class QdrantSearcher:
    """Enhanced Qdrant searcher dengan better error handling"""

    def __init__(self):
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant client not available")
            self.available = False
            return

        try:
            self.client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY,
                timeout=10
            )
            # Test connection
            self.client.get_collection(Config.COLLECTION)
            self.available = True
            logger.info("Qdrant searcher initialized successfully")
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            self.available = False

    def search_by_genre(self, genres: List[str], limit: int = 10,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """Search movies by genres using scroll with filter"""
        if not self.available:
            logger.warning("Qdrant not available")
            return []

        try:
            # Convert genre names to IDs
            genre_ids = GenreMapper.get_ids(genres)

            if not genre_ids:
                logger.warning(f"No valid genre IDs for: {genres}")
                return []

            # Build filter
            query_filter = Filter(
                should=[
                    FieldCondition(
                        key="genre_ids",
                        match=MatchAny(any=genre_ids)
                    )
                ]
            )

            # Execute scroll search
            points, _ = self.client.scroll(
                collection_name=Config.COLLECTION,
                scroll_filter=query_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            # Format results
            results = []
            for point in points:
                payload = point.payload or {}
                # Extract genres dari payload, atau fallback ke genre_ids reverse mapping
                genres = payload.get("genres", [])
                if not genres:
                    genre_ids = payload.get("genre_ids", [])
                    genres = GenreMapper.get_names(genre_ids)
                
                results.append({
                    "id": point.id,
                    "title": payload.get("title", "Unknown"),
                    "overview": payload.get("overview", "No description available"),
                    "genres": genres,
                    "genre_ids": payload.get("genre_ids", []),
                    "score": payload.get("vote_average", 0),
                    "release_date": payload.get("release_date", "Unknown"),
                    "poster_url": payload.get("poster_url"),
                    "payload": payload
                })

            logger.info(f"Found {len(results)} movies for genres: {genres}")
            return results

        except Exception as e:
            logger.exception(f"Qdrant search failed: {e}")
            return []

    def search_semantic(self, query: str, limit: int = 10,
                       genre_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Semantic search (requires embedder - fallback to genre search)"""
        if not self.available:
            return []

        # For now, fallback to genre-based search
        # TODO: Implement semantic search with embeddings
        if genre_filter:
            return self.search_by_genre(genre_filter, limit=limit)

        # Try title/overview substring match
        return self._title_fallback_search(query, limit)

    def _title_fallback_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fallback: search by title substring"""
        try:
            lower_query = query.lower()
            found = []

            points, _ = self.client.scroll(
                collection_name=Config.COLLECTION,
                limit=min(limit * 10, 100),  # Get more to filter
                with_payload=True,
                with_vectors=False
            )

            for point in points:
                payload = point.payload or {}
                title = (payload.get("title") or "").lower()
                overview = (payload.get("overview") or "").lower()

                if lower_query in title or lower_query in overview:
                    found.append({
                        "id": point.id,
                        "title": payload.get("title", "Unknown"),
                        "overview": payload.get("overview", ""),
                        "genres": payload.get("genres", []),
                        "score": payload.get("vote_average", 0),
                        "payload": payload
                    })

                    if len(found) >= limit:
                        break

            return found[:limit]

        except Exception as e:
            logger.exception(f"Title fallback search failed: {e}")
            return []


# -------------------------
# 3. TMDB Fetcher (Optional)
# -------------------------
class TMDBFetcher:
    """Optional TMDB integration for additional data"""

    BASE_URL = "https://api.themoviedb.org/3"

    @staticmethod
    def is_enabled() -> bool:
        return Config.TMDB_ENABLED

    @staticmethod
    def fetch_by_genre(genre_id: int, page: int = 1) -> Dict[str, Any]:
        """Fetch movies by genre from TMDB"""
        if not TMDBFetcher.is_enabled():
            return {"error": "TMDB not enabled", "results": []}

        url = f"{TMDBFetcher.BASE_URL}/discover/movie"
        params = {
            "api_key": Config.TMDB_API_KEY,
            "with_genres": genre_id,
            "sort_by": "popularity.desc",
            "language": "id-ID",
            "page": page
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"TMDB fetch error: {e}")
            return {"error": str(e), "results": []}

    @staticmethod
    def search_movie(title: str) -> Optional[int]:
        """Search movie ID by title"""
        if not TMDBFetcher.is_enabled():
            return None

        url = f"{TMDBFetcher.BASE_URL}/search/movie"
        params = {
            "api_key": Config.TMDB_API_KEY,
            "query": title,
            "language": "id-ID"
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("results"):
                return data["results"][0]["id"]
            return None
        except Exception as e:
            logger.error(f"TMDB search error: {e}")
            return None


# -------------------------
# 4. Streaming Finder (Optional - requires TMDB)
# -------------------------
class StreamingFinder:
    """Find streaming platforms using TMDB Watch Providers API"""

    @staticmethod
    def find_streaming(title: str, region: str = "ID") -> Dict[str, Any]:
        """Find streaming platforms for a movie"""
        if not TMDBFetcher.is_enabled():
            return {
                "title": title,
                "available_on": [],
                "note": "TMDB API tidak tersedia. Aktifkan untuk info streaming."
            }

        movie_id = TMDBFetcher.search_movie(title)

        if not movie_id:
            return {
                "title": title,
                "available_on": [],
                "error": "Film tidak ditemukan di TMDB"
            }

        url = f"{TMDBFetcher.BASE_URL}/movie/{movie_id}/watch/providers"
        params = {"api_key": Config.TMDB_API_KEY}

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", {})
            region_data = results.get(region)

            if not region_data:
                return {
                    "title": title,
                    "available_on": [],
                    "note": f"Data streaming tidak tersedia untuk region {region}"
                }

            flatrate = region_data.get("flatrate", [])
            rent = region_data.get("rent", [])
            buy = region_data.get("buy", [])

            return {
                "title": title,
                "available_on": list({p["provider_name"] for p in flatrate}),
                "rent": list({p["provider_name"] for p in rent}),
                "buy": list({p["provider_name"] for p in buy})
            }

        except Exception as e:
            logger.error(f"Streaming finder error: {e}")
            return {
                "title": title,
                "error": f"Gagal mengambil data streaming: {e}"
            }


# -------------------------
# 5. Review Summarizer (Enhanced)
# -------------------------
class ReviewSummarizer:
    """Summarize movie reviews using LLM"""

    def __init__(self, model_name: str = "gemini-flash-latest"):
        if not GENAI_AVAILABLE:
            logger.warning("Gemini not available for review summarization")
            self._model = None
            return

        try:
            self._model = genai.GenerativeModel(model_name)
            logger.info("ReviewSummarizer initialized")
        except Exception as e:
            logger.exception(f"Failed to initialize ReviewSummarizer: {e}")
            self._model = None

    def summarize_from_payload(self, movie_payload: Dict, max_sentences: int = 2) -> str:
        """Summarize reviews from Qdrant payload"""
        if not movie_payload or not self._model:
            return "Ringkasan review tidak tersedia."

        # Extract reviews
        raw_reviews = movie_payload.get("raw_reviews", [])
        if isinstance(raw_reviews, str):
            raw_reviews = [raw_reviews] if raw_reviews else []
        elif not isinstance(raw_reviews, list):
            raw_reviews = []

        if not raw_reviews:
            # Fallback to overview if no reviews
            overview = movie_payload.get("overview", "")
            if overview:
                return overview[:200] + "..." if len(overview) > 200 else overview
            return "Tidak ada review tersedia."

        # Combine reviews
        snippet = " ".join(str(r).strip() for r in raw_reviews if r)[:2000]

        prompt = f"""Ringkas review film berikut dalam {max_sentences} kalimat yang jelas dan objektif dalam bahasa Indonesia.
Fokus pada poin-poin utama tanpa menambahkan opini baru.

Review:
\"\"\"
{snippet}
\"\"\"

Ringkasan ({max_sentences} kalimat):"""

        try:
            gen_cfg = {
                "temperature": 0.1,
                "max_output_tokens": 150,
                "top_p": 0.9
            }

            resp = self._model.generate_content(prompt, generation_config=gen_cfg)
            summary = clean_json_blocks(resp.text.strip())

            # Limit to max_sentences
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            return " ".join(sentences[:max_sentences])

        except Exception as e:
            logger.exception(f"Review summarization failed: {e}")
            # Fallback to first review snippet
            first_review = str(raw_reviews[0]).strip()
            return (first_review[:200] + "...") if len(first_review) > 200 else first_review


# -------------------------
# 6. Movie Recommender Agent (Agentic System)
# -------------------------
class MovieRecommenderAgent:
    """Main agentic system orchestrating all tools"""

    def __init__(self, mood_tool: MoodAnalyzerTool,
                 qdrant_searcher: QdrantSearcher,
                 review_summarizer: ReviewSummarizer,
                 streaming_finder: StreamingFinder):
        self.tools = {
            "mood_tool": mood_tool,
            "qdrant_searcher": qdrant_searcher,
            "review_summarizer": review_summarizer,
            "streaming_finder": streaming_finder
        }

        # Session memory: {genre: set(titles)}
        self.session_memory: Dict[str, set] = defaultdict(set)
        self.search_offset: Dict[str, int] = defaultdict(int)
        
        # Conversation context
        self.conversation_history: List[Dict[str, str]] = []
        self.previous_moods: List[str] = []

    def plan(self, user_text: str) -> Dict[str, Any]:
        """Planner: Analyze mood and determine strategy"""
        logger.info("Planning phase started")

        mood_tool = self.tools["mood_tool"]
        mood_analysis = mood_tool.analyze(user_text)

        target_genres = mood_analysis.get("recommended_feel_good_genres", ["Comedy"])
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        # Track moods for context
        moods = mood_analysis.get('detected_moods', [])
        self.previous_moods.extend(moods)
        self.previous_moods = self.previous_moods[-5:]  # Keep last 5

        plan = {
            "mood_analysis": mood_analysis,
            "target_genres": target_genres,
            "search_strategy": "qdrant_primary"  # Always use Qdrant first
        }

        logger.info(f"Plan: {len(target_genres)} genres identified")
        return plan

    def execute(self, plan: Dict[str, Any], limit_per_genre: int = 3) -> List[Dict[str, Any]]:
        """Executor: Fetch movies based on plan"""
        logger.info("Execution phase started")

        qdrant_searcher = self.tools["qdrant_searcher"]
        target_genres = plan["target_genres"]
        recommended_movies = []

        for genre in target_genres:
            genre_key = genre.lower()

            # Get current offset for this genre
            offset = self.search_offset[genre_key]

            # Search in Qdrant
            movies = qdrant_searcher.search_by_genre(
                [genre],
                limit=limit_per_genre * 2,  # Get more to filter duplicates
                offset=offset
            )

            # Filter out already recommended
            new_movies = []
            for movie in movies:
                title = movie["title"]
                if title not in self.session_memory[genre_key]:
                    self.session_memory[genre_key].add(title)
                    new_movies.append(movie)

                    if len(new_movies) >= limit_per_genre:
                        break

            recommended_movies.extend(new_movies)

            # Update offset
            self.search_offset[genre_key] += len(movies)

            logger.info(f"Found {len(new_movies)} new movies for genre: {genre}")

        return recommended_movies

    def generate_follow_up(self, context: str, previous_moods: List[str]) -> str:
        """Generate contextual follow-up message"""
        if not GENAI_AVAILABLE:
            return "Apakah rekomendasi ini membantu? Ceritakan lebih lanjut tentang preferensi Anda."
        
        try:
            mood_str = ", ".join(previous_moods)
            prompt = f"""Berdasarkan mood sebelumnya ({mood_str}), buat pertanyaan follow-up yang singkat dan menarik (1 kalimat) 
untuk membantu pengguna lebih memperjelas preferensi film mereka. 
Pertanyaan harus terasa personal dan helpful, bukan generic.

Context: {context}

Pertanyaan follow-up:"""
            
            model = genai.GenerativeModel("gemini-flash-latest")
            gen_cfg = {"temperature": 0.7, "max_output_tokens": 100}
            resp = model.generate_content(prompt, generation_config=gen_cfg)
            
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            return "Apakah ada preferensi khusus film yang ingin Anda eksplorasi?"

    def get_similar_movies(self, movie: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        """Get movies similar to the given movie"""
        logger.info(f"Finding similar movies to: {movie.get('title')}")
        
        qdrant_searcher = self.tools["qdrant_searcher"]
        similar = []
        
        # Use first genre as seed
        genres = movie.get("genres", [])
        if genres:
            similar = qdrant_searcher.search_by_genre(
                [genres[0]],
                limit=limit * 3,
                offset=self.search_offset[genres[0].lower()]
            )
            
            # Filter out the original movie
            similar = [m for m in similar if m["title"] != movie.get("title")][:limit]
            
            # Update offset
            self.search_offset[genres[0].lower()] += limit * 3
        
        return similar

    def get_contrasting_movies(self, plan: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        """Get contrasting movies based on mood"""
        logger.info("Finding contrasting movies")
        
        qdrant_searcher = self.tools["qdrant_searcher"]
        mood_info = plan["mood_analysis"]
        emotion_type = mood_info.get("emotion_type", "neutral")
        
        # Map emotion to opposite genres
        opposite_map = {
            "negative": ["Comedy", "Animation", "Family"],
            "positive": ["Drama", "Thriller", "Mystery"],
            "neutral": ["Action", "Adventure", "Fantasy"]
        }
        
        contrast_genres = opposite_map.get(emotion_type, ["Comedy"])
        contrasting = []
        
        for genre in contrast_genres:
            movies = qdrant_searcher.search_by_genre(
                [genre],
                limit=2,
                offset=self.search_offset[genre.lower()]
            )
            contrasting.extend(movies)
            self.search_offset[genre.lower()] += 2
        
        return contrasting[:limit]

    def finalize(self, movies: List[Dict[str, Any]], show_streaming: bool = True) -> None:
        """
        Finalizer: Enrich and display results.
        NOTE: Diganti menggunakan st.write sesuai permintaan
        untuk kompatibilitas tampilan Streamlit jika dipanggil manual.
        """
        logger.info("Finalization phase started")

        if not movies:
            st.warning("❌ Maaf, tidak ada rekomendasi film yang cocok untuk mood Anda saat ini.")
            st.info("Coba ceritakan perasaan Anda dengan cara yang berbeda.")
            return

        review_summarizer = self.tools["review_summarizer"]
        streaming_finder = self.tools["streaming_finder"]

        st.markdown(f"### 🎬 REKOMENDASI FILM UNTUK ANDA ({len(movies)} film)")
        st.write("---")

        for idx, movie in enumerate(movies, 1):
            title = movie.get("title", "Tidak tersedia")
            overview = movie.get("overview", "Tidak ada deskripsi")
            score = movie.get("score", "N/A")
            genres = ", ".join(movie.get("genres", []))
            release = movie.get("release_date", "Unknown")

            # Menggunakan expander untuk menggantikan output terminal beruntun
            with st.expander(f"{idx}. {title} (⭐ {score})", expanded=True):
                st.write(f"**Genre:** {genres}")
                st.write(f"**Rilis:** {release}")
                st.write(f"**Sinopsis:** {overview[:300]}{'...' if len(overview) > 300 else ''}")

                # Review summary
                review = review_summarizer.summarize_from_payload(movie.get("payload", {}))
                st.info(f"📰 **Review:** {review}")

                # Trailer
                trailer = movie.get("payload", {}).get("trailer")
                if trailer:
                    st.write(f"🎥 **Trailer:** {trailer}")

                # Streaming info (optional)
                if show_streaming and TMDBFetcher.is_enabled():
                    try:
                        streaming_info = streaming_finder.find_streaming(title)
                        platforms = streaming_info.get("available_on", [])
                        if platforms:
                            st.success(f"📺 **Streaming:** {', '.join(platforms)}")
                        else:
                            note = streaming_info.get("note", "")
                            if note:
                                st.write(f"📺 **Streaming:** {note}")
                    except Exception as e:
                        logger.debug(f"Streaming lookup failed for {title}: {e}")