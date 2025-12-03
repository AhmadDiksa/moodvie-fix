# 🎬 Movie Therapist Agent - Streamlit Application

A conversational AI agent that analyzes user moods and recommends movies using an advanced agentic architecture with LangChain.

## 🎯 Features

- **Agentic Architecture**: LLM orchestrates strictly defined tools for mood analysis and recommendations
- **Mood-Based Recommendations**: Searches Qdrant vector database for personalized movie suggestions
- **Streaming Link Finder**: Uses Google Custom Search to locate legal streaming platforms
- **Review Summarization**: Automatically summarizes raw reviews into casual Indonesian summaries
- **Modern Chat Interface**: Streamlit chat UI with sidebar API configuration
- **Visual Movie Cards**: Display movie details, posters, and metadata
- **Multi-turn Conversations**: Maintains chat history for contextual conversations

## 🏗️ Architecture

### Agentic Process Flow

```
User Input
  ↓
Mood Analysis (LLM)
  ↓
Tool Call 1: Mood_Movie_Retriever (Qdrant)
  ↓
Review Summarization (LLM)
  ↓
Tool Call 2: Streaming_Link_Finder (Google Custom Search)
  ↓
Final Response with Recommendations & Links
```

### Tools

#### 1. **Mood_Movie_Retriever** (Primary Tool)

- Searches Qdrant vector database using semantic similarity
- Queries based on user's emotional state or plot keywords
- Returns: movie title, plot, poster URL, reviews, genre, rating

#### 2. **Streaming_Link_Finder** (Secondary Tool)

- Uses Google Custom Search API
- Searches for legal streaming links (Netflix, Disney+, Prime Video, etc.)
- Returns: search results with links and availability info

## 📋 Requirements

### Tech Stack

- **Framework**: Streamlit
- **Language**: Python 3.10+
- **LLM Framework**: LangChain
- **Vector Database**: Qdrant (Cloud or Local)
- **Embeddings**: HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **LLM Model**: ChatGoogleGenerativeAI (gemini-2.0-flash)
- **APIs**: Google Custom Search JSON API

### Configuration

- Qdrant Collection: `moodviedb`
- Embedding Vector Size: 384
- Similarity Search K: 3

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd moodvie-fix
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with:

- **GOOGLE_API_KEY**: Get from https://aistudio.google.com/app/apikeys
- **GOOGLE_SEARCH_ENGINE_ID**: Create at https://programmablesearchengine.google.com/
- **GOOGLE_SEARCH_API_KEY**: Get from Google Cloud Console
- **QDRANT_URL**: Local (http://localhost:6333) or Cloud instance
- **QDRANT_API_KEY**: Your Qdrant API key (if required)

### 5. Start Qdrant (if using local instance)

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or use Qdrant Cloud: https://cloud.qdrant.io/
```

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📝 How to Use

1. **Configure API Keys**: Fill in all required API keys in the sidebar
2. **Share Your Mood**: Tell the agent about your emotional state or mood
3. **Get Recommendations**: The agent searches Qdrant for matching movies
4. **Review Summaries**: Get casual Indonesian summaries of movie reviews
5. **Find Streaming**: Ask where to watch specific movies
6. **Continue Conversation**: Have multi-turn conversations with context

### Example Interactions

**User**: "Aku sedang sedih hari ini, butuh film yang bisa bikin hati terasa lebih baik"

**Agent**:

1. Analyzes your sad mood
2. Uses Mood_Movie_Retriever tool to find uplifting movies
3. Summarizes reviews
4. Provides movie recommendations with details

**User**: "Bisa bantu cari di mana bisa nonton film itu?"

**Agent**: Uses Streaming_Link_Finder tool to locate streaming platforms

## 🔧 Project Structure

```
moodvie-fix/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example          # API configuration template
├── README.md             # This file
└── .env                  # Actual API keys (create from .env.example)
```

## 🛠️ Code Structure

### MovieTherapistAgent Class

```python
class MovieTherapistAgent:
    def __init__(...)                    # Initialize agent with API clients

    # TOOL 1: MOOD_MOVIE_RETRIEVER
    @tool
    def get_recommendations_tool(...)    # Search Qdrant for movie recommendations

    # TOOL 2: STREAMING_LINK_FINDER
    @tool
    def get_streaming_links_tool(...)    # Find streaming links via Google Search

    def _setup_agent(...)               # Configure LangChain agent
    def run_chat_turn(...)              # Execute one conversation turn
```

### Key Features

- ✅ Clear tool labeling with `# TOOL:` comments
- ✅ Spinners showing which tool is working
- ✅ Error handling with user-friendly messages
- ✅ Responsive chat interface
- ✅ Multi-turn conversation history
- ✅ Indonesian language support

## 📊 Qdrant Data Format

Expected collection structure in Qdrant:

```json
{
  "id": "unique_id",
  "vector": [embedding_array],
  "payload": {
    "title": "Movie Title",
    "plot": "Movie plot description",
    "genre": "Drama, Comedy, etc.",
    "rating": 8.5,
    "poster_url": "https://...",
    "raw_reviews": "Review text from users..."
  }
}
```

## 🔐 Security Notes

- Never commit `.env` file with real API keys
- Use `.env.example` as template for others
- Rotate API keys regularly
- Use Qdrant Cloud for production (better security)
- Validate all user inputs

## 🐛 Troubleshooting

### "Qdrant connection failed"

- Check if Qdrant is running (local) or URL is correct (cloud)
- Verify QDRANT_URL and QDRANT_API_KEY are correct

### "API key not found"

- Ensure `.env` file exists in project root
- Check keys are correctly formatted
- Verify all required keys are present

### "Collection not found"

- Create collection in Qdrant or update QDRANT_COLLECTION name
- Ensure collection has proper vector size (384 for all-MiniLM-L6-v2)

### Agent not responding

- Check all API keys in sidebar
- Verify internet connection
- Check Streamlit logs for detailed errors

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Google Generative AI API](https://ai.google.dev/)
- [Google Custom Search API](https://developers.google.com/custom-search)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature suggestions.

---

**Made with ❤️ for Movie Therapy**
