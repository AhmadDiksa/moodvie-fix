# Example Usage & Testing Guide

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Setup secrets (create .streamlit/secrets.toml)
```

### 2. Konfigurasi Secrets

Create `.streamlit/secrets.toml`:

```toml
# Google AI
GOOGLE_API_KEY = "your-google-api-key"  # Get from: https://makersuite.google.com/app/apikey

# Qdrant
QDRANT_URL = "https://your-qdrant-url.com"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION = "moodviedb"

# Google Custom Search (optional for streaming links)
GOOGLE_CSE_ID = "your-cse-id"
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## Example Interactions

### Example 1: User Feeling Bored

```
User: "aku gabut nih, males kerjain tugas"
↓
Mood Analysis:
  - Detected Moods: ["bored", "procrastinating", "unmotivated"]
  - Summary: "Saya memahami Anda merasa bosan dan tidak bersemangat. Mari cari film yang bisa menghibur dan menginspirasi Anda kembali bekerja!"
  - Search Keywords: "A lighthearted comedy-adventure about unlikely heroes discovering their passion and potential, with humor and inspiring moments"
↓
Qdrant Search Results: Comedy films yang match deskripsi tersebut
↓
Movies Rendered:
  1. Movie Title (Year) ⭐ Rating
     Kata Netizen: "Gila sih film ini, ngakak banget! Cocok buat hilang bosan..."
     [Details] [Streaming Links]
```

### Example 2: User Feeling Heartbroken

```
User: "pacar aku pergi, hati aku hancur"
↓
Mood Analysis:
  - Detected Moods: ["heartbroken", "sad", "lonely"]
  - Summary: "Saya sangat memahami rasa sakit kehilangan yang Anda rasakan..."
  - Search Keywords: "Emotional drama films about love, loss, and healing, with beautiful cinematography and touching moments that validate deep feelings"
↓
Movies: Emotional/Drama films recommended
```

### Example 3: User Feeling Adventurous

```
User: "pengen petualangan seru, pengen yang intense!"
↓
Movies: Action/Adventure films dengan intense plot
```

---

## API Response Structure

### MoodAnalysis Object (JSON Output)

```json
{
  "detected_moods": ["bored", "need-entertainment"],
  "summary": "Saya mengerti Anda membutuhkan hiburan yang menyegarkan...",
  "search_keywords": "A fun comedy film with quick pacing, witty dialogue, and relatable characters that make viewers laugh out loud"
}
```

### Movie Card Data

```python
{
    "title": "Movie Title",
    "year": "2023",
    "rating": 7.5,
    "genre": "Comedy, Adventure",
    "overview": "Movie plot summary...",
    "poster_url": "https://...",
    "vote_count": 1250,
    "kata_netizen": "Gila sih ini film, top banget!",
    "streaming_links": [
        {"title": "Netflix", "link": "https://netflix.com/..."},
        {"title": "Disney+", "link": "https://disneyplus.com/..."}
    ]
}
```

---

## Key Features Explained

### 1. Mood Analysis

- **LLM**: Gemini 2.0 Flash dengan few-shot examples
- **Output**: Structured JSON via Pydantic
- **Accuracy**: Bahasa Indonesia + English hybrid understanding

### 2. Vector Search (Qdrant)

- **Embeddings**: Google Text Embeddings 004 (768D)
- **Algorithm**: Cosine similarity
- **Query**: Detailed plot descriptions (semantic search)
- **Result**: Top-3 most similar movies

### 3. Kata Netizen Generation

- **Input**: Movie title + overview
- **Processing**: LLM generates witty Indonesian review
- **Style**: Casual bahasa gaul, expressive
- **Output**: 2-3 sentences max

### 4. Streaming Link Finder

- **Method**: Google Custom Search API
- **Domains**: Netflix, Disney+, Hotstar, etc.
- **Validation**: Filters out browse-only pages
- **Limit**: Max 2 links per movie

---

## Common User Inputs

```
# Mood-based
"aku sedih banget"
"pengen ketawa"
"butuh inspirasi"
"stress dengan deadline"

# Activity-based
"lagi di rumah sakit, butuh film menghibur"
"pengen nonton bareng keluarga"
"butuh film untuk tidur"

# Emotional
"hati aku sakit"
"merasa sendirian"
"kangen masa lalu"

# Context-based
"aku sedang di mobil, butuh yang seru"
"lagi di kafe, butuh yang santai"
"sebelum tidur, butuh yang menenangkan"
```

---

## Troubleshooting

### Issue: "Tidak ada film yang cocok ditemukan"

**Solution**:

- Check Qdrant collection not empty: `qdrant-client --url <URL> --collection <COLLECTION>`
- Try more specific/longer mood description
- Verify embeddings are generated correctly

### Issue: Streaming links tidak muncul

**Solution**:

- Check GOOGLE_CSE_ID is configured
- Verify Google Custom Search API quota
- Check allowed_domains list in code

### Issue: Gemini rate limit exceeded

**Solution**:

- Implement caching for repeated queries
- Add delay between requests
- Use batch processing

### Issue: Memory issue dengan chat history

**Solution**:

- Implement history truncation (keep last 10 messages)
- Add clear history button
- Implement persistent storage (database)

---

## Performance Metrics

| Operation             | Time     | Notes              |
| --------------------- | -------- | ------------------ |
| Mood Analysis         | 0.5-1s   | LLM inference time |
| Embedding Generation  | 0.1s     | Per query          |
| Qdrant Search         | 0.05s    | Vector similarity  |
| Kata Netizen Gen      | 1-2s     | Per movie          |
| Streaming Link Search | 0.5-1s   | API call           |
| **Total Response**    | **3-5s** | End-to-end         |

---

## Development Tips

### Add debugging output:

```python
import streamlit as st
st.write("Debug:", agent.chat_history)
```

### Monitor Qdrant:

```bash
# Check collection stats
curl -X GET "http://localhost:6333/collections/moodviedb"
```

### Test LLM output:

```python
from langchain.chat_models import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
response = llm.invoke("Test prompt")
print(response.content)
```

---

## Production Checklist

- [ ] Set up proper error logging
- [ ] Implement rate limiting
- [ ] Add user analytics tracking
- [ ] Set up monitoring for API quotas
- [ ] Implement backup Qdrant instance
- [ ] Cache frequently generated embeddings
- [ ] Add health check endpoint
- [ ] Implement user feedback system
- [ ] Document all API keys management
- [ ] Set up automated testing
