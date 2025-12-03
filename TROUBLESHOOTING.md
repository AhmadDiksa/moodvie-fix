# 🔧 TROUBLESHOOTING GUIDE

## Installation & Setup Issues

### ❌ `ModuleNotFoundError: No module named 'langchain_qdrant'`

```bash
# Solution: Install missing dependency
pip install langchain-qdrant>=0.1.0
```

### ❌ `ModuleNotFoundError: No module named 'google.generativeai'`

```bash
# Solution: Install Google Generative AI
pip install google-generativeai>=0.3.0
```

### ❌ Requirements.txt installation fails

```bash
# Solution: Update pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration Issues

### ❌ `st.error: Konfigurasi Qdrant belum lengkap!`

**Cause**: Missing Qdrant configuration in `.streamlit/secrets.toml`

**Solution**:

```toml
# .streamlit/secrets.toml
QDRANT_URL = "https://..."
QDRANT_API_KEY = "..."
QDRANT_COLLECTION = "moodviedb"
```

Then restart Streamlit.

### ❌ `ValueError: Konfigurasi Qdrant belum lengkap!`

**Cause**: `QDRANT_URL` is None or empty

**Check**:

- Is `.streamlit/secrets.toml` in correct location?
- Is file format valid TOML?
- Is `QDRANT_URL` actually set?

```bash
# Debug: Print secrets (DO NOT COMMIT THIS!)
# Add to app.py temporarily:
import streamlit as st
st.write(st.secrets)
```

### ❌ `GOOGLE_API_KEY not found`

**Cause**: Missing Google API key in secrets

**Solution**:

```toml
GOOGLE_API_KEY = "AIzaSy..."  # From https://makersuite.google.com/app/apikey
```

---

## Runtime Issues

### ❌ `Tidak ada film yang cocok ditemukan`

**Possible Causes**:

1. Qdrant collection is empty
2. Embeddings don't match movie descriptions
3. Query is too specific/unusual

**Debug Steps**:

```bash
# Check Qdrant collection status
curl -X GET "https://your-qdrant-url/collections/moodviedb"

# Should return collection stats with point count > 0
```

**Solutions**:

- Use broader mood descriptions: "sedih" instead of "sangat sedih sekali"
- Verify collection has data: `collection.count == <some_number>`
- Check embedding dimensions match (should be 768 for Google Embeddings)

### ❌ LLM Response is generic/unhelpful

**Cause**: Gemini model limitations or temperature too high

**Solutions**:

Option 1: Adjust temperature (already set to 0.7):

```python
# In MovieChatbot.__init__:
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5  # Lower = more deterministic
)
```

Option 2: Improve prompt:

```python
# Better mood description in prompt
"User feels: " + user_input + " (be empathetic and specific)"
```

### ❌ Streaming links not appearing

**Possible Causes**:

1. GOOGLE_CSE_ID not configured
2. Google Custom Search API quota exceeded
3. Movie title has special characters

**Solutions**:

1. Check if Custom Search is enabled:

```bash
# Test API call manually
curl "https://www.googleapis.com/customsearch/v1?q=test&key=YOUR_KEY&cx=YOUR_CSE_ID"
```

2. Add debug logging:

```python
# In get_streaming_link():
print(f"DEBUG: Searching for {movie_title}")
print(f"DEBUG: Found {len(links)} links")
```

3. Verify API quota in Google Cloud Console

---

## Performance Issues

### ⏱️ Response takes >10 seconds

**Bottlenecks**:

1. Gemini LLM inference (1-2s)
2. Qdrant search (0.1-0.5s)
3. Kata Netizen generation (1-2s per movie, 3-6s total)
4. Google Custom Search (0.5-1s per movie, 1.5-3s total)

**Optimization**:

```python
# Option 1: Reduce k value in retrieve_movies
docs = self.vector_store.similarity_search(query, k=2)  # Was k=3

# Option 2: Skip Kata Netizen for cached requests
# (Implement caching for repeated movies)

# Option 3: Parallelize API calls
from concurrent.futures import ThreadPoolExecutor
# Execute kata_netizen & streaming_links in parallel
```

### 💾 Memory usage growing

**Cause**: Chat history growing indefinitely

**Solution**: Truncate history in session state

```python
# In process_query():
self.chat_history = self.chat_history[-6:]  # Keep last 6 messages only
```

---

## API Issues

### ❌ `403 Forbidden` from Google API

**Cause**: API key invalid or project not enabled

**Solutions**:

1. Verify API key is valid: `makersuite.google.com/app/apikey`
2. Enable required APIs in Google Cloud:
   - `Generative Language API` (for embeddings)
   - `Custom Search API` (for streaming search)

### ❌ Qdrant connection timeout

**Cause**: Network issue or server down

**Solutions**:

```bash
# Test connection
ping your-qdrant-url

# Or test via curl
curl -X GET "https://your-qdrant-url/health"

# Should return: {"status":"ok"}
```

### ❌ Rate limiting errors

**Cause**: Too many API requests

**Symptoms**:

- 429 status code
- "Quota exceeded" error
- Requests suddenly start failing

**Solutions**:

1. Implement exponential backoff:

```python
import time

def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

2. Cache results for same queries
3. Reduce batch size

---

## UI/Display Issues

### ❌ Movie cards not displaying

**Cause**: CSS not loading or HTML malformed

**Solutions**:

1. Check browser console for JS errors (F12)
2. Verify poster_url is valid:

```python
# Add validation in render code
if movie.get('poster_url') and movie['poster_url'].startswith('http'):
    # Use image
else:
    # Use placeholder
    poster_url = 'https://via.placeholder.com/200x300?text=No+Poster'
```

### ❌ Chat history not persisting

**Cause**: Session state reset

**Note**: This is expected behavior in Streamlit!

- Session state clears on page refresh
- Session state clears when script is edited and saved
- To persist across sessions, use database

**Solution** (if persistence needed):

```python
import json

# Save to file
def save_history():
    with open('chat_history.json', 'w') as f:
        json.dump(st.session_state.messages, f)

# Load from file
def load_history():
    try:
        with open('chat_history.json', 'r') as f:
            return json.load(f)
    except:
        return [default_message]
```

### ❌ Mood badges show garbled text

**Cause**: Special characters in mood names

**Solution**: URL-encode or filter special chars

```python
mood_badges = " ".join([
    f"🎭 {mood.replace('<', '').replace('>', '')}"
    for mood in content.get('detected_moods', [])
])
```

---

## JSON/Parsing Issues

### ❌ `JSONDecodeError: Expecting value`

**Cause**: LLM output is not valid JSON

**Solutions**:

1. Improve prompt to enforce JSON:

```python
self.prompt = PromptTemplate(
    template="""...\n{format_instructions}
    IMPORTANT: Output MUST be valid JSON only, no other text."""
)
```

2. Add fallback parsing:

```python
try:
    analysis = self.chain.invoke(...)
except Exception as e:
    analysis = {
        "detected_moods": ["unknown"],
        "summary": "Terjadi kesalahan parsing",
        "search_keywords": user_input
    }
```

3. Use regex extraction:

```python
import re
import json

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None
```

---

## Database Issues

### ❌ `Collection not found` error

**Cause**: Collection doesn't exist in Qdrant

**Solutions**:

```bash
# List existing collections
curl -X GET "https://your-qdrant-url/collections"

# Create collection (if needed)
curl -X POST "https://your-qdrant-url/collections/moodviedb" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'

# Note: Adjust size based on embedding model
# Google Embeddings = 768
# HuggingFace all-MiniLM-L6-v2 = 384
```

### ❌ Embedding dimension mismatch

**Error**: `Vector size mismatch`

**Cause**: Collection vector size doesn't match embeddings

**Fix**:

- Google Embeddings = 768 dimensions
- Update Qdrant collection if needed

---

## Debugging Techniques

### 1. Add print statements

```python
print("DEBUG: Entering process_query")
print(f"DEBUG: User input: {user_input}")
print(f"DEBUG: Analysis: {analysis}")
print(f"DEBUG: Retrieved movies: {len(search_results)}")
```

### 2. Use Streamlit debug mode

```bash
streamlit run app.py --logger.level=debug
```

### 3. Check logs

```bash
# Streamlit logs usually in ~/.streamlit/logs/
tail -f ~/.streamlit/logs/*.log
```

### 4. Interactive debugging

```python
# Add to session state for inspection
st.write("DEBUG - Session State:", st.session_state)
st.write("DEBUG - Agent Chat History:", agent.chat_history)
```

### 5. Test components separately

```python
# Test just the LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
response = llm.invoke("Test prompt")
print(response.content)

# Test just embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector = embeddings.embed_query("test query")
print(f"Vector dimension: {len(vector)}")

# Test just Qdrant
client = QdrantClient(url="...", api_key="...")
collections = client.get_collections()
print(f"Collections: {collections}")
```

---

## Emergency Procedures

### 🚨 App is completely broken

```bash
# 1. Clear cache
rm -rf ~/.streamlit/
rm -rf __pycache__/
rm -rf .pytest_cache/

# 2. Reinstall dependencies
pip install --upgrade --force-reinstall -r requirements.txt

# 3. Restart app
streamlit run app.py
```

### 🚨 Stuck in infinite loop

Press `Ctrl+C` in terminal to stop Streamlit

### 🚨 Memory leak suspected

```bash
# Monitor memory usage
watch -n 1 'ps aux | grep streamlit | grep -v grep'

# Or use Python's memory profiler
pip install memory-profiler
python -m memory_profiler app.py
```

---

## Getting Help

1. **Check console output** for specific error messages
2. **Review logs** in `.streamlit/logs/`
3. **Test individual components** in isolation
4. **Verify configuration** in `secrets.toml`
5. **Check API quotas** in respective dashboards
6. **Search GitHub issues** for similar problems

---

**Last Updated**: December 2024
