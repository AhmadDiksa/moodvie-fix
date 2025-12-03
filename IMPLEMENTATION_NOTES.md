# 📋 Implementation Notes - MovieChatbot dengan Qdrant

## Perubahan Utama

### 1. **Struktur Data Model**

- ✅ Added `MoodAnalysis` Pydantic model dengan 3 fields:
  - `detected_moods`: List of detected emotions
  - `summary`: Empathetic summary
  - `search_keywords`: Detailed plot description untuk vector search (tidak menggunakan TMDB)

### 2. **Class MovieChatbot**

Menggantikan `MovieTherapist` dengan implementasi lengkap yang sesuai spesifikasi:

#### Initialization (`__init__`)

- ✅ Menggunakan **Google Embeddings** (models/text-embedding-004) - 768 dimensi
- ✅ Menggunakan **Qdrant Vector Store** sebagai satu-satunya source data
- ✅ Mode: `QDRANT` (tidak ada fallback ke TMDB)
- ✅ Setup Gemini 2.0 Flash untuk NLP analysis

#### Method: `get_streaming_link(movie_title)`

- ✅ Menggunakan Google Custom Search API
- ✅ Filter domain streaming legal (Netflix, Disney+, Hotstar, etc.)
- ✅ Return maksimal 2 link
- ✅ Fallback ke Google search jika tidak ada API key

#### Method: `generate_kata_netizen(title, overview)`

- ✅ Generate witty Indonesian reviews
- ✅ Menggunakan Gemini LLM
- ✅ Output maksimal 3 kalimat dengan bahasa gaul
- ✅ Label: "Kata Netizen"

#### Method: `retrieve_movies(query)`

- ✅ Query Qdrant menggunakan similarity search
- ✅ Return 3 top results
- ✅ Extract metadata: title, year, rating, overview, poster_url, genre, vote_count

#### Method: `process_query(user_input)`

- ✅ Orchestrator utama
- ✅ Analyze mood menggunakan LangChain chain
- ✅ Generate `MoodAnalysis` object
- ✅ Maintain chat history
- ✅ Retrieve movies dan process dengan kata_netizen & streaming links
- ✅ Return structured response dict

### 3. **UI Integration (Streamlit)**

- ✅ Session state management untuk chat history
- ✅ Display mood summary + detected moods badges
- ✅ Render movie cards dengan:
  - Poster image
  - Title, Year, Genre, Rating (stars)
  - "Kata Netizen" box
  - Movie overview (truncated 200 char)
  - Details button (IMDB link)
  - Streaming links (jika ada)

### 4. **Dependencies**

Ditambahkan ke `requirements.txt`:

- `langchain-qdrant>=0.1.0` - LangChain Qdrant integration
- `langchain-google-genai>=0.0.1` - Google Generative AI (untuk embeddings)
- `google-generativeai>=0.3.0` - Google AI SDK

## Konfigurasi yang Diperlukan (secrets.toml)

```toml
GOOGLE_API_KEY = "your-google-api-key"
QDRANT_URL = "https://your-qdrant-instance.com"
QDRANT_API_KEY = "your-qdrant-api-key"
QDRANT_COLLECTION = "moodviedb"
GOOGLE_CSE_ID = "your-google-cse-id"  # Optional untuk streaming search
```

## Flow Aplikasi

1. **User Input** → "aku gabut nih"
2. **Mood Analysis** → LLM detects: ["bored", "lonely"]
3. **Generate Keywords** → "A lighthearted comedy about friendship and adventure..."
4. **Qdrant Search** → Vector similarity search menggunakan Google Embeddings
5. **Retrieve Movies** → 3 top results dari Qdrant
6. **Enrich Data**:
   - Generate "Kata Netizen" untuk setiap film
   - Cari streaming links menggunakan Google Custom Search
7. **Render UI** → Netflix-style cards dengan semua informasi

## Perbedaan Utama dengan TMDB Mode

| Aspek               | Qdrant Mode                  | TMDB Mode              |
| ------------------- | ---------------------------- | ---------------------- |
| **Search Keywords** | Detailed plot description    | Simple 2-3 word query  |
| **Embeddings**      | Google Embeddings (768D)     | HuggingFace embeddings |
| **Database**        | Qdrant Vector DB             | TMDB API               |
| **Search Type**     | Semantic/Vector similarity   | Text keyword search    |
| **Data Source**     | Pre-indexed movies di Qdrant | Live TMDB API calls    |

## Testing Recommendations

```python
# Test mood analysis
"aku sedih banget"  # Expect: sad, depressed emotions

# Test streaming search
"Any movie"  # Should find Netflix/Disney+ links

# Test Qdrant connectivity
# Should initialize without errors in console output

# Test chat history
Send multiple messages → Should maintain context
```

## Performance Notes

- ⚡ **Google Embeddings**: Lebih cepat & akurat untuk semantic search
- 🗄️ **Qdrant**: Vector DB yang scalable & efficient
- 🤖 **Gemini 2.0 Flash**: Model terbaru dengan latency rendah
- 💾 **Session State**: Semua history disimpan di client-side (session)

## Troubleshooting

### ❌ "Konfigurasi Qdrant belum lengkap!"

→ Check `secrets.toml` dan pastikan QDRANT_URL ada

### ❌ "Tidak ada film yang cocok ditemukan"

→ Kemungkinan collection Qdrant kosong atau query tidak match dengan embeddings yang ada

### ❌ GoogleGenerativeAIEmbeddings error

→ Pastikan GOOGLE_API_KEY set dengan benar dan akses Google AI API sudah enabled

## Future Enhancements

- [ ] Add multi-language support
- [ ] Implement user preference learning
- [ ] Add collaborative filtering
- [ ] Cache frequently used embeddings
- [ ] Add rating system untuk recommendations
- [ ] Export conversation history
