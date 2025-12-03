# Streamlit Secrets Management Guide

## 📋 Lokasi File

- **Development**: `.streamlit/secrets.toml` (lokal)
- **Production (Streamlit Cloud)**: Settings → Secrets di dashboard
- **Format**: TOML

## 🔧 Setup Local Development

### 1. Buat direktori `.streamlit`

```bash
mkdir .streamlit
```

### 2. Buat file `secrets.toml`

Salin konten dari `secrets_template.toml` ke `.streamlit/secrets.toml`

### 3. Isi dengan API keys Anda

```toml
GOOGLE_API_KEY = "sk-xxxx..."
GOOGLE_SEARCH_ENGINE_ID = "147ef6a3287fb443d"
QDRANT_URL = "https://your-instance.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiI..."
```

## 🚀 Akses Secrets dalam Kode

```python
import streamlit as st

# Akses individual secret
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Akses dengan default value
qdrant_url = st.secrets.get("QDRANT_URL", "http://localhost:6333")

# Akses seluruh secrets (dict)
all_secrets = st.secrets.to_dict()
```

## ☁️ Streamlit Cloud Deployment

### 1. Push ke GitHub (tanpa secrets.toml)

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy ke Streamlit Cloud

- Buka https://share.streamlit.io/
- Klik "New app"
- Pilih repository, branch, dan main file

### 3. Konfigurasi Secrets

1. Klik "Advanced settings" → "Secrets"
2. Paste konten secrets.toml Anda:

```
GOOGLE_API_KEY = "your_key"
GOOGLE_SEARCH_ENGINE_ID = "your_id"
GOOGLE_SEARCH_API_KEY = "your_api_key"
QDRANT_URL = "your_qdrant_url"
QDRANT_API_KEY = "your_qdrant_api_key"
QDRANT_COLLECTION = "moodviedb"
```

## 🔐 Security Best Practices

✅ **DO:**

- Gunakan `.streamlit/secrets.toml` hanya untuk development
- Simpan secrets di Streamlit Cloud dashboard untuk production
- Rotate API keys secara berkala
- Gunakan environment-specific secrets

❌ **DON'T:**

- Commit `secrets.toml` ke Git
- Share secrets di Slack/Email
- Hardcode API keys di source code
- Gunakan API key yang sama untuk testing dan production

## 📝 .gitignore Configuration

Pastikan `.streamlit/` ada di `.gitignore`:

```
# Streamlit
.streamlit/secrets.toml
.streamlit/
```

## 🛡️ File Structure

```
moodvie-fix/
├── .streamlit/
│   ├── secrets.toml          # ⚠️ Jangan commit ini!
│   └── config.toml           # Opsional: konfigurasi Streamlit
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ✨ Testing Secrets

```python
import streamlit as st

st.write("Secrets yang tersedia:")
try:
    keys = st.secrets.to_dict().keys()
    for key in keys:
        st.write(f"✓ {key}")
except Exception as e:
    st.error(f"Error: {e}")
```

---

**Untuk informasi lebih lanjut**: https://docs.streamlit.io/library/develop/connections/secrets-management
