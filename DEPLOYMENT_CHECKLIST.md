# ✅ DEPLOYMENT CHECKLIST

## Pre-Deployment Requirements

### 1. Environment Setup

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed: `pip install -r requirements.txt`

### 2. API Keys & Configuration

- [ ] Google API Key obtained (https://makersuite.google.com/app/apikey)
- [ ] Qdrant instance running and accessible
- [ ] `.streamlit/secrets.toml` created with all required keys
- [ ] Google Custom Search ID configured (optional but recommended)

### 3. Code Review

- [ ] `app.py` has no syntax errors
- [ ] All imports are present and correct
- [ ] MoodAnalysis model defined properly
- [ ] MovieChatbot class fully implemented
- [ ] Streamlit UI components render correctly

### 4. Data Validation

- [ ] Qdrant collection exists and contains movies
- [ ] Qdrant collection has correct vector dimensions (768)
- [ ] Sample movies retrieved successfully from Qdrant
- [ ] Google Embeddings working correctly

---

## Local Testing Checklist

### Basic Functionality

```bash
# Start application
streamlit run app.py

# Test Cases:
```

#### Test 1: Mood Analysis

```
Input: "aku sedih banget"
Expected:
  ✓ Mood analysis shows "sad", "lonely", or similar
  ✓ Summary is empathetic in Indonesian/English
  ✓ Search keywords are detailed plot descriptions
```

#### Test 2: Movie Retrieval

```
Expected:
  ✓ 3 movies appear (or fewer if collection small)
  ✓ Each movie has: title, year, rating, genre
  ✓ Poster images load correctly
  ✓ Overview text displays
```

#### Test 3: Kata Netizen Generation

```
Expected:
  ✓ Each movie has a witty Indonesian review
  ✓ Review uses casual language (bahasa gaul)
  ✓ Review is 2-3 sentences
```

#### Test 4: Streaming Links

```
Expected:
  ✓ Streaming links appear for popular movies
  ✓ Links are from legal domains (Netflix, Disney+, etc.)
  ✓ Links are clickable and work
```

#### Test 5: Chat History

```
Expected:
  ✓ User message appears in chat
  ✓ Assistant response appears immediately after
  ✓ Multiple messages build conversation history
  ✓ Mood badges appear correctly
```

#### Test 6: Error Handling

```
Test Cases:
  ✓ Send empty message → error or ignored
  ✓ Send very long message → truncated/handled
  ✓ Disconnect internet → graceful error
  ✓ No results found → "tidak ada film" message
```

---

## Performance Benchmarks

### Expected Response Times

- Mood Analysis: 1-2 seconds
- Qdrant Search: 0.1-0.5 seconds
- Kata Netizen Generation: 1-2 seconds per movie (3-6 total)
- Streaming Link Search: 0.5-1 second per movie (1.5-3 total)
- **Total End-to-End: 3-8 seconds**

### Memory Usage

- Initial load: ~200-300 MB
- Per message: +50-100 MB
- With 10 messages: ~500-600 MB

If exceeding 1 GB: implement history truncation

---

## Browser Compatibility Testing

- [ ] Chrome/Chromium: Full support expected
- [ ] Firefox: Full support expected
- [ ] Safari: Full support (might need font adjustments)
- [ ] Edge: Full support expected
- [ ] Mobile browsers: CSS should adapt (responsive design)

### Test on Mobile

```
- Portrait view: Cards should stack vertically ✓
- Landscape view: Cards should maintain layout ✓
- Touch input: Chat input should work ✓
```

---

## Documentation Review

- [ ] README.md updated with new features
- [ ] IMPLEMENTATION_NOTES.md explains all changes
- [ ] USAGE_GUIDE.md has clear examples
- [ ] TROUBLESHOOTING.md covers common issues
- [ ] secrets.example.toml shows all required keys

---

## Pre-Production Deployments

### Staging Environment

```bash
# Test on staging server first
streamlit run app.py --server.port 8501

# Verify:
- All API connections work ✓
- No console errors ✓
- Response times acceptable ✓
- All UI elements render ✓
```

### Production Environment

1. **Server Requirements**

   - [ ] Minimum 2 CPU cores
   - [ ] Minimum 4 GB RAM
   - [ ] Minimum 10 GB disk space
   - [ ] Python 3.8+ runtime

2. **Docker Deployment** (Optional)

   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["streamlit", "run", "app.py"]
   ```

3. **Environment Variables**
   ```bash
   export STREAMLIT_SERVER_HEADLESS=true
   export STREAMLIT_LOGGER_LEVEL=info
   export STREAMLIT_CLIENT_SHOWERRORDETAILS=true
   ```

---

## Monitoring & Maintenance

### Daily Monitoring

- [ ] Check error logs
- [ ] Verify API quotas not exceeded
- [ ] Monitor response times
- [ ] Check Qdrant collection health

### Weekly Maintenance

- [ ] Review chat analytics
- [ ] Update movie database if needed
- [ ] Check for security updates
- [ ] Backup Qdrant collection

### Monthly Reviews

- [ ] Performance analysis
- [ ] User feedback review
- [ ] API cost optimization
- [ ] Security audit

---

## Deployment Commands

### Local Testing

```bash
cd /path/to/moodvie-fix
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

### Docker Deployment

```bash
docker build -t moodvie:latest .
docker run -p 8501:8501 \
  --env-file .env \
  --mount type=bind,source=$(pwd)/.streamlit,target=/app/.streamlit \
  moodvie:latest
```

### Cloud Deployment (Streamlit Cloud)

```bash
# Requires GitHub repo
# Push to GitHub
git push origin main

# Go to https://share.streamlit.io
# Connect GitHub repo
# Deploy!
```

---

## Go/No-Go Decision

### ✅ Go Live If:

- [ ] All syntax errors resolved
- [ ] API keys tested and working
- [ ] Movies retrieve successfully
- [ ] Response times within acceptable range
- [ ] All UI elements render correctly
- [ ] Chat history works
- [ ] Error handling tested
- [ ] No console errors

### ❌ Don't Go Live If:

- [ ] Any critical errors in console
- [ ] API keys not working
- [ ] No movies retrieved
- [ ] Response time > 15 seconds
- [ ] Qdrant connection failing
- [ ] Memory usage > 2GB
- [ ] Security concerns identified

---

## Post-Deployment

### First 24 Hours

- [ ] Monitor error logs closely
- [ ] Check API usage/quota
- [ ] Verify user experience
- [ ] Be ready for quick fixes

### First Week

- [ ] Collect user feedback
- [ ] Monitor performance metrics
- [ ] Adjust parameters as needed
- [ ] Document issues found

### First Month

- [ ] Analyze usage patterns
- [ ] Optimize slow queries
- [ ] Plan improvements
- [ ] Security audit

---

## Rollback Plan

If critical issues arise:

```bash
# Stop current version
streamlit stop

# Revert to backup
git revert <commit-hash>

# Or restore from backup
cp backups/app.py.backup app.py
streamlit run app.py
```

---

## Support & Escalation

### Level 1 Issues (Can Wait)

- UI improvements
- Feature requests
- Performance optimization
- Documentation updates

### Level 2 Issues (Should Fix)

- Occasional errors
- API quota warnings
- Slow response times
- Data inconsistencies

### Level 3 Issues (Urgent)

- Complete service down
- Critical security vulnerability
- Data loss
- API key exposed

---

**Last Updated**: December 2024
**Status**: Ready for Deployment ✅
