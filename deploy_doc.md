# Render Deployment Guide — Amazon RAG CS Agent

## What gets created
| Resource | Render type | Plan |
|---|---|---|
| `amazon-cs-agent` | Web Service (FastAPI) | Free |
| `amazon-agent-db` | PostgreSQL | Free |

---

## Step 1 — Files to add to your repo

Copy these 4 files into the **root** of your project folder:

| File | Purpose |
|---|---|
| `requirements.txt` | All Python dependencies |
| `build.sh` | Build script (installs deps + builds FAISS index) |
| `render.yaml` | Render blueprint (wires web service + DB together) |
| `main.py` | Fixed version (3-value unpack from `answer()`) |

Your repo root should look like:
```
RAG C.S. AGENT/
├── DATA/                  ← your PDF/TXT source documents (must be committed)
├── static/                ← demo.html, widget.js
├── app.py
├── build.sh               ← new
├── chunker.py
├── database.py
├── docs_loader.py
├── embedder.py
├── eval.py
├── generator.py
├── main.py                ← replaced
├── models.py
├── render.yaml            ← new
├── requirements.txt       ← new
├── retriever.py
└── vector_store.py
```

> ⚠️ **Critical**: The `DATA/` folder with your PDFs must be committed to Git.  
> The FAISS index is built from it during the Render build step.  
> The `vector_store/` folder should **not** be committed (add it to `.gitignore`).

---

## Step 2 — .gitignore

Make sure your `.gitignore` includes:
```
venv/
__pycache__/
.env
vector_store/
report_visuals/
*.pkl
*.index
eval_results.json
```

---

## Step 3 — Push to GitHub

```bash
git add .
git commit -m "add render deployment files"
git push origin main
```

---

## Step 4 — Deploy on Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New → Blueprint**
3. Connect your GitHub account and select your repo
4. Render detects `render.yaml` automatically — click **Apply**
5. Two resources are created: the PostgreSQL DB and the web service

---

## Step 5 — Set your Groq API key

`render.yaml` marks `GROQ_API_KEY` as `sync: false` (a manual secret):

1. In the Render dashboard, open your **amazon-cs-agent** web service
2. Go to **Environment** tab
3. Click **Add Environment Variable**
4. Key: `GROQ_API_KEY` → Value: your key from [console.groq.com](https://console.groq.com)
5. Click **Save Changes** — Render redeploys automatically

---

## Step 6 — Watch the build logs

The first build takes **8–15 minutes** because it:
- Installs all Python packages (torch, sentence-transformers, faiss-cpu)
- Downloads the `intfloat/multilingual-e5-base` model (~500 MB, cached after first deploy)
- Processes your DATA/ documents and builds the FAISS index

You'll see this in the logs when it's ready:
```
✅ Vector store loaded: N vectors
✅ Database tables verified / created
INFO:     Application startup complete.
```

---

## Step 7 — Test your endpoints

Once deployed, your service URL is shown in the Render dashboard (e.g. `https://amazon-cs-agent.onrender.com`).

```bash
# Health check
curl https://amazon-cs-agent.onrender.com/health

# Ask a question
curl -X POST https://amazon-cs-agent.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return window for damaged products?", "session_id": "test-1"}'

# Analytics summary
curl https://amazon-cs-agent.onrender.com/analytics/summary

# Demo UI
open https://amazon-cs-agent.onrender.com/demo
```

---

## Free tier limitations & tips

| Limitation | Workaround |
|---|---|
| Service sleeps after 15 min inactivity | Upgrade to Starter ($7/mo) for always-on |
| 512 MB RAM on free web service | May be tight with torch; upgrade if OOM |
| PostgreSQL free tier expires after 90 days | Upgrade to Starter DB ($7/mo) before expiry |
| Build timeout: 15 min | First build is slow due to model download; subsequent builds use the HuggingFace cache |

### Speeding up subsequent deploys
`render.yaml` sets `HF_HOME` to a path inside the project directory so Render's build cache preserves the downloaded model between deploys. After the first deploy, rebuilds skip the ~500 MB download.

---

## Environment variables reference

| Variable | Source | Description |
|---|---|---|
| `DATABASE_URL` | Auto-injected from DB | PostgreSQL connection string |
| `GROQ_API_KEY` | Set manually in dashboard | Groq API key |
| `HF_HOME` | `render.yaml` | HuggingFace model cache path |
| `TOKENIZERS_PARALLELISM` | `render.yaml` | Suppresses tokenizer warnings |
| `PYTHONUNBUFFERED` | `render.yaml` | Unbuffered logs |