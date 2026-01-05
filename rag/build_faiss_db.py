# build_faiss_db.py
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pickle
import os

# ======================
# Config
# ======================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "rag_db"
INDEX_PATH = os.path.join(DB_DIR, "agnews.faiss")
META_PATH = os.path.join(DB_DIR, "agnews_meta.pkl")

MAX_DOCS = 5000

os.makedirs(DB_DIR, exist_ok=True)

# ======================
# 1. Load dataset
# ======================
print("Loading AG News...")
ds = load_dataset("ag_news", split=f"train[:{MAX_DOCS}]")

documents = [ex["text"] for ex in ds]
labels = [ex["label"] for ex in ds]

# ======================
# 2. Embed documents
# ======================
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Encoding documents...")
embeddings = embedder.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

dim = embeddings.shape[1]

# ======================
# 3. Build FAISS index
# ======================
print("Building FAISS index...")
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

# save metadata
with open(META_PATH, "wb") as f:
    pickle.dump(
        {
            "documents": documents,
            "labels": labels
        },
        f
    )

print(f"✅ DB built with {index.ntotal} docs")
