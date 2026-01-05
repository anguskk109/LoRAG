# rag_eval_p5.py
import json
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================
# Config
# ======================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
S2_LORA_PATH = "outputs/S2-reasoning-lora"

DB_DIR = "rag_db"
INDEX_PATH = f"{DB_DIR}/agnews.faiss"
META_PATH = f"{DB_DIR}/agnews_meta.pkl"
EVAL_PATH = "data/lora_eval.jsonl"

TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# 1. Load FAISS DB
# ======================
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

documents = meta["documents"]

# ======================
# 2. Load embedding model
# ======================
embedder = SentenceTransformer(EMBED_MODEL)

# ======================
# 3. Load Stage-2 LoRA
# ======================
print("Loading Stage-2 evaluator...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, S2_LORA_PATH, device_map="auto")
model.eval()

# ======================
# 4. Helpers
# ======================
def build_prompt(query, document):
    system = {
        "role": "system",
        "content": "You evaluate relevance between a query and a document. Respond in valid JSON."
    }
    user = {
        "role": "user",
        "content": f"Query: {query}\nDocument: {document}"
    }
    return tokenizer.apply_chat_template(
        [system, user],
        tokenize=False,
        add_generation_prompt=True
    )

def judge_relevance(query, document):
    prompt = build_prompt(query, document)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=192
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    try:
        parsed = json.loads(text)
        return bool(parsed.get("relevant", False))
    except Exception:
        return False

# ======================
# 5. Load queries
# ======================
def extract_query(input_text):
    q, _ = input_text.split("\nDocument:", 1)
    return q.replace("Query:", "").strip()

queries = []
seen = set()
MAXX_QUERIES = 10

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        q = extract_query(ex["input"])
        if q not in seen:
            seen.add(q)
            queries.append(q)
        if len(queries) >= MAXX_QUERIES:
            break

print(f"Loaded {len(queries)} queries")

# ======================
# 6. Evaluation
# ======================
precisions = []

for q in queries:
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, indices = index.search(q_emb, TOP_K)

    relevant_cnt = 0
    for idx in indices[0]:
        doc = documents[idx]
        if judge_relevance(q, doc):
            relevant_cnt += 1

    p_at_5 = relevant_cnt / TOP_K
    precisions.append(p_at_5)

    print(f"Query: {q}")
    print(f"P@5 = {p_at_5:.2f}\n")

mean_p5 = sum(precisions) / len(precisions)

print("==== RAG Evaluation ====")
print(f"Queries evaluated: {len(queries)}")
print(f"Mean P@5: {mean_p5:.3f}")
