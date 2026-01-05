# eval_lora_s2.py
import json
import torch
from difflib import SequenceMatcher
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
LORA_PATH = "outputs/S2-reasoning-lora"
EVAL_PATH = "data/lora_eval.jsonl"

# ======================
# 1. Load model
# ======================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="auto")
model.eval()

# ======================
# 2. Prompt builder
# ======================
def build_prompt(instruction: str, query: str, document: str) -> str:
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant that evaluates relevance between a query and a document. Always respond in valid JSON."
    }
    user_msg = {
        "role": "user",
        "content": f"{instruction}\n\nQuery: {query}\nDocument: {document}"
    }

    return tokenizer.apply_chat_template(
        [system_msg, user_msg],
        tokenize=False,
        add_generation_prompt=True
    )

# ======================
# 3. Generation
# ======================
def generate(instruction: str, query: str, document: str, max_new_tokens=128) -> str:
    prompt = build_prompt(instruction, query, document)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

# ======================
# 4. JSONL loader (schema-aware)
# ======================
def load_lora_eval_jsonl(path: str, limit: int = None) -> List[Dict]:
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            raw = json.loads(line)

            # ---- Parse input ----
            inp = raw["input"]
            try:
                q_part, d_part = inp.split("\nDocument:", 1)
                query = q_part.replace("Query:", "").strip()
                document = d_part.strip()
            except Exception:
                continue

            # ---- Parse gold output ----
            gold = json.loads(raw["output"])

            samples.append({
                "instruction": raw["instruction"],
                "query": query,
                "document": document,
                "relevant": gold["relevant"],
                "gold_query_topic": gold["reason"]["query_topic"],
                "gold_document_topic": gold["reason"]["document_topic"],
                "gold_summary": gold["reason"]["summary"],
            })

    return samples

# ======================
# 5. Metrics helpers
# ======================
def try_parse_json(text: str):
    try:
        return json.loads(text), True
    except Exception:
        return None, False

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ======================
# 6. Evaluation
# ======================
def evaluate(samples: List[Dict]):
    stats = {
        "total": 0,
        "json_valid": 0,
        "relevance_match": 0,
        "reason_complete": 0,
        "query_topic_match": 0,
        "document_topic_match": 0,
        "summary_sim": [],
    }

    for s in samples:
        stats["total"] += 1

        output = generate(
            s["instruction"],
            s["query"],
            s["document"]
        )

        parsed, valid = try_parse_json(output)
        if not valid:
            continue

        stats["json_valid"] += 1

        # relevance agreement
        if parsed.get("relevant") == s["relevant"]:
            stats["relevance_match"] += 1

        reason = parsed.get("reason", {})
        if all(k in reason for k in ["query_topic", "document_topic", "summary"]):
            stats["reason_complete"] += 1

            if reason["query_topic"] == s["gold_query_topic"]:
                stats["query_topic_match"] += 1

            if reason["document_topic"] == s["gold_document_topic"]:
                stats["document_topic_match"] += 1

            stats["summary_sim"].append(
                similarity(reason["summary"], s["gold_summary"])
            )

    total = stats["total"]

    print("\n=== Stage-2 LoRA Evaluation ===")
    print(f"Samples evaluated: {total}")
    print(f"JSON valid rate:        {stats['json_valid'] / total:.3f}")
    print(f"Relevance agreement:    {stats['relevance_match'] / total:.3f}")
    print(f"Reason completeness:   {stats['reason_complete'] / total:.3f}")
    print(f"Query topic accuracy:  {stats['query_topic_match'] / total:.3f}")
    print(f"Doc topic accuracy:    {stats['document_topic_match'] / total:.3f}")

    if stats["summary_sim"]:
        print(f"Avg summary similarity:{sum(stats['summary_sim']) / len(stats['summary_sim']):.3f}")

if __name__ == "__main__":
    samples = load_lora_eval_jsonl(EVAL_PATH, limit=100)
    print(f"Loaded {len(samples)} evaluation samples")

    evaluate(samples)
