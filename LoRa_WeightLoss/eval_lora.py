# eval_lora.py
import torch
import json
import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "outputs/qwen-relevance-lora"
EVAL_DATA_PATH = "data/lora_eval.jsonl"

MAX_PROMPT_LEN = 192
MAX_NEW_TOKENS = 64


# ======================
# 1. Load model + LoRA
# ======================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


# ======================
# 2. Prompt building
# ======================
def build_eval_prompt(sample):
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful assistant that evaluates relevance "
            "between a query and a document. Always respond in valid JSON."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"{sample['instruction']}\n{sample['input']}"
    }

    return tokenizer.apply_chat_template(
        [system_msg, user_msg],
        tokenize=False,
        add_generation_prompt=True
    )


# ======================
# 3. Generation
# ======================
@torch.no_grad()
def generate_relevance(prompt_text):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()


# ======================
# 4. Parsing
# ======================
def extract_relevant_flag(text):

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            obj = json.loads(text[start:end])
            return bool(obj.get("relevant"))
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Fallback to regex
    match = re.search(r'"relevant"\s*:\s*(true|false|True|False)', text)
    if match:
        return match.group(1).lower() == "true"
    
    return None


# ======================
# 5. Evaluation loop
# ======================
print("Loading eval dataset...")

dataset = load_dataset("json", data_files=EVAL_DATA_PATH, split="train").select(range(100))

#dataset = load_dataset("json", data_files="data/lora_train.jsonl", split="train").select(range(50))

correct = 0
total = 0
failed = 0

print("Running evaluation...")
for sample in dataset:
    prompt = build_eval_prompt(sample)
    output = generate_relevance(prompt)
    pred = extract_relevant_flag(output)

    try:
        gold = json.loads(sample["output"])["relevant"]
    except Exception:
        continue

    if pred is None:
        failed += 1
    else:
        correct += int(pred == gold)

    total += 1

accuracy = correct / max(total, 1)

print("\n Eval Results")
print(f"Total samples : {total}")
print(f"Correct       : {correct}")
print(f"Failed parse  : {failed}")
print(f"Accuracy      : {accuracy:.4f}")
