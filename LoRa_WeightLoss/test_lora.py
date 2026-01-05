# text_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. Load base model + LoRA adapter ===
BASE_MODEL = "Qwen/Qwen2.5-1.5B"
LORA_PATH = "outputs/qwen-relevance-lora"

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

# === 2. Prepare prompt function ===
def build_prompt(query: str, document: str) -> str:
    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant that evaluates relevance between a query and a document. Always respond in valid JSON."
    }
    user_msg = {
        "role": "user",
        "content": f"Evaluate if the following news article matches the user query. Respond in JSON.\nQuery: {query}\nDocument: {document}"
    }

    # Use Qwen chat template
    prompt_text = tokenizer.apply_chat_template(
        [system_msg, user_msg],
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt_text

# === 3. Inference function ===
def generate_relevance(query: str, document: str, max_new_tokens=128):
    prompt = build_prompt(query, document)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=192
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode output
    text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

# === 4. Test run ===
if __name__ == "__main__":
    query = "Anything new in science or technology?"
    document = "Blog Interrupted The instant message blinked on the computer at Jessica Cutler's desk in the Russell Senate Office Building. \"Oh my God, you're famous.\""

    result = generate_relevance(query, document)
    print("=== Relevance Output ===")
    print(result)
