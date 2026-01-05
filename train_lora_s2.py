# train_lora_s2.py
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import List, Dict

# ======================
# Data Collator Helper
# ======================
@dataclass
class RelevanceDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad input_ids & attention_mask
        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        # Pad labels with -100
        padded_labels = []
        for lbl in labels:
            padded_labels.append(lbl + [-100] * (max_len - len(lbl)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def main():

    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    DATA_PATH = "data/lora_train.jsonl"
    OUTPUT_DIR = "outputs/S2-reasoning-lora"

    STAGE1_ADAPTER = "outputs/S1-relevance-lora"

    # ======================
    # 1. Load Dataset
    # ======================
    train_dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train"
    )

    # ======================
    # 2. Model & Tokenizer
    # ======================

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.eos_token_id = tokenizer.eos_token_id

    # ======================
    # 3. Load in Stage-1 LoRA
    # ======================
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER)
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.train()

    # ======================
    # 4. Format Data
    # ======================
    def format_sample(sample):
        system_msg = {
            "role": "system",
            "content":(
                 "You are a helpful assistant that evaluates relevance between a query and a document."
                 "Provide a structured reasoning for why the document is relevant or not."
            )
        }
        user_msg = {
            "role": "user",
            "content": f"{sample['instruction']}\n{sample['input']}"
        }
        assistant_msg = {
            "role": "assistant",
            "content": sample["output"] + tokenizer.eos_token
        }

        # Prompt only (no assistant answer)
        prompt_text = tokenizer.apply_chat_template(
            [system_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True
        )

        # Full conversation (with answer)
        full_text = tokenizer.apply_chat_template(
            [system_msg, user_msg, assistant_msg],
            tokenize=False,
            add_generation_prompt=False
        )

        return {
            "prompt_text": prompt_text,
            "full_text": full_text
        }

    # ======================
    # Stage-2: Only unmask "reason" part
    # ======================
    def tokenize_function_stage2(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for prompt_text, full_text in zip(examples["prompt_text"], examples["full_text"]):
            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=192,
                padding=False
            )

            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=192,
                padding=False
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]

            assistant_start = len(prompt_tokens["input_ids"])

            # Start with all masked
            labels = [-100] * len(input_ids)

            # Stage-2: unmask everything after "relevant"
            assistant_text = full_text[len(prompt_text):]
            relevant_idx = assistant_text.find('"relevant"')
            if relevant_idx != -1:
                start_token_idx = assistant_start
                # skip "relevant" key and its value, assume ~5 tokens covers it
                unmask_start = min(start_token_idx + 5, len(labels))
                # unmask everything after
                for i in range(unmask_start, len(labels)):
                    labels[i] = input_ids[i]

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }


    train_dataset = train_dataset.map(format_sample, remove_columns=train_dataset.column_names)
    tokenized_dataset = train_dataset.map(
        tokenize_function_stage2,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    # Sanity Check

    # print(tokenizer.chat_template)
    # sample = tokenized_dataset[0]
    # print(tokenizer.decode(sample["input_ids"]))
    # print(sample["labels"])


    # ======================
    # 5. Training
    # ======================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, # effective batch = 16
        optim="paged_adamw_32bit",
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        num_train_epochs=1,
        # max_steps=500,
        
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    data_collator = RelevanceDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training completed.")

if __name__ == "__main__":
    main()