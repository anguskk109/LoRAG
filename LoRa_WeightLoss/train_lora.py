# train_lora.py
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass

# ======================
# Data Collator Helper
# ======================
@dataclass
class RelevanceDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        loss_weights = [f["loss_weights"] for f in features]

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        def pad(seq, pad_val):
            return seq + [pad_val] * (max_len - len(seq))

        batch["labels"] = torch.tensor(
            [pad(lbl, -100) for lbl in labels], dtype=torch.long
        )
        batch["loss_weights"] = torch.tensor(
            [pad(w, 0.0) for w in loss_weights], dtype=torch.float
        )

        return batch

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        weighted_loss = loss * shift_weights.view(-1)
        final_loss = weighted_loss.sum() / (shift_weights.sum() + 1e-8)

        return (final_loss, outputs) if return_outputs else final_loss


def main():

    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    DATA_PATH = "data/lora_train.jsonl"
    OUTPUT_DIR = "outputs/qwen-relevance-lora"

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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # ======================
    # 3. Apply LoRA
    # ======================
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # ======================
    # 4. Format Data
    # ======================
    def format_sample(sample):
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant that evaluates relevance between a query and a document. Always respond in valid JSON."
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

    def tokenize_function(examples):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        loss_weights_list = []

        for prompt_text, full_text in zip(examples["prompt_text"], examples["full_text"]):

            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=192,
                padding=False,
                add_special_tokens=True
            )

            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=192,
                padding=False,
                add_special_tokens=True,
                return_offsets_mapping=True
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]
            offsets = full_tokens["offset_mapping"]

            assistant_start = len(prompt_tokens["input_ids"])

            labels = [-100] * assistant_start + input_ids[assistant_start:]
            labels = labels[:len(input_ids)]

            # ----------------------
            # Build loss weights
            # ----------------------
            loss_weights = [0.0] * assistant_start + [1.0] * (len(input_ids) - assistant_start)
            loss_weights = loss_weights[:len(input_ids)]

            assistant_text = full_text[len(prompt_text):]

            relevant_idx = assistant_text.find('"relevant"')
            if relevant_idx != -1:
                for i, (start, end) in enumerate(offsets):
                    if start is None or end is None:
                        continue
                    if start >= relevant_idx and start <= relevant_idx + 30:
                        if i < len(loss_weights):
                            # weight boost
                            loss_weights[i] = 5.0  

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            loss_weights_list.append(loss_weights)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
            "loss_weights": loss_weights_list,
        }


    train_dataset = train_dataset.map(format_sample, remove_columns=train_dataset.column_names)
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)


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
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,

        #num_train_epochs=2,
        max_steps=200,
        
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    data_collator = RelevanceDataCollator(tokenizer)

    trainer = WeightedLossTrainer(
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