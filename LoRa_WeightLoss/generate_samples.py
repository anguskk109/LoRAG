# generate_samples.py
from datasets import load_dataset
import random
import json
import os

#OUTPUT_FILE = "data/lora_train.jsonl"
#train_ds = load_dataset("ag_news", split="train[:3000]")


train_ds = load_dataset("ag_news", split="train[10000:10501]")
OUTPUT_FILE = "data/lora_eval.jsonl"

os.makedirs("data", exist_ok=True)



INSTRUCTIONS = [
    "Decide whether the document is relevant to the query. Output JSON with 'relevant' and 'reason'.",
    "Evaluate if the following news article matches the user query. Respond in JSON.",
    "Determine the relevance of the document to the query. Answer using JSON format.",
    "Is the document relevant to the given query? Provide a JSON answer with reasoning."
]

LABEL_TO_QUERIES = {
    0: [
        "Give me world news",
        "International news articles",
        "News about global events and politics",
        "I want news about global events.",
        "Show me international news.",
        "News from around the world, please."
    ],
    1: [
        "Give me sports news",
        "News about sports events and competitions",
        "Latest updates in sports",
        "I'm looking for the latest sports updates.",
        "Show me headlines about sports.",
        "Any recent sports news?"
    ],
    2: [
        "Give me business news",
        "News about companies and markets",
        "Business and financial news",
        "I want updates on the market and companies.",
        "Show me corporate or financial news.",
        "News about business and finance, please."
    ],
    3: [
        "Give me science and technology news",
        "Technology or science-related articles",
        "News about scientific and technological developments",
        "I'm interested in science and technology updates.",
        "Show me news about new tech or research.",
        "Anything new in science or technology?"
    ]
}


POS_REASONS = {
    0: "The article discusses international events and global affairs, which matches the world news topic.",
    1: "The article is about sports events or competitions, aligning with the sports query.",
    2: "The article discusses companies, markets, or financial performance, which fits business news.",
    3: "The article focuses on scientific or technological developments, matching the query."
}

NEG_REASONS = {
    0: "The article does not focus on global or international events.",
    1: "The article is not related to sports or athletic activities.",
    2: "The article does not discuss business, companies, or financial topics.",
    3: "The article is not related to science or technology topics."
}

def make_input(query, document):
    return f"Query: {query}\nDocument: {document}"


def make_output(is_relevant, reason):
    return json.dumps({
        "relevant": is_relevant,
        "reason": reason
    }, ensure_ascii=False)

def generate_samples(example, num_negatives=1):
    samples = []

    doc_text = example["text"]
    doc_label = example["label"]

    # ---- Positive sample ----
    query = random.choice(LABEL_TO_QUERIES[doc_label])
    instruction = random.choice(INSTRUCTIONS)

    samples.append({
        "instruction": instruction,
        "input": make_input(query, doc_text),
        "output": make_output(True, POS_REASONS[doc_label])
    })

    # ---- Negative samples ----
    neg_labels = [l for l in LABEL_TO_QUERIES.keys() if l != doc_label]
    neg_labels = random.sample(neg_labels, k=num_negatives)

    for neg_label in neg_labels:
        neg_query = random.choice(LABEL_TO_QUERIES[neg_label])
        instruction = random.choice(INSTRUCTIONS)

        samples.append({
            "instruction": instruction,
            "input": make_input(neg_query, doc_text),
            "output": make_output(False, NEG_REASONS[neg_label])
        })

    return samples

all_samples = []

for ex in train_ds:
    all_samples.extend(generate_samples(ex, num_negatives=1))

random.shuffle(all_samples)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("✅ Done.")