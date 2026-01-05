# generate_samples.py
from datasets import load_dataset
import random
import json
import os

# OUTPUT_FILE = "data/lora_train.jsonl"
# train_ds = load_dataset("ag_news", split="train[:5000]")


train_ds = load_dataset("ag_news", split="train[10000:10500]")
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

LABEL_TO_TOPIC = {
    0: "world",
    1: "sports",
    2: "business",
    3: "science and technology"
}

def make_input(query, document):
    return f"Query: {query}\nDocument: {document}"


def make_output(is_relevant, query_topic, document_topic):
    summary = (
        f"The document discusses {document_topic}, which "
        f"{'matches' if is_relevant else 'does not match'} the query."
    )
    return json.dumps({
        "relevant": is_relevant,
        "reason": {
            "query_topic": query_topic,
            "document_topic": document_topic,
            "summary": summary
        }
    }, ensure_ascii=False)

def generate_samples(example, num_negatives=1):
    samples = []

    doc_text = example["text"]
    doc_label = example["label"]
    doc_topic = LABEL_TO_TOPIC[doc_label]

    # ---- Positive sample ----
    query = random.choice(LABEL_TO_QUERIES[doc_label])
    instruction = random.choice(INSTRUCTIONS)

    samples.append({
        "instruction": instruction,
        "input": make_input(query, doc_text),
        "output": make_output(
            True,
            query_topic=doc_topic,
            document_topic=doc_topic
        )
    })

    # ---- Negative samples ----
    neg_labels = [l for l in LABEL_TO_QUERIES.keys() if l != doc_label]
    neg_labels = random.sample(neg_labels, k=num_negatives)

    for neg_label in neg_labels:
        neg_query = random.choice(LABEL_TO_QUERIES[neg_label])
        instruction = random.choice(INSTRUCTIONS)
        neg_topic = LABEL_TO_TOPIC[neg_label]

        samples.append({
            "instruction": instruction,
            "input": make_input(neg_query, doc_text),
            "output": make_output(
                False,
                query_topic=neg_topic,
                document_topic=doc_topic
            )
        })

    return samples

all_samples = []

for ex in train_ds:
    all_samples.extend(generate_samples(ex, num_negatives=1))

# random.shuffle(all_samples)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print("✅ Done.")