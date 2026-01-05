# LoRA-Guided Relevance Evaluation for RAG Systems

This repository presents a **two-stage LoRA fine-tuning framework** for relevance evaluation and structured reasoning in Retrieval-Augmented Generation (RAG) systems. The approach relies purely on **decoder-only language model token loss**, selectively masked to focus on the desired outputs.

The pipeline demonstrates how a **decoder-only LLM** can serve as both a high-precision relevance evaluator and a structured reasoning generator for RAG pipelines, without adding classification heads or switching to encoder-based architectures.

---

## ✨Motivation

RAG systems typically rely on embedding similarity for document retrieval.

This project investigates whether a **decoder-only LLM**, trained with **masked token-level loss**, can achieve:
- accurate binary relevance judgments, and
- semantic, topic-level reasoning about retrieved documents.

---

## 🔬Methodology Overview

The methodology follows a progressive journey of experiments and refinements:

1. **Pure LM token loss**  
    The initial approach fine-tuned the model using standard causal language modeling loss over the full JSON output. While training loss decreased steadily, this did **not** translate to improved relevance accuracy. It is observed that the model primarily learned to reproduce the **JSON schema**, common phrasing, and frequent tokens (e.g., defaulting to `"relevant": true`), rather than conditioning its prediction on the semantic alignment between query and document. This highlights a critical limitation: **minimizing overall token loss does not guarantee learning of the target signal**—especially when the task-critical tokens (e.g., `"true"`/`"false"`) constitute a tiny fraction of the sequence. 
    
    The key challenge, therefore, is to **reweight or mask the loss** so the model is explicitly encouraged to learn what we care about: faithful relevance judgment.

2. **Weighted token-level loss**  
    To prioritize learning of the `"relevant"` field, the loss corresponding to its tokens (`"true"`/`"false"`) was **scaled up by a factor of 5×**, while the remaining tokens in the JSON output retained unit weight. Although this yielded a marginal improvement in relevance accuracy, the approach remained fundamentally limited: the model still optimized over the entire token sequence, causing interference between relevance prediction and reasoning generation. Because both tasks shared the same output stream and training objective, the model struggled to disentangle semantic judgment from descriptive elaboration, resulting in inconsistent reasoning—even when relevance was correctly predicted.

3. **Two-stage LoRA fine-tuning with selective token masking**  
   To address the limitations of the baseline, a **two-stage LoRA fine-tuning strategy** is adpoted with **targeted token masking**, ensuring that each stage learns only what it’s meant to:

   - **Stage-1: Relevance Classification**  
     The model is trained to predict the `"relevant"` field **only**. All tokens **outside** the `"relevant"` field (i.e., everything before and after) are **masked out** during loss computation. This forces the model to focus exclusively on learning binary relevance judgments without being distracted by the reasoning content.  
     → Output: A LoRA adapter (`s1-relevance-lora`) that reliably predicts relevance.

   - **Stage-2: Structured Reasoning Generation**  
     The Stage-1 LoRA adapter is loaded as the base, and training continues **only on the `"reason"` field**. Specifically, all tokens **before the start of the `"reason"` field** are masked, so loss is computed solely on the reasoning tokens. This preserves the frozen relevance judgment while enabling the model to generate detailed, topic-aware justifications.  
     → Output: A final LoRA adapter (`s2-reasoning-lora`) that maintains relevance accuracy while adding high-quality reasoning.


4. **Naive RAG evaluation**  
   A simple FAISS-based retrieval setup is used to demonstrate how the Stage-2 model can evaluate retrieved documents. The focus is on retrieval (`R`) rather than full generative RAG; the Stage-2 LoRA serves as a high-precision evaluator of both relevance and reasoning quality.

This staged approach ensures **controlled, interpretable training** and avoids interference between relevance classification and reasoning generation.


---

## 📝Project Structure

The repository is organized into modular directories for data, training, evaluation, and RAG components:

LoRAG/
├── LoRa_WeightLoss/ # Early training attempts
│
├── outputs/ # NOT Uploaded Here
│ ├── S1-relevance-lora/ # Stage-1 adapter: relevance classification only
│ └── S2-reasoning-lora/ # Stage-2 adapter: reasoning generation + frozen relevance
│
├── rag/ # RAG pipeline components (retrieval & evaluation)
│ ├── build_faiss_db.py # Script to build FAISS vector database from AG News
│ ├── eval_result.txt # Output of RAG evaluation metrics
│ └── rag_eval.py # Main script to run RAG retrieval + LoRA evaluation
│
├── requirements.txt
├── generate_samples.py # Shared by both stages
├── eval_lora_s1.py # Stage-1 specific evaluation script
├── eval_lora_s2.py # Stage-2 specific evaluation script
├── train_lora_s1.py # Stage-1 training script (masks non-"relevant" tokens)
├── train_lora_s2.py # Stage-2 training script (masks pre-"reason" tokens)
└── README.md # This document

---

## 🚂Training Details

- **Base model**: Qwen2.5-1.5B (decoder-only)  
- **Quantization**: 4-bit NF4 via `bitsandbytes`  
- **Fine-tuning**: LoRA (PEFT) with rank `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, and no bias  
- **Target modules**: Attention (`q_proj`, `v_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`)  
- **Optimizer**: `paged_adamw_32bit`  
- **Training objective**: Causal language modeling with **selective token loss masking**  
  - **Stage-1**: Loss applied only to the `"relevant"` field (`true`/`false`)  
  - **Stage-2**: Loss applied only to the `"reason"` field (all prior tokens masked)  
- **No modifications** to model architecture: no classification heads, no encoder, no task-specific layers

---

## 📄Output Format

The model produces structured JSON outputs with relevance and reasoning:

```json
{
  "relevant": true,
  "reason": {
    "query_topic": "sports",
    "document_topic": "sports",
    "summary": "The article discusses recent sports events and competitions, which aligns with the query."
  }
}
```

---

## 🎯Evaluation Results

### Stage-1 Relevance Evaluation  
After Stage-1 training, evaluated on Stage-1 task:  
- Total samples: 100  
- Correct: 93  
- Failed parse: 0  
- **Accuracy: 0.9300**

### Stage-2 Impact on Relevance  
After Stage-2 fine-tuning, evaluated on Stage-1 task:  
- Total samples: 100  
- Correct: 87  
- Failed parse: 0  
- **Accuracy: 0.8700**

> ✅ the slight drop in relevance accuracy is due to **continued parameter updates during Stage-2 training**. Although the loss was masked to apply only to the `"reason"` field, the underlying LoRA weights still receive gradient updates that indirectly affect the representation of earlier tokens —including those responsible for the `"relevant"` prediction.

### Stage-2 LoRA Evaluation (Structured Reasoning)  
Samples evaluated: 100  
- JSON valid rate: **0.970**  
- Relevance agreement: **0.870**  
- Reason completeness: **0.970**  
- Query topic accuracy: **0.970**  
- Document topic accuracy: **0.840**  
- Avg summary similarity: **0.971**

### RAG Precision@5  
Queries evaluated: 10  
- **Mean P@5: 0.940**

**Individual query P@5:**  
- International news articles: 1.00  
- Give me business news: 0.80  
- Show me headlines about sports: 1.00  
- Business and financial news: 0.80  
- Give me world news: 0.80  
- I'm looking for the latest sports updates: 1.00  
- Technology or science-related articles: 1.00  
- I want updates on the market and companies: 1.00  
- I want news about global events: 1.00  
- Latest updates in sports: 1.00

---

## 🧠 Key Takeaways and Reflections

### Decoder-Only Models Can Do Classification — But Only With Care

This project demonstrates that **decoder-only LM can be repurposed for classification-style tasks** *without* adding classification heads or switching to encoder architectures.

Without constrains, a decoder-only LM will optimize for the **easiest token-level objective**, often learning surface patterns (JSON structure, frequent phrases) rather than the semantic decision boundary. This makes naive LM fine-tuning misleading: low loss does not imply task understanding.

The central lesson is that **loss design and supervision structure matter more than model choice**.

---

### Why Not Just Use an Encoder?

From an industrial standpoint, an encoder-based model would likely achieve:
- higher raw classification accuracy,
- faster convergence,
- simpler training dynamics.

However, the goal of this project is **not to beat SOTA**, but to explore **learning behavior and controllability** in generative models. By training a decoder-only model to jointly perform **relevance classification and structured reasoning**, this project exposes failure modes, trade-offs, and design patterns that are invisible in encoder-only pipelines.

In particular, the ability to generate **transparent, inspectable reasoning** alongside a decision is a key advantage for research, debugging, and RAG evaluation.

---

### The Core Insight: Ensure the Model Learns the Actual Task

The most important takeaway is this:

> **A LM doesn’t learn the task itself—it learns whatever behavior minimizes the loss.**

If the task signal is sparse, entangled, or dominated by easier tokens, the model will ignore it. The two-stage LoRA strategy works not because it is complex, but because it:
- isolates decision-making from explanation,
- constrains gradient flow to the intended outputs,
- enforces a clear semantic contract at each stage.

This principle generalizes and applies to any structured generation task involving decision-making.
