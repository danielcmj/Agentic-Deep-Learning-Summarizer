# Agentic Deep Learning Summarizer 🧠🤖

An end-to-end Natural Language Processing (NLP) pipeline that fine-tunes a Deep Learning Transformer model for text summarization, wrapped in an autonomous AI Agent workflow for automated fact-checking and verification.



## 🎯 Project Overview
In information-dense fields like medicine, law, and news, professionals struggle with information overload. This project solves that by building a **Self-Verifying AI Pipeline**. 
1. **Deep Learning (Information Distillation):** A local `facebook/bart-base` model is fine-tuned to compress long-form text into concise abstracts.
2. **Agentic Workflow (Trust & Safety):** A `smolagents` reasoning engine takes the generated summary, autonomously searches the live internet, and verifies the AI's claims against real-world data to prevent hallucinations.

## 🛠️ Tech Stack & Hardware
* **Frameworks:** PyTorch, Hugging Face (`transformers`, `datasets`, `smolagents`, `evaluate`)
* **Architecture:** Encoder-Decoder Transformer (BART), LLM-based Reasoning Agent
* **Compute Optimization:** NVIDIA T4 GPU, Mixed Precision Training (`fp16=True`)



## 🏗️ Architecture & PACE Methodology
This project strictly follows the **PACE (Plan, Analyze, Construct, Execute)** framework:

* **Analyze:** Conducted EDA on the text corpus to map token length distributions, determining a `max_length=1024` context window constraint for the Transformer. Implemented robust `try/except` fallback logic to bypass deprecated Hugging Face dataset scripts, shifting from PubMed to CNN/DailyMail parquet shards.
* **Construct:** Tokenized inputs using the modern API. Fine-tuned the BART model using the `Seq2SeqTrainer` optimized for GPU VRAM via half-precision floating-point math.
* **Execute (Inference):** Deployed **Beam Search** (`num_beams=4`) during inference to explore multiple word paths simultaneously, minimizing repetitive text generation.
* **Agentic Verification:** Integrated Hugging Face's `smolagents` to equip a cloud-based inference model with a `DuckDuckGoSearchTool`. The agent autonomously parses the local BART model's summary, writes Python code to query the web, and outputs a factual verification report.

## 📊 Evaluation Metrics
The deep learning model's fidelity was quantified using **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) metrics, measuring the n-gram and longest common subsequence overlap against human-written abstracts. 

* *Result:* The local model successfully reduced text volume by >90% while preserving core entities. The autonomous Research Agent independently verified the historical accuracy of the generated summary against live web data.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies (ensure `huggingface-hub` is strictly managed to prevent conflicts with `smolagents`):
   ```bash
   pip install transformers[torch] datasets evaluate rouge_score accelerate "smolagents[toolkit]" "huggingface_hub<1.0.0"
