# 🖼️ Advanced Image Captioning with PaLI-Gemma, BLIP, and RAG

This project investigates different methods for **image captioning**, combining both **vision-language generation** and **retrieval-augmented generation (RAG)** techniques. After testing multiple state-of-the-art models, we observed that **Google's `google/paligemma-3b-pt-224`** model provided the best results. The project also includes experiments with retrieval-based caption refinement.

---

## 🔬 Models Tried & Benchmarked

| Model Name                                 | Description                                          | Quality     |
|-------------------------------------------|------------------------------------------------------|-------------|
| `Salesforce/blip-image-captioning-base`   | Lightweight version of BLIP for fast generation.     | Medium      |
| `Salesforce/blip-image-captioning-large`  | Larger BLIP model for more expressive outputs.       | Good        |
| `google/paligemma2-3b-mix-224`            | PaLI-Gemma v2 pretrained on mixed tasks.             | Decent      |
| **`google/paligemma-3b-pt-224`**          | Fine-tuned for captioning. High coherence & detail.  | ⭐ Best      |

✅ **Best overall performance was achieved with `google/paligemma-3b-pt-224`.**

---

## 🔁 RAG (Retrieval-Augmented Generation) Attempt with BLIP + GTE-small + T5

As part of the exploration, we implemented a **RAG-style captioning system** to assess whether retrieval could improve caption quality.

### ✨ RAG Flow

1. A **base caption** is generated from a test image using a fine-tuned BLIP model (`blip1-finetuned`).
2. This caption is embedded using **`thenlper/gte-small`** (SentenceTransformer).
3. The top 3 semantically similar captions are retrieved from a **train set corpus**.
4. The **FLAN-T5-base** model receives the base caption + top-3 retrieved captions and refines the original output through a structured prompt.

### 📌 Purpose

This approach aimed to improve fluency, specificity, and alignment with domain-specific phrasing in training data. It followed the Retrieval-Augmented Generation paradigm:

> `"image ➝ base caption (BLIP) ➝ retrieve similar captions (GTE-small) ➝ refine via prompt (FLAN-T5)"`

### ⚠️ Observation

Despite its structured approach, this **RAG setup did not significantly outperform the direct outputs of PaLI-Gemma** in terms of caption clarity or semantic alignment. The improvement was minimal, and in some cases, even introduced unnecessary changes. Thus, it was not used in the final pipeline.

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `submission.ipynb` | PaLI-Gemma-based final pipeline |
| `image_captioning.ipynb` | RAG-based prototype with BLIP + GTE + T5 |
| `submission.csv` | Final caption predictions |

---

## 🛠️ Installation

```bash
pip install transformers sentence-transformers pandas tqdm pillow
