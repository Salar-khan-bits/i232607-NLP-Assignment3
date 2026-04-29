# CS-4063 NLP — Assignment 3
### Transformer-based Review Understanding with RAG
**Student ID:** i232607 &nbsp;|&nbsp; **Course:** CS-4063 Natural Language Processing &nbsp;|&nbsp; **FAST NUCES**

---

## What This Is

A three-stage NLP pipeline built entirely from scratch — no pretrained models, no `nn.Transformer`, nothing abstracted away. Raw PyTorch only.

The system reads Amazon product reviews, understands their sentiment, finds similar reviews from a database, and then writes a short explanation for why the review feels the way it does.

---

## Pipeline at a Glance

```
Review Text
    │
    ▼
┌─────────────────────────────┐
│   Encoder-only Transformer  │  → Sentiment (Neg / Neu / Pos)
│        (Part A)             │  → Category  (Beauty / Phones / Sports)
│                             │  → 128-dim review vector
└─────────────┬───────────────┘
              │ vector
              ▼
┌─────────────────────────────┐
│     Retrieval Module        │  Cosine similarity search
│        (Part B)             │  Top-3 similar training reviews
└─────────────┬───────────────┘
              │ context
              ▼
┌─────────────────────────────┐
│  Decoder-only Transformer   │  Generates 1-2 sentence explanation
│        (Part C)             │  conditioned on review + labels + context
└─────────────────────────────┘
```

---

## Dataset

Amazon Reviews across 3 product categories — 10,000 reviews each, 30,000 total.

| Category | File |
|---|---|
| Beauty | `Beauty_5.json` |
| Cell Phones | `Cell_Phones_and_Accessories_5.json` |
| Sports | `Sports_and_Outdoors_5.json` |

Split: **70% train · 15% val · 15% test**

Place all three `.json` files inside a `dataset/` folder before running.

---

## Project Structure

```
├── dataset/                  
├── models/
│   ├── encoder.pt
│   ├── decoder_with_retrieval.pt
│   └── decoder_without_retrieval.pt
├── results/
│   ├── embeddings.pt
│   ├── retrieval.pt
│   ├── prep_data.pt
│   └── explanation_outputs.pt
├── i232607-NLP-Assignment2.ipynb
└── README.md
```

---

## How to Run

```bash
pip install torch numpy scikit-learn matplotlib
```

Open `i232607-NLP-Assignment2.ipynb` and run all cells top to bottom. That's it.

The notebook runs in 4 sections in order — dataset → encoder → retrieval → decoder. Each section saves its outputs so you can restart from any point without rerunning everything.

---

## Model Specs

| Component | Detail |
|---|---|
| Embedding dim | 128 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Feed-forward dim | 256 |
| Dropout | 0.1 |
| Encoder optimizer | Adam + StepLR (step=2, γ=0.5) |
| Encoder epochs | 6, batch 64 |
| Decoder optimizer | Adam lr=1e-3 |
| Decoder epochs | 5, batch 32 |
| Max sequence length | 120 (encoder) · 220 (decoder) |
| Retrieved reviews k | 3 |

---

## Restrictions Followed

- ❌ `nn.Transformer`
- ❌ `nn.MultiheadAttention`
- ❌ `nn.TransformerEncoder`
- ❌ Any pretrained model (BERT, GPT, T5, etc.)
- ✅ Everything built from scratch with `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`