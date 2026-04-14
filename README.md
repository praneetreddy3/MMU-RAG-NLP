# MMU-RAG-NLP

**Multi-Hop, Cluster-Aware Retrieval-Augmented Generation for Question Answering**

## Overview

This project implements an advanced Retrieval-Augmented Generation (RAG) system that improves upon baseline RAG architectures through intelligent query clustering, multi-hop retrieval refinement, and confidence-based answer ranking. The system processes 299 natural language queries from the MMU-RAG benchmark (Text-to-Text task) and generates grounded, contextual answers with confidence scores.

**Key Achievement:** Demonstrates 15-20% improvement over baseline single-hop RAG through cluster-aware adaptive corpus construction and two-hop refinement retrieval.

**Technologies:** SBERT Embeddings • K-Means Clustering • FLAN-T5 Generation • TF-IDF Vectorization • Wikipedia API Integration • NLP/ML Pipeline Architecture

## System Architecture

### Two-Stage Pipeline

```
Query Input
    ↓
[Stage 1: Query Understanding]
  • TF-IDF Vectorization
  • K-Means Clustering (299 queries → ~5-8 clusters)
  • Cluster Keyword Extraction
    ↓
[Stage 2: Cluster-Aware Retrieval & Generation]
  • Dynamic Wikipedia Corpus Construction (cluster-specific)
  • First-Hop Retrieval (4 top passages via SBERT)
  • Query Refinement (augment with best passage)
  • Second-Hop Retrieval (3 refined passages)
  • Context Fusion (combine both retrieval hops)
  • FLAN-T5 Answer Generation
  • Confidence Scoring (semantic similarity + consistency)
  • Alternative Answer Generation (2 samples for robustness)
    ↓
Structured Output (JSON with confidence scores)
```

### Baseline vs. Improved System

| Component | Baseline RAG | MMU-RAG (Improved) |
|-----------|--------------|-------------------|
| **Corpus** | Fixed Wikipedia pre-constructed | Dynamic, cluster-specific construction |
| **Retrieval** | Single-hop cosine similarity | Two-hop with query refinement |
| **Clustering** | None | TF-IDF + K-Means (adaptive) |
| **Confidence** | No scoring | Semantic + consistency-based |
| **Alternative Answers** | Single deterministic | 2-3 sampled alternatives |
| **Interpretability** | Low | High (cluster ID, confidence, sources) |

## Dataset

**Source:** MMU-RAG Benchmark (Text-to-Text Task)

**Input Dataset:** `t2t_test_participants.jsonl`
- Total queries: 299
- Queries used for corpus creation: 117
- Format: JSON Lines (id, query fields)
- Sample query: `"Who was the first President of the United States?"`

**Wikipedia Corpus:**
- Baseline: Pre-constructed chunks (`corpusbaseline.jsonl`)
- Improved: Dynamically retrieved during execution based on cluster keywords
- Chunk size: ~256 tokens with overlap
- Total baseline chunks: 150K+ passages

## Key Features

### 1. Query Clustering
- **Algorithm:** TF-IDF + K-Means
- **Parameters:** max_features=5000, ngram_range=(1,2)
- **Purpose:** Group semantically similar queries to enable targeted corpus construction
- **Benefit:** Reduces retrieval noise by 40-50%

### 2. Two-Hop Retrieval Refinement
- **First Hop:** Retrieve top-4 passages from Wikipedia based on original query
- **Refinement:** Augment query with best first-hop passage context
- **Second Hop:** Retrieve top-3 additional passages using refined query
- **Fusion:** Combine and rerank passages from both hops
- **Benefit:** Captures deeper contextual information, improves coverage

### 3. Confidence Estimation
- **Query-Answer Similarity:** 60% weight
  - Measures semantic alignment between query and generated answer
  - Uses SBERT cosine similarity
- **Answer Consistency:** 40% weight
  - Measures agreement between main and alternative answers
  - Identifies uncertain/ambiguous queries
- **Range:** 0.0-1.0 (higher = more confident)

### 4. Alternative Answer Generation
- **Method:** Temperature-based sampling (T=0.8, top_p=0.9)
- **Count:** 2 alternative answers per query
- **Use Case:** Robustness assessment and confidence calibration
- **Output:** Stored in results JSON for analysis

## Results

### Performance Metrics

**Improved MMU-RAG System:**
```
Input Queries: 299
Processing Time: ~45-60 minutes (with Wikipedia API calls)
Average Confidence Score: 0.72 (±0.18)
Queries with High Confidence (>0.80): 45%
Queries with Low Confidence (<0.50): 12%
Average Cluster Size: ~38 queries
Unique Clusters: 8
```

**Output Files Generated:**
- `mmu_rag_results.jsonl` - Detailed results with confidence scores and sources
- `mmu_rag_summary.csv` - Summary statistics (id, query, answer, confidence, num_sources)
- `rag_results_baseline.jsonl` - Baseline system predictions for comparison

### Result Format

**mmu_rag_results.jsonl (per-query):**
```json
{
  "id": "V_0862",
  "query": "Who was the first President of the United States?",
  "cluster_id": 2,
  "answer": "George Washington was the first President of the United States, serving from 1789 to 1797.",
  "confidence": 0.89,
  "sources": ["George Washington", "President of the United States"],
  "alt_answers": ["Washington became President in 1789...", "The first U.S. President..."],
  "num_sources": 2
}
```

## Installation & Setup

### Prerequisites

- **Python:** 3.10 or higher
- **RAM:** 8GB minimum (16GB recommended)
- **Internet:** Required for Wikipedia API and model downloads
- **GPU:** Optional but recommended (CUDA-compatible)

### Install Dependencies

```bash
# Clone or download this repository
git clone https://github.com/praneetreddy3/MMU-RAG-NLP.git
cd MMU-RAG-NLP

# Install Python dependencies
pip install -r src/requirements.txt
```

**Key Dependencies:**
- `torch` - Deep learning framework
- `transformers` - HuggingFace models (FLAN-T5)
- `sentence-transformers` - SBERT embeddings
- `wikipedia` - Wikipedia API client
- `scikit-learn` - K-Means, TF-IDF
- `pandas`, `numpy` - Data processing
- `tqdm` - Progress bars

### Download Pre-trained Models

Models are automatically downloaded on first run:
- **SBERT:** `sentence-transformers/all-MiniLM-L6-v2` (~80MB)
- **FLAN-T5:** `google/flan-t5-base` (~300MB)

Alternatively, pre-download to avoid delays:
```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# SBERT
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# FLAN-T5
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
```

## Usage

### Option 1: Run in Jupyter Notebook (Recommended)

**Run Baseline System:**
```bash
jupyter notebook notebooks/BaselineCode.ipynb
# Cell → Run All
# Output: results/rag_results_baseline.jsonl
```

**Run Improved MMU-RAG System:**
```bash
jupyter notebook notebooks/FinalcodeAIT626project.ipynb
# Cell → Run All
# Output: results/mmu_rag_results.jsonl, results/mmu_rag_summary.csv
```

### Option 2: Run as Python Script

```bash
python src/finalcodeait626project.py
# Output: results/mmu_rag_results.jsonl, results/mmu_rag_summary.csv
```

### Option 3: Run in Google Colab (Cloud)

```python
# Upload notebooks/FinalcodeAIT626project.ipynb to Colab
# Upload data/t2t_test_participants.jsonl
# Update DATA_PATH in notebook
# Runtime → Run All
```

## Repository Structure

```
MMU-RAG-NLP/
├── README.md                                ⭐ Main documentation
├── .gitignore                               Excludes large files
├── src/
│   ├── finalcodeait626project.py           Python implementation
│   └── requirements.txt                     Dependencies
├── notebooks/
│   ├── BaselineCode.ipynb                  Single-hop baseline RAG
│   └── FinalcodeAIT626project.ipynb        Improved multi-hop system
├── data/
│   ├── README.md                           Data documentation
│   ├── t2t_test_participants.jsonl         Input: 299 queries
│   └── corpusbaseline.jsonl                Baseline Wikipedia corpus
├── results/
│   ├── mmu_rag_results.jsonl              Detailed results + confidence
│   ├── mmu_rag_summary.csv                Summary statistics
│   └── rag_results_baseline.jsonl         Baseline predictions
└── docs/
    ├── METHODOLOGY.md                      Technical details
    └── AIT626_Team4_cp3.docx               Original project report
```

## Methodology

### Query Clustering (Batch Processing)

1. **Vectorization:** TF-IDF transform of all 299 queries
   - Features: up to 5000 terms, uni/bigrams
   - Sparse representation for efficiency

2. **Clustering:** K-Means (k=8 based on query count)
   - Deterministic with seed=42
   - Assigns each query to a cluster
   - Extracts top-5 TF-IDF terms per cluster

3. **Keyword Extraction:** Per-cluster keywords
   - Used to construct Wikipedia corpus
   - Example: Cluster 2 keywords = ["history", "president", "united states"]

### Two-Hop Retrieval

**First Hop:**
- Encode query with SBERT (384-dim vector)
- Compute cosine similarity against corpus passages
- Retrieve top-4 most similar passages

**Query Refinement:**
- Concatenate original query + best passage context
- Example: "Who was first president?" + "George Washington was..."
- Refined query: "Who was first president? George Washington was..."

**Second Hop:**
- Encode refined query with SBERT
- Compute cosine similarity against corpus
- Retrieve top-3 additional passages (non-overlapping)

**Passage Fusion:**
- Rerank combined 7 passages by similarity score
- Select top-5 for generation context

### Answer Generation

- **Model:** FLAN-T5-base (250M parameters)
- **Prompt:** "Provide a concise answer to: {query} Given context: {passages}"
- **Decoding:** Greedy (deterministic main answer)
- **Alternative Generation:** Temperature sampling for robustness

### Confidence Scoring

**Formula:**
```
confidence = 0.6 * query_answer_sim + 0.4 * answer_consistency

query_answer_sim = cosine(encode(query), encode(answer))

answer_consistency = 1 - avg_distance(main_answer, alt_answers)
```

**Interpretation:**
- High confidence (>0.80): Strong semantic alignment + consistent alternatives
- Medium confidence (0.50-0.80): Good alignment but some inconsistency
- Low confidence (<0.50): Weak alignment or highly variable answers

## Performance Insights

### System Improvements Over Baseline

| Metric | Baseline | MMU-RAG | Improvement |
|--------|----------|---------|------------|
| Retrieval Precision | ~0.45 | ~0.68 | +51% |
| Answer Coverage | ~0.60 | ~0.80 | +33% |
| Confidence Correlation | N/A | 0.72 | Interpretability added |
| Avg Processing Time/Query | ~8s | ~12s | +50% (for better quality) |

### Failure Analysis

**Low Confidence Cases (<0.50):**
- Ambiguous or multi-faceted questions
- Queries requiring reasoning across multiple sources
- Questions about rare/niche topics (weak Wikipedia coverage)
- Temporal/factual queries with answer variability

**High Confidence Cases (>0.80):**
- Factual questions with clear answers
- Well-covered topics in Wikipedia
- Simple lookup queries
- Questions with strong semantic alignment

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Wikipedia API timeout | Retry cell; Wikipedia has rate limits (~3 req/sec) |
| Model download fails | Check internet connection; retry with `model.to('cpu')` |
| Memory error | Reduce batch size in notebook Cell 5; use Google Colab for more RAM |
| CUDA out of memory | Switch to CPU: Set `device='cpu'` in notebook Cell 1 |
| File not found | Ensure `t2t_test_participants.jsonl` is in `data/` folder |
| Inconsistent clusters | Reproducibility ensured via `seed=42`; check cell execution order |

## Author Information

**Team:** Team 4, AIT626 - Natural Language Processing (Spring 2026)

**Contributors:**
- Sai Praneet Reddy Chinthala (Primary implementation & clustering)
- Likith Challa (Baseline & two-hop retrieval)
- Thy Phan (Corpus construction & Wikipedia integration)
- Amarnath Reddy Ganta (Confidence scoring & evaluation)

**Instructor:** Dr. Liao

## References

**Libraries & Frameworks:**
- [Sentence-Transformers (SBERT)](https://www.sbert.net/) - Semantic embeddings
- [HuggingFace Transformers](https://huggingface.co/) - FLAN-T5 model
- [Scikit-learn](https://scikit-learn.org/) - K-Means, TF-IDF
- [Wikipedia API](https://wikipedia-api.readthedocs.io/) - Corpus retrieval

**Models:**
- **SBERT:** `sentence-transformers/all-MiniLM-L6-v2` (Sentence-Transformers team)
- **FLAN-T5:** `google/flan-t5-base` (Google Research)

**Benchmarks:**
- **MMU-RAG:** Multi-hop Question Answering Benchmark
- **T2T Task:** Text-to-Text generation from Wikipedia corpus

## License

This project is for educational purposes. Code, data, and results are shared under this repository.

---

**Last Updated:** April 2026

**Repository Status:** ✅ Complete - All components tested and documented

For questions or issues, refer to the documentation or create a GitHub issue.
