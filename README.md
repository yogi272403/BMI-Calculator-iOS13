## R2L Lab Onboarding: Part 2 - Implementation Challenge

This repository contains my submission for Part 2 of the R2L Lab onboarding quiz. It implements both a sparse (BM25) and dense (bi-encoder + FAISS) retrieval pipeline evaluated on the BEIR SciFact dataset.

## Repository Structure

* `download_data.py`: Script to fetch and unpack the SciFact dataset using the `beir` library.
* `sparse_retriever.py`: Implements the BM25 retrieval pipeline using `rank-bm25`.
* `dense_retriever.py`: Implements the dense semantic search pipeline using `sentence-transformers` and `faiss-cpu`.
* `requirements.txt`: Python dependencies.
* `README.md`: Project documentation and analysis.

## Setup & Execution

### 1. Environment Setup
It is recommended to use a Python virtual environment (Python 3.8+ required).

```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Download the Data
Run the ingestion script. This will download the dataset and place it in a local `datasets/scifact` directory.

```bash
python download_data.py
```

### 3. Run the Retrievers
Execute both scripts to generate the `.json` result files required for evaluation. The scripts will automatically create a `results/` directory and place the outputs there.

```bash
python sparse_retriever.py
python dense_retriever.py
```

---

## Discussion & Results

After generating the results, both pipelines were evaluated using the provided `evaluation.py` script against the SciFact test split. 

Here are the empirical metrics observed:

| Metric | Sparse (BM25) | Dense (all-MiniLM-L6-v2) |
| :--- | :--- | :--- |
| **nDCG@10** | 0.5597 | 0.6450 |
| **Recall@100** | 0.7929 | 0.9250 |
| **MAP@10** | 0.5147 | 0.5959 |

### Which retriever performed better and why?

The dense retriever clearly outperformed the sparse retriever across all metrics. The most significant improvement was in **Recall@100**, which increased from ~79.3% to 92.5%. 

This performance gap is largely due to the nature of the SciFact dataset. The task involves verifying scientific claims, meaning the queries are often natural language assertions that paraphrase the source material. BM25 relies entirely on exact lexical matching (TF-IDF). If a query uses a synonym or rephrases a concept, BM25 struggles to establish relevance. 

The dense retriever (`all-MiniLM-L6-v2`) overcomes this limitation by mapping both the queries and the corpus into a shared high-dimensional semantic vector space. It does not require exact word overlap; instead, it matches based on contextual meaning, allowing it to successfully retrieve relevant documents much more consistently.

### Performance Trade-offs

While dense retrieval achieved higher retrieval quality, there are clear engineering trade-offs between the two approaches:

1. **Compute & Speed:** Building the BM25 index took only a fraction of a second on a standard CPU. In contrast, passing the entire SciFact corpus through the transformer model to generate dense embeddings took several minutes. While FAISS makes the actual vector search incredibly fast, the initial indexing phase for dense retrieval is heavily compute-bound (and ideally requires GPU acceleration in a production environment).
2. **Memory Footprint:** BM25 uses a lightweight inverted index. The dense approach requires storing high-dimensional vectors (384 dimensions of `float32` per document) in memory via FAISS. For SciFact, this is manageable, but for an enterprise-scale corpus, this memory footprint becomes a major bottleneck without aggressive vector quantization (e.g., PQ or IVF).
3. **Retrieval Quality vs. Domain:** Dense retrieval dominated in this semantic task, but it can struggle out-of-domain if the corpus contains highly specific, unseen jargon. BM25 remains superior if a user is searching for a highly specific, exact string (like a unique alphanumeric identifier or chemical compound) where semantic similarity is not the primary goal.
```
