#  Multi-Modal Search & Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" />
  <img src="https://img.shields.io/badge/Streamlit-Live-red" />
  <img src="https://img.shields.io/badge/NLP-SentenceTransformers-green" />
  <img src="https://img.shields.io/badge/Recommender-SVD-orange" />
  <img src="https://img.shields.io/badge/Status-Deployed-success" />
</p>

<p align="center">
  <b>Semantic Search • Personalized Recommendations • Interactive ML Visualization</b>
</p>

<p align="center">
  <a href="https://searchnrec.streamlit.app/"> Live Demo</a>
</p>

---

##  Live Demo

**Website:** https://searchnrec.streamlit.app/

This project combines modern NLP-based semantic search with collaborative filtering to simulate the discovery architecture used in large-scale e-commerce platforms.

---

## Project Overview
<img width="253" height="533" alt="image" src="https://github.com/user-attachments/assets/6b8a41cb-f415-4568-8cac-af5886dece2f" />


Traditional e-commerce search relies heavily on keyword matching, which often fails to capture user intent.

This system solves that problem using two independent machine learning pipelines:

### Semantic Search Tower

Converts product titles and descriptions into dense vector embeddings using Sentence Transformers and retrieves products based on meaning rather than exact keywords.

### Recommendation Tower

Uses Singular Value Decomposition (SVD) on user-item interactions to learn hidden user preferences and generate personalized recommendations.

###  Interactive Dashboard

Built with Streamlit and includes visual explanations of:

* Embedding vectors
* Matrix dimensions
* Similarity computations
* SVD decomposition
* Recommendation generation

---

##  Architecture

```text
                    USER QUERY
                         │
                         ▼
              Sentence Transformer
             (all-MiniLM-L6-v2)
                         │
                         ▼
                 Query Embedding
                         │
                         ▼
                 Cosine Similarity
                         │
                         ▼
                  Top-K Products



         USER HISTORY / RATINGS
                         │
                         ▼
                 User-Item Matrix
                         │
                         ▼
                SVD Factorization
                         │
                         ▼
            Personalized Recommendations
```

---

##  Project Statistics

| Metric               | Value |
| -------------------- | ----- |
| Products             | 200   |
| Users                | 50    |
| Ratings              | 854   |
| Sparsity             | 91.4% |
| Embedding Dimensions | 384   |
| Latent Factors       | 10    |
| RMSE                 | 0.69  |

---

## Features

* Semantic Product Search
* Transformer-Based Embeddings
* Dense Vector Retrieval
* Cosine Similarity Ranking
* SVD Matrix Factorization
* Personalized Recommendations
* Interactive Streamlit Dashboard
* Real-Time ML Architecture Visualization
* Reproducible Synthetic Dataset Generation

---

## Machine Learning Pipeline

### 1️. Semantic Search

Product metadata is merged:

```python
text = title + " " + description
```

The combined text is encoded using:

```python
all-MiniLM-L6-v2
```

Resulting embedding matrix:

```text
E ∈ ℝ(N × 384)
```

where:

* N = number of products
* 384 = embedding dimensions

Retrieval is performed using cosine similarity.

---

### 2️. Recommendation Engine

A sparse user-item interaction matrix is generated:

```text
R ∈ ℝ(M × N)
```

where:

* M = users
* N = products

The matrix is mean-centered and decomposed using:

```text
R ≈ UΣVᵀ
```

where:

* U = user latent factors
* Σ = singular values
* Vᵀ = item latent factors

The reconstructed matrix predicts unseen user preferences.

---

##  Project Structure

```text
.
├── app.py
├── search_engine.py
├── recommender.py
├── generate_data.py
├── products.csv
├── ratings.csv
├── requirements.txt
└── README.md
```

---

##  Tech Stack

### Machine Learning

* Sentence Transformers
* PyTorch
* NumPy
* Pandas
* SciPy

### Recommendation Systems

* Matrix Factorization
* Singular Value Decomposition (SVD)

### Frontend

* Streamlit

### Backend

* Python

---

##  Local Setup

### Clone Repository

```bash
git clone https://github.com/mitalikakhosla25/amazon-ml-rec-engine.git
cd amazon-ml-rec-engine
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Dataset

```bash
python generate_data.py
```

### Run Search Engine

```bash
python search_engine.py
```

### Run Recommender

```bash
python recommender.py
```

### Launch Dashboard

```bash
streamlit run app.py
```

---
