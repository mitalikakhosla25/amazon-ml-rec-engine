#  Multi-Modal Search & Recommendation System

<p align="center">
  <b>Semantic Search • Personalized Recommendations • Interactive ML Visualization</b>
</p>

##  Live Demo
**Website:** https://searchnrec.streamlit.app/

An end-to-end e-commerce discovery engine that combines dense vector embeddings for semantic search with collaborative filtering via matrix factorization for personalized user feeds. This architecture replicates the industry-standard Two-Tower retrieval and ranking framework used by modern large-scale marketplace platforms.

---

## Architecture Overview

This project bypasses generic string-matching (keyword search) and naive recommendation heuristics by implementing two independent machine learning pipelines:

1. **Semantic Search Tower (NLP):** Encodes raw textual metadata (product titles and descriptions) into continuous vector spaces using a deep Transformer network, running vectorized similarity lookups over a normalized embedding matrix.
2. **Personalization Tower (Collaborative Filtering):** Leverages Matrix Factorization on a sparse user-item interaction matrix, isolating pure consumer preferences from rating inflation scale biases via explicit mean-centering.

---

##  Deep-Dive Technical Pipeline

### 1. Candidate Retrieval & Dense Vector Search
Standard search engines rely on keyword overlap (e.g., SQL `LIKE` or basic BM25). If a customer searches for *"something to read at night"*, keyword lookups fail to find items missing those exact tokens.

This pipeline maps strings into a geometric semantic space:
* **Feature Fusion:** Product text is compiled via $x_{\text{feat}} = \text{Title} + \text{ " " } + \text{Description}$ to preserve attention contextual relationships across fields.
* **Vector Encoding:** A pre-trained `all-MiniLM-L6-v2` Bi-Encoder model transforms $x_{\text{feat}}$ into a dense embedding matrix $\mathbf{E} \in \mathbb{R}^{N \times D}$, where $N=200$ products and $D=384$ semantic dimensions.
* **Vectorized Cosine Similarity:** To maximize inference execution speed, rows of $\mathbf{E}$ are pre-normalized to unit vectors ($L_2$ norm $\approx 1$). When a natural language query string is processed, it is mapped to query vector $\mathbf{q} \in \mathbb{R}^{D}$. The similarity score vector $\mathbf{s}$ is resolved concurrently using a single highly optimized BLAS matrix-vector dot product operation:

$$\mathbf{s} = \mathbf{E}\mathbf{q}$$

* **Time Complexity Optimization:** Instead of fully sorting scores using a naive $\mathcal{O}(N \log N)$ algorithm, the backend uses `np.argpartition` to isolate the top-K elements in linear time $\mathcal{O}(N + K \log K)$, replicating production ANN scale constraints.

### 2. SVD Matrix Factorization Recommender
The left sidebar personalization channel targets behavioral user patterns via user-item implicit ratings matrices.
* **Matrix Pivoting:** A sparse explicit interaction matrix $\mathbf{R} \in \mathbb{R}^{M \times N}$ ($M=50$ users, $N=198$ rated products) is constructed, exhibiting a baseline marketplace sparsity of $91.4\%$.
* **Mean Centering (Bias Regularization):** To mitigate user scale biases (critical users rating items between 1-3★ vs. generous users rating 4-5★), personal mean scores are calculated per user row and subtracted:

$$\mathbf{R}_{\text{centred}} = \mathbf{R} - \mu_{\text{user}}$$

* **Singular Value Decomposition (SVD):** The centered matrix is decomposed into low-rank representations capturing the top $k=10$ latent affinity spaces using the ARPACK Lanczos solver algorithm:

$$\mathbf{R}_{\text{centred}} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where $\mathbf{U} \in \mathbb{R}^{50 \times 10}$ dictates User Latent Factors, $\mathbf{V}^T \in \mathbb{R}^{10 \times 198}$ dictates Item Latent Factors, and $\mathbf{Sigma}$ balances factor magnitudes. 
* **Matrix Reconstruction & Masking:** Latent factor representations are reconstructed through vectorized broadcasting. Before exposing top picks to the customer dashboard, items previously purchased or rated by the active user are explicitly masked to ensure zero prediction leakage.

---

##  Dataset Specifications & Simulation

The platform includes a built-in synthetic pipeline (`generate_data.py`) configured with seed tracking to produce fully reproducible commercial simulations:
* **Product Catalog (`products.csv`):** 200 items spread evenly across four distinct retail pillars: *Electronics, Books, Apparel, and Home*.
* **Behavior Matrix (`ratings.csv`):** 854 user engagements capturing structured cluster tendencies. For instance, specific users are generated with skewed weight factors towards individual groups (e.g., User 03 is seeded within the high-affinity *Electronics Cluster*). SVD validation metrics prove the algorithms successfully discover these mathematical anomalies, reaching an internal reconstruction evaluation RMSE of `0.69`.

---

##  Tech Stack & Engineering Practices

* **Frameworks & Core ML:** Python, PyTorch, Sentence-Transformers, SciPy (ARPACK), NumPy, Pandas.
* **User Interface & Serving:** Streamlit Dashboard Engineering.
* **Optimization Systems:** Application runtime leverages specific object caching (`st.cache_resource` for static Transformer weights and SVD factor memory buffers) to bypass data-load processing stalls during interface state reruns.
* **Fault Tolerant Fallback Control:** Integrates architectural safe-degradation paths. If the deep learning dependency environment fails or lacks GPU/network resource execution capabilities, the candidate retriever gracefully falls back to basic Regex keyword matching patterns without crashing the UI.

---

##  Setup & Local Execution

1. Install dependencies

bash
pip install numpy scipy pandas sentence-transformers streamlit

2. Generate the datasets

bash
python generate_data.py

Output confirms 200 products and 854 ratings with a reproducible random seed.

3. Verify the backend modules

bash
python search_engine.py    # encodes 200 products, runs 4 test queries
python recommender.py      # runs SVD, prints recommendations for 5 user profiles


Note: On first run, search_engine.py will download and cache the all-MiniLM-L6-v2 weights (~90 MB) from HuggingFace. Subsequent runs load from the local cache with no network access required.



4. Launch the dashboard

bashstreamlit run app.py

Opens at http://localhost:8501. The sidebar simulates user login (select any User ID 1–50) and displays personalised SVD recommendations. The main panel accepts free-text queries and returns the top-5 semantically similar products with cosine similarity scores and an expandable linear-algebra explainer.
