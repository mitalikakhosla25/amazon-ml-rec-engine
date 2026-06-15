"""
search_engine.py
----------------
Semantic Search backend for the Amazon-Style Multi-Modal
Search & Recommendation Engine.

Architecture Overview
---------------------
Text → SentenceTransformer encoder → L2-normalised dense embeddings
Query → same encoder → L2-normalised query vector
Retrieval → cosine similarity via matrix dot product (pure NumPy)

All matrix shapes are annotated inline so the linear-algebra
mechanics are explicit and reviewable.

Dependencies
------------
    pip install sentence-transformers pandas numpy
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────
PRODUCTS_FILE  = "products.csv"
MODEL_NAME     = "all-MiniLM-L6-v2"   # 384-dim embeddings, fast & accurate
EMBED_DIM      = 384                   # output dimension of the chosen model


# ── 1. Load product catalogue ──────────────────────────────────────────────────

def load_products(path: str = PRODUCTS_FILE) -> pd.DataFrame:
    """
    Load products.csv and engineer the text feature that will be encoded.

    Combined field rationale
    ------------------------
    Concatenating title + description gives the encoder both the compact
    product label (high-signal, dense) and the long-form feature text
    (richer semantic context).  This single string is what we embed.
    """
    df = pd.read_csv(path)

    # ── Text feature engineering ───────────────────────────────────────────────
    # Shape of resulting column: (N,) where N = number of products (200)
    df["text_feature"] = df["title"] + " " + df["description"]

    print(f"[load_products] Loaded {len(df)} products from '{path}'.")
    print(f"[load_products] Columns: {list(df.columns)}")
    return df


# ── 2. Build the embedding matrix ─────────────────────────────────────────────

def build_embedding_matrix(
    texts: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Encode a list of N text strings into a normalised embedding matrix.

    Parameters
    ----------
    texts  : list of N raw text strings
    model  : loaded SentenceTransformer instance

    Returns
    -------
    E : np.ndarray, shape (N, D)
        Row i is the L2-normalised embedding of texts[i].
        N = number of products (200), D = embedding dimension (384).

    Linear-algebra notes
    --------------------
    Raw encoder output  →  shape (N, D)   e.g. (200, 384)

    L2 normalisation per row:
        ||v||₂  = sqrt( Σ vⱼ² )               scalar per row
        v̂       = v / ||v||₂                   unit vector in R^D

    After normalisation every row vector satisfies  ||E[i]||₂ = 1.
    This is the prerequisite that makes the dot product equivalent to
    cosine similarity (proven below in semantic_search).
    """
    print(f"\n[build_embedding_matrix] Encoding {len(texts)} documents …")
    print(f"[build_embedding_matrix] Model: '{MODEL_NAME}'  |  embed dim: {EMBED_DIM}")

    # Raw embeddings — shape: (N, D)  →  (200, 384)
    raw_embeddings: np.ndarray = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"[build_embedding_matrix] Raw matrix shape: {raw_embeddings.shape}")
    # Expected: (200, 384)

    # ── L2 normalisation ───────────────────────────────────────────────────────
    # Compute per-row L2 norms → shape: (N, 1)  →  (200, 1)
    norms: np.ndarray = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    # keepdims=True preserves the column dimension so broadcasting works:
    #   (200, 384) / (200, 1)  →  (200, 384)  ✓

    # Guard against the (practically impossible) zero-norm edge case
    norms = np.where(norms == 0, 1.0, norms)

    # Normalised matrix — shape: (N, D)  →  (200, 384)
    embedding_matrix: np.ndarray = raw_embeddings / norms
    print(f"[build_embedding_matrix] Normalised matrix shape: {embedding_matrix.shape}")

    # Sanity check: every row should now have ||row||₂ ≈ 1.0
    sample_norms = np.linalg.norm(embedding_matrix[:5], axis=1)
    print(f"[build_embedding_matrix] Sample row norms (first 5, expect ≈1): "
          f"{np.round(sample_norms, 6)}")

    return embedding_matrix


# ── 3. Semantic search via cosine similarity ───────────────────────────────────

def semantic_search(
    query_text: str,
    products_df: pd.DataFrame,
    embedding_matrix: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve the top-K most semantically similar products for a free-text query.

    Parameters
    ----------
    query_text       : natural-language search string
    products_df      : the full product DataFrame (N rows)
    embedding_matrix : pre-built, L2-normalised matrix, shape (N, D)
    model            : the SentenceTransformer used to build the matrix
    top_k            : number of results to return

    Returns
    -------
    pd.DataFrame with columns [product_id, title, category, price,
                                similarity_score], sorted descending.

    ── Linear-algebra mechanics ──────────────────────────────────────────────

    Cosine similarity between two vectors u, v:
        cos(u, v) = (u · v) / (||u||₂ · ||v||₂)

    Because BOTH the query vector AND every row of the embedding matrix are
    L2-normalised (||u||₂ = ||v||₂ = 1), the denominator collapses to 1:
        cos(u, v) = u · v                    (when ||u|| = ||v|| = 1)

    So we can compute ALL N similarity scores in one matrix–vector multiply:

        q  : shape (D,)    →  (384,)         query unit-vector
        E  : shape (N, D)  →  (200, 384)     product embedding matrix
        s  : shape (N,)    →  (200,)         cosine similarity scores

        s = E @ q
          = [E[0]·q, E[1]·q, …, E[N-1]·q]

    This is an O(N·D) operation — a single BLAS DGEMV call under the hood.
    At N=200, D=384 this takes < 1 ms; at N=10M it scales linearly and can
    be batched or approximated with ANN indices (e.g. FAISS IVF, HNSW).

    np.argpartition gives us the top-K indices in O(N) rather than O(N log N)
    (full sort), then we sort only those K elements.
    """
    # ── 3a. Encode & normalise the query ──────────────────────────────────────
    # Raw query embedding — shape: (D,)  →  (384,)
    # encode() returns (1, D) when given a single string; squeeze to (D,)
    raw_query: np.ndarray = model.encode(
        [query_text],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).squeeze()                                 # shape: (384,)

    # L2 normalise — scalar division, result shape: (384,)
    query_norm: float   = float(np.linalg.norm(raw_query))
    query_vec:  np.ndarray = raw_query / (query_norm if query_norm > 0 else 1.0)
    # shape: (D,)  →  (384,)

    # ── 3b. Matrix–vector cosine similarity ───────────────────────────────────
    # E  : (N, D)  →  (200, 384)
    # q  : (D,)    →  (384,)
    # s  = E @ q   →  (N,)  →  (200,)
    #
    # Each element s[i] = dot(E[i], q) = cos(E[i], q) (since both are unit vecs)
    similarity_scores: np.ndarray = embedding_matrix @ query_vec
    # shape: (N,)  →  (200,)  values ∈ [-1, 1]

    # ── 3c. Efficient top-K retrieval ─────────────────────────────────────────
    # np.argpartition: O(N) partial sort — guarantees the top_k largest values
    # are in the last top_k positions, but not yet sorted among themselves.
    # We then sort only those K indices: O(K log K) instead of O(N log N).
    k        = min(top_k, len(similarity_scores))
    part_idx = np.argpartition(similarity_scores, -k)[-k:]          # (K,) unordered
    top_idx  = part_idx[np.argsort(similarity_scores[part_idx])[::-1]]  # (K,) sorted ↓

    # ── 3d. Build result DataFrame ────────────────────────────────────────────
    result_df = products_df.iloc[top_idx][
        ["product_id", "title", "category", "price"]
    ].copy()
    result_df["similarity_score"] = np.round(similarity_scores[top_idx], 4)
    result_df = result_df.reset_index(drop=True)

    return result_df


# ── 4. Module initialisation helper ───────────────────────────────────────────

def init_search_engine(products_path: str = PRODUCTS_FILE):
    """
    Load data, initialise the model, and build the embedding matrix.

    Returns (products_df, embedding_matrix, model) so the caller can
    re-use them across multiple queries without redundant I/O or encoding.
    """
    products_df     = load_products(products_path)
    model           = SentenceTransformer(MODEL_NAME)
    texts           = products_df["text_feature"].tolist()
    embedding_matrix = build_embedding_matrix(texts, model)
    return products_df, embedding_matrix, model


# ── 5. Pretty-print helper ────────────────────────────────────────────────────

def _print_results(query: str, results: pd.DataFrame) -> None:
    width = 80
    print("\n" + "═" * width)
    print(f"  Query : \"{query}\"")
    print("═" * width)
    print(f"  {'#':<3}  {'Score':<8}  {'Category':<14}  {'Price':>7}  Title")
    print("─" * width)
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title_trunc = row["title"][:48] + "…" if len(row["title"]) > 48 else row["title"]
        print(f"  {rank:<3}  {row['similarity_score']:<8.4f}  "
              f"{row['category']:<14}  ₹{row['price']:>6.2f}  {title_trunc}")
    print("═" * width + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── One-time setup (amortised across all queries at serving time) ──────────
    products_df, embedding_matrix, model = init_search_engine()

    # Print final shape summary
    print(f"\n── Embedding matrix ready ──────────────────────────────────────")
    print(f"   Shape : {embedding_matrix.shape}   (products × embed_dim)")
    print(f"   dtype : {embedding_matrix.dtype}")
    print(f"   Size  : {embedding_matrix.nbytes / 1024:.1f} KB in memory")

    # ── Test queries ───────────────────────────────────────────────────────────
    test_queries = [
        ("something to read at night", 5),
        ("portable workstation for coding on the go", 5),
        ("keep my home clean without effort", 5),
        ("comfortable outfit for a morning jog", 5),
    ]

    for query_text, top_k in test_queries:
        results = semantic_search(
            query_text      = query_text,
            products_df     = products_df,
            embedding_matrix = embedding_matrix,
            model           = model,
            top_k           = top_k,
        )
        _print_results(query_text, results)
