"""
recommender.py
--------------
Personalised Recommendation Engine via Matrix Factorisation (SVD)
for the Amazon-Style Multi-Modal Search & Recommendation System.

Architecture Overview
---------------------
ratings (long format)
    │
    ▼
User-Item Matrix  R  :  shape (M, N)   M=users, N=products
    │
    ▼  Truncated SVD  (k latent factors)
    │
    ├── U   : (M, k)   — user latent factors
    ├── Σ   : (k,)     — singular values (importance of each factor)
    └── Vᵀ  : (k, N)   — item latent factors
    │
    ▼  Reconstruction
    R̂ = U · diag(Σ) · Vᵀ   shape (M, N)   — predicted ratings for every
                                              user × product pair
    │
    ▼  Recommendation
    For user u: mask already-rated items → argsort R̂[u] ↓ → top-K

Why SVD?
--------
SVD is the mathematical foundation of modern collaborative filtering.
It decomposes the sparse, noisy rating matrix into a low-rank approximation
that captures *latent factors* — abstract concepts like "tech-savvy user",
"home-improvement enthusiast", or "literary fiction lover" that explain
why certain users rate certain products highly.  Netflix's 2006 prize-
winning model was built on exactly this decomposition.

Dependencies
------------
    pip install numpy scipy pandas
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# ── Constants ──────────────────────────────────────────────────────────────────
RATINGS_FILE  = "ratings.csv"
PRODUCTS_FILE = "products.csv"

# k: number of latent factors to retain after truncation.
# Higher k → better reconstruction fidelity but more overfitting risk.
# k=10 is a sensible default for a 50-user × 200-product toy matrix;
# production systems (Netflix, Amazon) use k in the hundreds.
K_FACTORS = 10


# ── 1. Data Loading ────────────────────────────────────────────────────────────

def load_data(
    ratings_path: str = RATINGS_FILE,
    products_path: str = PRODUCTS_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ratings (long format) and product metadata.

    ratings_df columns : user_id (int), product_id (int), rating (int 1-5)
    products_df columns: product_id, title, description, category, price
    """
    ratings_df  = pd.read_csv(ratings_path)
    products_df = pd.read_csv(products_path)
    print(f"[load_data] ratings  : {ratings_df.shape}  "
          f"(unique users={ratings_df['user_id'].nunique()}, "
          f"unique products={ratings_df['product_id'].nunique()})")
    print(f"[load_data] products : {products_df.shape}")
    return ratings_df, products_df


# ── 2. Build the User-Item Matrix ──────────────────────────────────────────────

def build_user_item_matrix(
    ratings_df: pd.DataFrame,
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Pivot long-format ratings into a dense User × Item matrix.

    Parameters
    ----------
    ratings_df : DataFrame with columns [user_id, product_id, rating]

    Returns
    -------
    R          : np.ndarray, shape (M, N)
                 M = number of unique users
                 N = number of unique products in the catalogue (200)
                 R[i, j] = explicit rating of user i for product j,
                            0 if the user has not rated that product.
    user_ids   : list of M user IDs (row index mapping)
    product_ids: list of N product IDs (column index mapping)

    Matrix structure
    ----------------
    The pivot creates:

        product_id →  1     2     3   … 200
        user_id
        1          [  0     4     0   …   0  ]
        2          [  3     0     5   …   0  ]
        …
        50         [  0     0     2   …   4  ]

    NaN (unobserved) → 0  (implicit "no interaction").
    This is the standard "zero-imputation" strategy for explicit-feedback CF;
    it treats unknowns as neutral rather than negative.

    Sparsity note
    -------------
    With ~854 ratings across 50×200 = 10,000 cells the matrix is ~91.5%
    zeros — exactly the regime where low-rank approximation shines.
    """
    pivot = ratings_df.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean",         # handles duplicate (user, product) pairs cleanly
    ).fillna(0)

    R           = pivot.values.astype(np.float64)   # shape: (M, N)
    user_ids    = list(pivot.index)                  # length M
    product_ids = list(pivot.columns)                # length N

    M, N = R.shape
    sparsity = 1.0 - np.count_nonzero(R) / R.size
    print(f"\n[build_user_item_matrix] Shape : {R.shape}  (M={M} users, N={N} products)")
    print(f"[build_user_item_matrix] Sparsity : {sparsity*100:.1f}%")
    print(f"[build_user_item_matrix] dtype    : {R.dtype}")

    return R, user_ids, product_ids


# ── 3. Mean-centre the matrix (de-biasing) ────────────────────────────────────

def mean_centre(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract each user's mean rating (computed over rated items only).

    Why centre?
    -----------
    Some users rate everything 5★; others are harsh and average 2★.
    Without centring, SVD would waste latent factors modelling this
    per-user *bias* rather than genuine *preferences*.

    Mean is computed only over non-zero entries so that unobserved zeros
    don't drag the mean down.

    Parameters
    ----------
    R : (M, N) raw rating matrix

    Returns
    -------
    R_centred  : (M, N) — each row shifted by its observed mean
    user_means : (M,)   — the per-user mean subtracted (needed to
                          reverse the shift at prediction time)
    """
    # For each row, average only the non-zero (rated) entries.
    # np.true_divide with where= avoids division-by-zero for users with 0 ratings.
    rated_mask = R != 0                                    # (M, N) boolean
    row_sums   = np.sum(R * rated_mask, axis=1)            # (M,)
    row_counts = np.sum(rated_mask, axis=1).clip(min=1)    # (M,)  clip→avoid /0
    user_means = row_sums / row_counts                     # (M,)

    # Broadcast subtraction: (M, N) - (M, 1)
    R_centred = np.where(rated_mask, R - user_means[:, np.newaxis], 0.0)
    # shape: (M, N) — zeros remain zero; rated entries are now mean-centred

    print(f"\n[mean_centre] user_means range : "
          f"[{user_means.min():.2f}, {user_means.max():.2f}]")
    print(f"[mean_centre] R_centred shape  : {R_centred.shape}")
    return R_centred, user_means


# ── 4. Truncated SVD ──────────────────────────────────────────────────────────

def run_svd(
    R_centred: np.ndarray,
    k: int = K_FACTORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose the user-item matrix into k latent factors via truncated SVD.

    Full SVD (conceptually)
    -----------------------
    Any real matrix R of shape (M, N) can be written as:

        R  =  U · diag(Σ) · Vᵀ

    where:
        U   : (M, M)   — left singular vectors   (user latent space)
        Σ   : (M,)     — singular values, sorted ↓  (factor importance)
        Vᵀ  : (M, N)   — right singular vectors  (item latent space)

    Truncated SVD (rank-k approximation)
    -------------------------------------
    We keep only the k *largest* singular values, giving us the best
    rank-k approximation of R in the Frobenius-norm sense (Eckart-Young
    theorem).  With k << min(M, N) this:
        • Removes noise (small singular values ≈ noise)
        • Compresses memory from O(MN) to O(k(M+N))
        • Forces generalisation — the model can't memorise individual ratings

    After truncation the shapes become:
        U   : (M, k)   — each user is a point in k-dimensional latent space
        Σ   : (k,)     — k singular values (energy retained per factor)
        Vᵀ  : (k, N)   — each product is a point in k-dimensional latent space

    With M=50, N=200, k=10:
        U   : (50,  10)
        Σ   : (10,)
        Vᵀ  : (10, 200)

    scipy.sparse.linalg.svds
    ------------------------
    Uses ARPACK (an implicitly restarted Lanczos method) to compute only the
    top-k singular triplets — O(k · M · N) vs O(min(M,N)³) for full SVD.
    Note: svds returns singular values in *ascending* order; we flip to
    descending for convention.

    Parameters
    ----------
    R_centred : (M, N) mean-centred rating matrix
    k         : number of latent factors to retain

    Returns
    -------
    U   : (M, k)
    S   : (k,)      singular values, descending
    Vt  : (k, N)
    """
    M, N = R_centred.shape
    # k must be < min(M, N) for svds
    k_actual = min(k, min(M, N) - 1)
    if k_actual != k:
        print(f"[run_svd] Clipping k from {k} → {k_actual} (constraint: k < min(M,N))")

    print(f"\n[run_svd] Running truncated SVD  (k={k_actual})  on shape {R_centred.shape}")

    # scipy.sparse.linalg.svds expects a sparse or dense matrix;
    # use csr_matrix for efficiency on sparse data.
    R_sparse = csr_matrix(R_centred)                    # sparse (M, N)

    # svds returns (U, S, Vt) sorted in ASCENDING order of singular values
    U_asc, S_asc, Vt_asc = svds(R_sparse, k=k_actual)  # ascending order

    # Flip to DESCENDING order (largest factor first — standard convention)
    idx = np.argsort(S_asc)[::-1]                       # (k,) index permutation
    U   = U_asc[:, idx]                                 # (M, k)
    S   = S_asc[idx]                                    # (k,)
    Vt  = Vt_asc[idx, :]                                # (k, N)

    # ── Shape report ──────────────────────────────────────────────────────────
    print(f"[run_svd] U   shape : {U.shape}   (users  × latent factors)")
    print(f"[run_svd] Σ   shape : {S.shape}   (singular values, ↓ sorted)")
    print(f"[run_svd] Vᵀ  shape : {Vt.shape}  (latent factors × products)")

    # ── Explained variance ────────────────────────────────────────────────────
    # The squared singular values sum to the total Frobenius norm² of R.
    # The fraction captured by the top-k factors tells us how much
    # "signal" we retained vs discarded as noise.
    total_variance   = np.sum(R_centred ** 2)           # Frobenius norm²
    captured_var     = np.sum(S ** 2)
    explained_pct    = 100.0 * captured_var / (total_variance + 1e-9)
    print(f"[run_svd] Singular values          : {np.round(S, 3)}")
    print(f"[run_svd] Variance explained by k={k_actual}: {explained_pct:.1f}%")

    return U, S, Vt


# ── 5. Reconstruct the predicted rating matrix ─────────────────────────────────

def reconstruct_matrix(
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
    user_means: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the full predicted rating matrix R̂ from SVD factors.

    Reconstruction formula
    ----------------------

        R̂  =  U · diag(Σ) · Vᵀ  +  user_bias

    Step 1 — diag(Σ) · Vᵀ
        Scale each row of Vᵀ by the corresponding singular value.
        Equivalent to weighting each latent dimension by its importance.

        diag(Σ) : (k, k)    [implicit; we use np.diag()]
        Σ · Vᵀ  : (k, N)    or equivalently S[:, None] * Vt

    Step 2 — U · (Σ · Vᵀ)
        Project each user's latent vector through the weighted item space.

        U   : (M, k)
        ΣVᵀ : (k, N)
        R̂_c : (M, N)    ← centred predictions

    Step 3 — Add back user means (reverse the centring from step 3)
        user_means[:, np.newaxis] broadcasts (M,) → (M, 1)
        R̂ = R̂_c + user_means[:, np.newaxis]       shape (M, N)

    Interpretation
    --------------
    R̂[i, j] is the model's best estimate of how user i would rate
    product j, even for products user i has never seen.
    This is the core of collaborative filtering: it generalises from
    *observed* ratings to *unobserved* ones via the shared latent space.

    Parameters
    ----------
    U          : (M, k)
    S          : (k,)
    Vt         : (k, N)
    user_means : (M,)

    Returns
    -------
    R_hat : (M, N) predicted rating matrix, values approximately in [1, 5]
    """
    # Step 1+2: R̂_centred = U · diag(Σ) · Vᵀ
    # We avoid materialising the full (k, k) diagonal matrix:
    #   U · diag(Σ) = U * S[np.newaxis, :]    → (M, k)  (row-wise scale)
    #   then  @ Vt                             → (M, N)
    R_hat_centred = (U * S[np.newaxis, :]) @ Vt   # (M, k) @ (k, N) → (M, N)

    # Step 3: add per-user mean back
    # user_means[:, np.newaxis] : (M, 1)  broadcasts across N columns
    R_hat = R_hat_centred + user_means[:, np.newaxis]   # (M, N)

    print(f"\n[reconstruct_matrix] R̂ shape  : {R_hat.shape}")
    print(f"[reconstruct_matrix] R̂ range  : [{R_hat.min():.3f}, {R_hat.max():.3f}]")

    return R_hat


# ── 6. Recommendation function ────────────────────────────────────────────────

def get_collaborative_recommendations(
    user_id: int,
    R_hat: np.ndarray,
    R_observed: np.ndarray,
    user_ids: list[int],
    product_ids: list[int],
    products_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Return the top-K unrated products with the highest predicted ratings
    for a given user.

    Parameters
    ----------
    user_id      : the target user (must exist in user_ids)
    R_hat        : (M, N) reconstructed predicted rating matrix
    R_observed   : (M, N) original rating matrix (zeros = unrated)
    user_ids     : list mapping row index → user_id
    product_ids  : list mapping column index → product_id
    products_df  : product metadata DataFrame
    top_k        : number of recommendations to return

    Returns
    -------
    pd.DataFrame with columns:
        [product_id, title, category, price, predicted_rating]

    Linear-algebra walkthrough for user u
    --------------------------------------
    1.  user_row = R̂[u, :]           shape (N,)
        — the full vector of predicted ratings for this user across all N items.

    2.  already_rated_mask = R_obs[u, :] != 0   shape (N,) boolean
        — True where user has already given an explicit rating.

    3.  user_row[already_rated_mask] = -∞
        — Mask out rated items so they can't appear in recommendations.
          We don't want to "recommend" something the user already knows.

    4.  top_k_idx = argsort(user_row)[-top_k:][::-1]
        — Indices of the K products with the highest predicted score.

    5.  Map indices → product_ids → metadata join.
    """
    if user_id not in user_ids:
        raise ValueError(
            f"user_id={user_id} not found. "
            f"Valid range: {min(user_ids)}–{max(user_ids)}"
        )

    u_idx = user_ids.index(user_id)     # row index of this user in R̂

    # Step 1 — Extract this user's predicted rating vector
    user_row = R_hat[u_idx, :].copy()   # shape (N,)  e.g. (200,)

    # Step 2 — Build the already-rated mask
    already_rated = R_observed[u_idx, :] != 0   # shape (N,) boolean
    n_rated = int(already_rated.sum())

    # Step 3 — Suppress rated items
    user_row[already_rated] = -np.inf   # push them to the bottom of the ranking

    # Step 4 — Efficient top-K retrieval (O(N) argpartition + O(k log k) sort)
    k          = min(top_k, int((~already_rated).sum()))
    part_idx   = np.argpartition(user_row, -k)[-k:]              # (k,) unordered
    top_k_idx  = part_idx[np.argsort(user_row[part_idx])[::-1]]  # (k,) sorted ↓

    # Step 5 — Map column indices → product_ids → metadata
    rec_product_ids     = [product_ids[i] for i in top_k_idx]
    rec_predicted_scores = user_row[top_k_idx]

    rec_df = pd.DataFrame({
        "product_id":       rec_product_ids,
        "predicted_rating": np.round(rec_predicted_scores, 4),
    }).merge(
        products_df[["product_id", "title", "category", "price"]],
        on="product_id",
        how="left",
    )[["product_id", "title", "category", "price", "predicted_rating"]]

    # Attach rated-item summary for context
    rec_df.attrs["user_id"]  = user_id
    rec_df.attrs["n_rated"]  = n_rated
    return rec_df


# ── 7. Build the full pipeline (one-time setup) ────────────────────────────────

def build_recommender(
    ratings_path:  str = RATINGS_FILE,
    products_path: str = PRODUCTS_FILE,
    k:             int = K_FACTORS,
):
    """
    End-to-end pipeline: load → pivot → centre → SVD → reconstruct.

    Returns everything needed to serve recommendations without repeating
    the expensive SVD step.

    Returns
    -------
    R_hat        : (M, N) predicted rating matrix
    R_observed   : (M, N) original observed matrix (for masking)
    user_ids     : list of M user IDs
    product_ids  : list of N product IDs
    products_df  : product metadata
    """
    ratings_df, products_df = load_data(ratings_path, products_path)
    R, user_ids, product_ids = build_user_item_matrix(ratings_df)
    R_centred, user_means   = mean_centre(R)
    U, S, Vt               = run_svd(R_centred, k=k)
    R_hat                   = reconstruct_matrix(U, S, Vt, user_means)
    return R_hat, R, user_ids, product_ids, products_df


# ── 8. Pretty-print helper ────────────────────────────────────────────────────

def _print_recommendations(recs: pd.DataFrame) -> None:
    user_id = recs.attrs.get("user_id", "?")
    n_rated = recs.attrs.get("n_rated", "?")
    width   = 84
    print("\n" + "═" * width)
    print(f"  Collaborative Filter Recommendations  —  User {user_id}  "
          f"(already rated: {n_rated} products)")
    print("═" * width)
    print(f"  {'#':<3}  {'Pred ★':<8}  {'Category':<14}  {'Price':>7}  Title")
    print("─" * width)
    for rank, (_, row) in enumerate(recs.iterrows(), start=1):
        title_t = (row["title"][:50] + "…"
                   if len(row["title"]) > 50 else row["title"])
        print(f"  {rank:<3}  {row['predicted_rating']:<8.4f}  "
              f"{row['category']:<14}  ₹{row['price']:>6.2f}  {title_t}")
    print("═" * width + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  Collaborative Filtering Recommender — SVD Matrix Factorisation")
    print("=" * 68)

    # ── One-time setup ─────────────────────────────────────────────────────────
    R_hat, R_observed, user_ids, product_ids, products_df = build_recommender(
        k=K_FACTORS
    )

    # ── Matrix summary ─────────────────────────────────────────────────────────
    print(f"\n── Final matrices ──────────────────────────────────────────────")
    print(f"   R_observed : {R_observed.shape}   (original sparse matrix)")
    print(f"   R̂          : {R_hat.shape}   (dense predicted ratings)")
    print(f"   K factors  : {K_FACTORS}")

    # ── Test with several user IDs from different taste clusters ───────────────
    # Cluster map (from generate_data.py):
    #   Electronics → users 1-11  |  Apparel → 12-22
    #   Home        → 23-36       |  Books   → 37-50
    test_users = [
        3,   # Electronics cluster
        18,  # Apparel cluster
        30,  # Home cluster
        45,  # Books cluster
        5,   # Cross-category omnivore user
    ]

    for uid in test_users:
        recs = get_collaborative_recommendations(
            user_id     = uid,
            R_hat       = R_hat,
            R_observed  = R_observed,
            user_ids    = user_ids,
            product_ids = product_ids,
            products_df = products_df,
            top_k       = 5,
        )
        _print_recommendations(recs)
