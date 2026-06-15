"""
generate_data.py
----------------
Synthetic dataset generator for an Amazon-Style Multi-Modal Search &
Recommendation Engine.

Outputs
-------
products.csv  – 200 products with realistic titles, descriptions, categories, prices
ratings.csv   – Sparse explicit-feedback ratings (user_id, product_id, rating)

Reproducibility
---------------
All randomness is controlled via a single RANDOM_SEED constant.
"""

import random
import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_PRODUCTS   = 200
NUM_USERS      = 50
RATINGS_FILE   = "ratings.csv"
PRODUCTS_FILE  = "products.csv"

CATEGORIES = ["Electronics", "Apparel", "Home", "Books"]

# Fraction of the full user×product matrix that will have a rating (sparsity)
BASE_SPARSITY  = 0.06   # ~6 % overall density

# ── Product template data ──────────────────────────────────────────────────────

ELECTRONICS = {
    "templates": [
        ("{brand} {line} {form} – {feat1} & {feat2}",
         "{brand} {line} {form} delivers {feat1} and {feat2}. "
         "Engineered for power users, it features a {spec1}, a {spec2}, "
         "and {spec3}. {tagline}"),
    ],
    "brands":  ["Sony", "Samsung", "Anker", "Logitech", "Bose", "Apple",
                 "LG", "Razer", "JBL", "Belkin", "SteelSeries", "Jabra"],
    "lines":   ["ProMax", "UltraView", "NovaSeries", "ZenCore", "HyperEdge",
                "PulseX", "ClearCast", "VisionPro", "SwiftLink", "AuraSync"],
    "forms":   ["Wireless Earbuds", "Noise-Cancelling Headphones",
                "Mechanical Keyboard", "Gaming Mouse", "4K Webcam",
                "Portable Charger", "Smart Speaker", "USB-C Hub",
                "Curved Monitor", "Ergonomic Trackpad", "Wi-Fi 6 Router",
                "Action Camera", "LED Desk Lamp", "Smart Plug Strip"],
    "feat1":   ["Active Noise Cancellation", "Adaptive EQ", "Fast Charging",
                "Bluetooth 5.3", "RGB Backlighting", "AI-Enhanced Mic",
                "Ultra-Low Latency", "Hi-Res Audio", "Multi-Device Pairing"],
    "feat2":   ["30-Hour Battery Life", "IPX5 Water Resistance",
                "USB-C PD 100 W", "2 ms Response Time", "8K DPI Sensor",
                "360° Spatial Audio", "Matter Smart Home Support",
                "Thunderbolt 4 Pass-Through", "Auto Game Mode"],
    "spec1":   ["6-core DSP chip", "1.5\" bio-cellulose driver",
                "tactile optical switch", "PixArt 3395 sensor",
                "Sony STARVIS 2 sensor", "graphene-coated membrane"],
    "spec2":   ["USB-C + 3.5 mm dual output", "dedicated bass radiator",
                "per-key actuation logging", "tilt-adjustable scroll wheel",
                "f/1.8 wide aperture lens", "planar magnetic transducer"],
    "spec3":   ["a companion EQ app for iOS & Android",
                "swappable hot-swap switch sockets",
                "ambient sound pass-through mode",
                "real-time noise suppression powered by AI",
                "hardware-level encryption for secure calls"],
    "taglines": ["Perfect for remote workers, commuters, and audiophiles alike.",
                 "Dominate every session with tournament-grade precision.",
                 "Your creative workflow, supercharged.",
                 "Where premium sound meets everyday convenience.",
                 "Engineered to keep up with your ambitions."],
    "price_range": (19.99, 499.99),
}

APPAREL = {
    "templates": [
        ("{brand} {line} {form} – {feat1} & {feat2}",
         "{brand} {line} {form} combines {feat1} with {feat2}. "
         "Crafted from {spec1} and finished with {spec2}, "
         "this {form.lower()} is designed for {spec3}. {tagline}"),
    ],
    "brands":  ["Nike", "Adidas", "Levi's", "Uniqlo", "Patagonia", "The North Face",
                 "Columbia", "Under Armour", "Champion", "Carhartt", "Everlane"],
    "lines":   ["AeroFit", "UrbanEdge", "TrailBlaze", "EcoWeave", "FlexCore",
                "CloudStretch", "AlpineShield", "PureCotton", "DriLayer"],
    "forms":   ["Running Jacket", "Merino Wool Crew-Neck Sweater",
                "5-Pocket Slim-Fit Jeans", "Compression Training Tights",
                "Waterproof Shell Parka", "Organic Cotton T-Shirt",
                "Insulated Puffer Vest", "Lightweight Cargo Shorts",
                "Thermal Base-Layer Set", "Performance Polo Shirt",
                "Fleece Quarter-Zip Pullover"],
    "feat1":   ["moisture-wicking technology", "four-way stretch fabric",
                "wind-resistant outer shell", "UPF 50+ sun protection",
                "recycled material construction", "seamless flatlock stitching"],
    "feat2":   ["reflective safety detailing", "hidden zip pockets",
                "adjustable hood and cuffs", "anti-odour silver-ion treatment",
                "articulated knee panelling", "breathable mesh lining"],
    "spec1":   ["88% recycled polyester / 12% elastane",
                "100% GOTS-certified organic cotton",
                "Merino wool sourced from ZQ-certified farms",
                "DWR-treated ripstop nylon",
                "bio-based nylon derived from castor oil"],
    "spec2":   ["flatlock seams for chafe-free comfort",
                "YKK® AquaGuard zippers", "bonded hem tape",
                "enzyme-washed finish", "laser-cut ventilation panels"],
    "spec3":   ["trail runners and urban explorers alike",
                "all-day wear from office to outdoor adventure",
                "high-intensity training in all conditions",
                "effortless layering in transitional weather",
                "weekend trips and everyday casual wear"],
    "taglines": ["Move freely. Look sharp. Feel great.",
                 "Built for the planet, built for performance.",
                 "Comfort that keeps pace with your lifestyle.",
                 "Where function meets timeless style.",
                 "Gear up for whatever comes next."],
    "price_range": (14.99, 249.99),
}

HOME = {
    "templates": [
        ("{brand} {line} {form} – {feat1} & {feat2}",
         "The {brand} {line} {form} brings {feat1} and {feat2} to your home. "
         "Constructed with {spec1} and equipped with {spec2}, "
         "it offers {spec3}. {tagline}"),
    ],
    "brands":  ["Instant Pot", "Cuisinart", "iRobot", "Dyson", "Philips",
                 "KitchenAid", "Breville", "Nespresso", "Shark", "OXO", "Vitamix"],
    "lines":   ["SmartHome", "ProChef", "PureClean", "ZenSpace", "EcoLiving",
                "ClearAir", "IntelliCook", "TranquilSleep", "BrightKitchen"],
    "forms":   ["Air Purifier", "Robot Vacuum & Mop Combo", "Stand Mixer",
                "Programmable Coffee Maker", "Sous Vide Precision Cooker",
                "Cordless Stick Vacuum", "Smart Air Fryer", "Weighted Blanket",
                "Bamboo Cutting Board Set", "Cast-Iron Dutch Oven",
                "Countertop Bread Maker", "Water Filter Pitcher"],
    "feat1":   ["HEPA H13 filtration", "LiDAR-guided navigation",
                "10-speed precision motor", "30-bar pressure extraction",
                "±0.1°C temperature accuracy", "75 dB ultra-quiet operation",
                "self-emptying dustbin", "7-in-1 cooking modes"],
    "feat2":   ["real-time air-quality display", "room-mapping with no-go zones",
                "tilt-head attachment hub", "built-in milk frother",
                "app-based scheduling", "voice-assistant integration",
                "dishwasher-safe components", "child-safety auto shut-off"],
    "spec1":   ["aircraft-grade anodised aluminium",
                "food-grade stainless steel 18/10",
                "BPA-free Tritan co-polyester",
                "enamelled cast iron",
                "sustainably sourced moso bamboo"],
    "spec2":   ["a 3-layer activated carbon + HEPA filter stack",
                "dual spinning side-brushes and V-shaped main brush",
                "a 5.5 L stainless steel bowl",
                "a 19-bar Italian pump and pre-infusion system",
                "a 1200 W immersion circulator",
                "a 0.9 L removable water reservoir"],
    "spec3":   ["silent yet powerful performance in any room",
                "effortless floor-to-ceiling cleaning on a single charge",
                "professional baking and cooking results at home",
                "barista-quality espresso at the touch of a button",
                "restaurant-grade sous vide cooking for home chefs",
                "all-day comfort with temperature-regulating fill"],
    "taglines": ["Transform your home into a smart sanctuary.",
                 "Spend less time cleaning, more time living.",
                 "Precision craftsmanship for everyday cooking.",
                 "Elevate your morning ritual, every single day.",
                 "Designed to simplify. Built to last."],
    "price_range": (12.99, 399.99),
}

BOOKS = {
    "templates": [
        ("{title_full}",
         "In {title_full}, {author} explores {feat1} and {feat2}. "
         "{spec1}. {spec2}. "
         "Spanning {spec3}, this {form} has been described as {tagline}"),
    ],
    "brands":  ["", ""],  # unused but keeps uniform structure
    "lines":   ["", ""],
    "forms":   ["gripping debut novel", "authoritative non-fiction account",
                "bestselling self-help guide", "definitive technical reference",
                "celebrated essay collection", "award-winning memoir",
                "illustrated science compendium", "practical business handbook"],
    # For books, 'brand' slot will hold author, 'line' slot holds subtitle cue
    "authors": ["Dr. Priya Nair", "James Calloway", "Elena Vasquez",
                "Dr. Marcus Webb", "Aisha Okonkwo", "Thomas Lin",
                "Dr. Sarah Brennan", "Ravi Patel", "Claire Fontaine",
                "David Osei", "Maya Krishnamurthy", "Prof. Neil Hartley"],
    "title_stubs": [
        "The Architecture of Tomorrow",
        "Deep Roots: A Memoir of Soil and Silicon",
        "Thinking in Systems",
        "The Last Algorithm",
        "Untangling Complexity",
        "The Quiet Revolution",
        "Code & Culture",
        "Rethinking Intelligence",
        "The Design of Everyday Decisions",
        "When Data Speaks",
        "The Human Edge",
        "Patterns of Resilience",
        "Zero to One Hundred",
        "The Empathy Advantage",
        "Signals & Noise",
    ],
    "feat1":   ["the intersection of technology and human psychology",
                "the hidden forces shaping modern economics",
                "the neuroscience of creativity and focus",
                "the history of artificial intelligence from its roots",
                "the global supply chain and its environmental cost",
                "the science of habit formation and behaviour change",
                "the philosophy of ethical leadership in the digital age"],
    "feat2":   ["how individuals can thrive amid rapid change",
                "the overlooked voices that shaped a scientific revolution",
                "strategies for building resilient organisations",
                "the personal cost of relentless optimisation culture",
                "why diversity drives measurable innovation outcomes",
                "the surprising psychology of risk and reward"],
    "spec1":   ["Drawing on decades of field research and exclusive interviews",
                "Blending rigorous data analysis with compelling narrative",
                "Weaving together case studies from five continents",
                "Grounded in peer-reviewed cognitive science",
                "Combining vivid personal storytelling with hard empirical evidence"],
    "spec2":   ["the author presents a fresh framework for navigating uncertainty",
                "readers gain actionable tools for immediate real-world impact",
                "it challenges assumptions held by experts and laypeople alike",
                "the book reframes conventional wisdom with striking clarity",
                "it provides a roadmap that is both inspiring and deeply practical"],
    "spec3":   ["350 pages of meticulously researched content",
                "400+ pages enriched with charts, interviews, and case studies",
                "280 pages of concise, high-impact insights",
                "500 pages that read like a page-turning narrative",
                "320 pages that balance depth with accessibility"],
    "taglines": ["'a landmark work for our times' — The Economist.",
                 "'essential reading for anyone curious about the future' — MIT Tech Review.",
                 "'transformative, urgent, and beautifully written' — The Guardian.",
                 "'the definitive guide to navigating complexity' — Harvard Business Review.",
                 "'a must-read that will change how you see the world' — Nature."],
    "price_range": (7.99, 49.99),
}

CATEGORY_DATA = {
    "Electronics": ELECTRONICS,
    "Apparel":     APPAREL,
    "Home":        HOME,
    "Books":       BOOKS,
}


# ── Helper functions ───────────────────────────────────────────────────────────

def _pick(lst: list):
    """Randomly pick one element from a list."""
    return random.choice(lst)


def _generate_product_row(product_id: int, category: str) -> dict:
    """Generate a single product row for the given category."""
    data = CATEGORY_DATA[category]
    pr   = data["price_range"]

    if category == "Books":
        author     = _pick(data["authors"])
        title_stub = _pick(data["title_stubs"])
        title_full = f"{title_stub} by {author}"
        form       = _pick(data["forms"])
        feat1      = _pick(data["feat1"])
        feat2      = _pick(data["feat2"])
        spec1      = _pick(data["spec1"])
        spec2      = _pick(data["spec2"])
        spec3      = _pick(data["spec3"])
        tagline    = _pick(data["taglines"])

        title = title_full
        desc  = (
            f"In {title_full}, {author} explores {feat1} and {feat2}. "
            f"{spec1}. {spec2}. "
            f"Spanning {spec3}, this {form} has been described as {tagline}"
        )
    else:
        brand   = _pick(data["brands"])
        line    = _pick(data["lines"])
        form    = _pick(data["forms"])
        feat1   = _pick(data["feat1"])
        feat2   = _pick(data["feat2"])
        spec1   = _pick(data["spec1"])
        spec2   = _pick(data["spec2"])
        spec3   = _pick(data["spec3"])
        tagline = _pick(data["taglines"])

        title = f"{brand} {line} {form} – {feat1} & {feat2}"
        desc  = (
            f"The {brand} {line} {form} delivers {feat1} and {feat2}. "
            f"Engineered for discerning customers, it features {spec1}, "
            f"{spec2}, and {spec3}. {tagline}"
        )

    price = round(random.uniform(*pr), 2)

    return {
        "product_id":  product_id,
        "title":       title,
        "description": desc,
        "category":    category,
        "price":       price,
    }


def generate_products(n: int = NUM_PRODUCTS) -> pd.DataFrame:
    """
    Generate `n` product rows.

    Category distribution is approximately uniform but randomised so the
    exact counts vary slightly around n / num_categories.
    """
    # Assign categories: ensure at least n//len(CATEGORIES) per category,
    # then fill remaining slots randomly.
    per_cat  = n // len(CATEGORIES)
    cats     = CATEGORIES * per_cat
    leftover = n - len(cats)
    cats    += random.choices(CATEGORIES, k=leftover)
    random.shuffle(cats)

    rows = [_generate_product_row(pid + 1, cat)
            for pid, cat in enumerate(cats)]
    df = pd.DataFrame(rows)
    return df


# ── Rating generation ──────────────────────────────────────────────────────────

# User clusters: each cluster is a group of users that share a preferred
# category.  Users in the same cluster are more likely to rate products in
# that category highly, creating the "overlapping tastes" effect.
USER_CLUSTERS = {
    "Electronics": list(range(1,  12)),   # users 1-11
    "Apparel":     list(range(12, 23)),   # users 12-22
    "Home":        list(range(23, 37)),   # users 23-36
    "Books":       list(range(37, 51)),   # users 37-50
}
# A few "omnivore" users overlap clusters to add cross-category noise
CROSS_USERS = [5, 15, 28, 42]   # these users rate across all categories


def _rating_for_user_product(user_id: int, category: str) -> int:
    """
    Return a biased rating (1-5).
    Users in the matching cluster tend to rate higher (4-5).
    Other users rate more uniformly (1-5) with a slight positive skew.
    """
    user_cluster = next(
        (cat for cat, uids in USER_CLUSTERS.items() if user_id in uids), None
    )
    if user_cluster == category or user_id in CROSS_USERS:
        # Preferred category → higher ratings
        return int(np.clip(np.random.normal(loc=4.2, scale=0.7), 1, 5).round())
    else:
        # Non-preferred category → broader spread
        return int(np.clip(np.random.normal(loc=3.0, scale=1.2), 1, 5).round())


def generate_ratings(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a sparse ratings matrix.

    Strategy
    --------
    1. For each user, sample a subset of products proportional to
       BASE_SPARSITY (≈6 % of 200 = ~12 products per user).
    2. Users in a cluster are biased toward products in their preferred
       category (3× higher sampling probability).
    3. CROSS_USERS sample uniformly across all categories.
    4. Ratings are drawn from category-biased distributions (see above).
    """
    records = []

    cat_product_ids: dict[str, list[int]] = {
        cat: products_df.loc[products_df["category"] == cat, "product_id"].tolist()
        for cat in CATEGORIES
    }
    all_product_ids = products_df["product_id"].tolist()

    base_n = max(1, int(NUM_PRODUCTS * BASE_SPARSITY))   # ~12

    for user_id in range(1, NUM_USERS + 1):
        user_cluster = next(
            (cat for cat, uids in USER_CLUSTERS.items() if user_id in uids), None
        )

        if user_id in CROSS_USERS:
            # Omnivore: sample uniformly, but rate a few more products
            n_to_rate = base_n + random.randint(5, 15)
            sampled_ids = random.sample(all_product_ids,
                                        min(n_to_rate, NUM_PRODUCTS))
        else:
            # Build a weighted pool: 3× weight on preferred-category products
            weighted_pool = []
            for cat, pids in cat_product_ids.items():
                weight = 3 if cat == user_cluster else 1
                weighted_pool.extend(pids * weight)

            n_to_rate = base_n + random.randint(0, 10)
            # sample without replacement from the weighted pool
            # (deduplicate with seen set)
            seen: set[int] = set()
            sampled_ids = []
            random.shuffle(weighted_pool)
            for pid in weighted_pool:
                if pid not in seen:
                    seen.add(pid)
                    sampled_ids.append(pid)
                if len(sampled_ids) >= n_to_rate:
                    break

        for pid in sampled_ids:
            cat = products_df.loc[
                products_df["product_id"] == pid, "category"
            ].values[0]
            rating = _rating_for_user_product(user_id, cat)
            records.append({
                "user_id":    user_id,
                "product_id": pid,
                "rating":     rating,
            })

    df = pd.DataFrame(records).drop_duplicates(subset=["user_id", "product_id"])
    df = df.sort_values(["user_id", "product_id"]).reset_index(drop=True)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Synthetic Dataset Generator — Multi-Modal Rec Engine")
    print(f"  Random seed: {RANDOM_SEED}")
    print("=" * 60)

    # ── Products ──────────────────────────────────────────────────────────────
    print("\n[1/2] Generating products …")
    products_df = generate_products(NUM_PRODUCTS)
    products_df.to_csv(PRODUCTS_FILE, index=False)

    print(f"      Saved {len(products_df)} rows → '{PRODUCTS_FILE}'")
    print("      Category distribution:")
    for cat, cnt in products_df["category"].value_counts().items():
        print(f"        {cat:<14} {cnt:>3} products")
    print(f"      Price range: ₹{products_df['price'].min():.2f} – "
          f"₹{products_df['price'].max():.2f}")

    # ── Ratings ───────────────────────────────────────────────────────────────
    print("\n[2/2] Generating ratings …")
    ratings_df = generate_ratings(products_df)
    ratings_df.to_csv(RATINGS_FILE, index=False)

    total_possible = NUM_USERS * NUM_PRODUCTS
    density = len(ratings_df) / total_possible * 100

    print(f"      Saved {len(ratings_df)} rows → '{RATINGS_FILE}'")
    print(f"      Matrix density : {density:.2f}%  "
          f"({len(ratings_df)} / {total_possible} possible interactions)")
    print(f"      Users          : {ratings_df['user_id'].nunique()}")
    print(f"      Rated products : {ratings_df['product_id'].nunique()} "
          f"/ {NUM_PRODUCTS}")
    print(f"      Rating dist    :")
    for r, cnt in sorted(ratings_df["rating"].value_counts().items()):
        bar = "█" * (cnt // 5)
        print(f"        ★{r}  {cnt:>4}  {bar}")

    print("\n✓ Done. Files are ready for the next pipeline stage.\n")


if __name__ == "__main__":
    main()
