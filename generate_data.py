"""
generate_data_v2.py
-------------------
UPGRADED Dataset Generator for Amazon-Style ML System

Improvements over v1:
  • 500 products (was 200)
  • 100 users (was 50)
  • 8 categories (was 4)
  • 3x larger word pools
  • More realistic descriptions with varied sentence structures
  • Better product diversity
  • Explicit user personas
  • Better taste cluster separation
"""

import random
import numpy as np
import pandas as pd
from itertools import product as iterproduct

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_PRODUCTS   = 500
NUM_USERS      = 100
RATINGS_FILE   = "ratings_v2.csv"
PRODUCTS_FILE  = "products_v2.csv"
BASE_SPARSITY  = 0.04   # ~4% density (sparse, realistic)

CATEGORIES = ["Electronics", "Apparel", "Home", "Books", "Sports", "Beauty", "Toys", "Food"]

# ── Expanded Product Templates ─────────────────────────────────────────────────

ELECTRONICS = {
    "brands": [
        "Sony", "Samsung", "Anker", "Logitech", "Bose", "Apple", "LG", "Razer",
        "JBL", "Belkin", "SteelSeries", "Jabra", "Corsair", "ASUS", "Lenovo",
        "Dell", "HP", "Nvidia", "Intel", "Qualcomm", "Google", "Amazon", "Meta"
    ],
    "lines": [
        "ProMax", "UltraView", "NovaSeries", "ZenCore", "HyperEdge", "PulseX",
        "ClearCast", "VisionPro", "SwiftLink", "AuraSync", "Nexus", "Titan",
        "Phoenix", "Quantum", "Stellar", "Velocity", "Infinity", "Precision",
        "Harmony", "Zenith", "Apex", "Elite", "Pro", "Ultra"
    ],
    "forms": [
        "Wireless Earbuds", "Noise-Cancelling Headphones", "Mechanical Keyboard",
        "Gaming Mouse", "4K Webcam", "Portable Charger", "Smart Speaker",
        "USB-C Hub", "Curved Monitor", "Ergonomic Trackpad", "Wi-Fi 6 Router",
        "Action Camera", "LED Desk Lamp", "Smart Plug Strip", "Bluetooth Speaker",
        "Wireless Mouse", "Gaming Headset", "Monitor Stand", "Cable Organizer",
        "Phone Stand", "Laptop Dock", "External SSD", "Memory Card Reader",
        "USB 3.0 Hub", "Portable Projector", "Smart Display"
    ],
    "features": [
        ["Active Noise Cancellation", "30-Hour Battery Life"],
        ["Adaptive EQ", "IPX5 Water Resistance"],
        ["Fast Charging", "Bluetooth 5.3"],
        ["RGB Backlighting", "AI-Enhanced Mic"],
        ["Ultra-Low Latency", "Hi-Res Audio"],
        ["Multi-Device Pairing", "Auto Game Mode"],
        ["360° Spatial Audio", "Matter Smart Home Support"],
        ["Thunderbolt 4 Pass-Through", "Dual Type-C Ports"],
        ["120Hz Refresh Rate", "USB Power Delivery"],
        ["Wireless Charging", "Sleep Mode"],
        ["Motion Tracking", "Voice Control"],
        ["Temperature Regulation", "Eco Mode"],
        ["Quick Connect", "One-Click Pairing"],
        ["Silent Operation", "Customizable Profiles"]
    ],
    "details": [
        "6-core DSP chip", "1.5\" bio-cellulose driver", "Tactile optical switch",
        "PixArt 3395 sensor", "Sony STARVIS 2 sensor", "Graphene-coated membrane",
        "Dual antenna design", "AI noise suppression", "Contextual awareness",
        "Adaptive power management", "Real-time sync", "Cloud backup"
    ],
    "taglines": [
        "Perfect for remote workers, commuters, and audiophiles alike.",
        "Dominate every session with tournament-grade precision.",
        "Your creative workflow, supercharged.",
        "Where premium sound meets everyday convenience.",
        "Engineered to keep up with your ambitions.",
        "Silence the world. Own your focus.",
        "Built for professionals. Loved by gamers.",
        "Experience audio like never before.",
        "The future of connectivity is here."
    ],
    "price_range": (15.99, 999.99),
}

APPAREL = {
    "brands": [
        "Nike", "Adidas", "Levi's", "Uniqlo", "Patagonia", "The North Face",
        "Columbia", "Under Armour", "Champion", "Carhartt", "Everlane", "Outdoor Voices",
        "Arc'teryx", "Rab", "Black Diamond", "Salomon", "REI", "Gap", "H&M"
    ],
    "lines": [
        "AeroFit", "UrbanEdge", "TrailBlaze", "EcoWeave", "FlexCore", "CloudStretch",
        "AlpineShield", "PureCotton", "DriLayer", "FrostGuard", "VentureCore",
        "ClassicEssence", "PerformancePro", "WanderlustGear"
    ],
    "forms": [
        "Running Jacket", "Merino Wool Crew-Neck Sweater", "5-Pocket Slim-Fit Jeans",
        "Compression Training Tights", "Waterproof Shell Parka", "Organic Cotton T-Shirt",
        "Insulated Puffer Vest", "Lightweight Cargo Shorts", "Thermal Base-Layer Set",
        "Performance Polo Shirt", "Fleece Quarter-Zip Pullover", "Windproof Softshell Pants",
        "Seamless Sports Bra", "Technical Running Shorts", "Hiking Boots",
        "Casual Canvas Sneakers", "Wool Blend Beanie", "Moisture-Wicking Tank Top"
    ],
    "features": [
        ["moisture-wicking technology", "four-way stretch fabric"],
        ["wind-resistant outer shell", "UPF 50+ sun protection"],
        ["recycled material construction", "seamless flatlock stitching"],
        ["reflective safety detailing", "hidden zip pockets"],
        ["adjustable hood and cuffs", "anti-odour silver-ion treatment"],
        ["articulated knee panelling", "breathable mesh lining"],
        ["water-repellent coating", "taped seams"],
        ["ergonomic design", "compression fit"],
        ["breathable lining", "durable construction"]
    ],
    "details": [
        "88% recycled polyester / 12% elastane",
        "100% GOTS-certified organic cotton",
        "Merino wool sourced from ZQ-certified farms",
        "DWR-treated ripstop nylon",
        "Bio-based nylon derived from castor oil",
        "European manufacturing", "Vegan leather accents"
    ],
    "taglines": [
        "Move freely. Look sharp. Feel great.",
        "Built for the planet, built for performance.",
        "Comfort that keeps pace with your lifestyle.",
        "Where function meets timeless style.",
        "Gear up for whatever comes next.",
        "Engineered for adventure.",
        "Your second skin for every challenge."
    ],
    "price_range": (12.99, 349.99),
}

HOME = {
    "brands": [
        "Instant Pot", "Cuisinart", "iRobot", "Dyson", "Philips", "KitchenAid",
        "Breville", "Nespresso", "Shark", "OXO", "Vitamix", "Le Creuset",
        "GreenPan", "All-Clad", "Calphalon", "Zwilling", "Staub", "Oxo", "Rubbermaid"
    ],
    "lines": [
        "SmartHome", "ProChef", "PureClean", "ZenSpace", "EcoLiving", "ClearAir",
        "IntelliCook", "TranquilSleep", "BrightKitchen", "EasyClean", "NestComfort"
    ],
    "forms": [
        "Air Purifier", "Robot Vacuum & Mop Combo", "Stand Mixer", "Programmable Coffee Maker",
        "Sous Vide Precision Cooker", "Cordless Stick Vacuum", "Smart Air Fryer",
        "Weighted Blanket", "Bamboo Cutting Board Set", "Cast-Iron Dutch Oven",
        "Countertop Bread Maker", "Water Filter Pitcher", "Smart Thermostat",
        "Adjustable Desk Lamp", "Air Humidifier", "Dishwasher-Safe Cookware Set",
        "Slow Cooker", "Blender", "Toaster Oven", "Microwave Steamer"
    ],
    "features": [
        ["HEPA H13 filtration", "LiDAR-guided navigation"],
        ["10-speed precision motor", "30-bar pressure extraction"],
        ["±0.1°C temperature accuracy", "75 dB ultra-quiet operation"],
        ["self-emptying dustbin", "7-in-1 cooking modes"],
        ["real-time air-quality display", "room-mapping with no-go zones"],
        ["tilt-head attachment hub", "built-in milk frother"],
        ["app-based scheduling", "voice-assistant integration"],
        ["dishwasher-safe components", "child-safety auto shut-off"],
        ["Timer function", "Energy efficient"]
    ],
    "details": [
        "aircraft-grade anodised aluminium",
        "food-grade stainless steel 18/10",
        "BPA-free Tritan co-polyester",
        "enamelled cast iron",
        "sustainably sourced moso bamboo",
        "Non-stick ceramic coating",
        "Lifetime warranty"
    ],
    "taglines": [
        "Transform your home into a smart sanctuary.",
        "Spend less time cleaning, more time living.",
        "Precision craftsmanship for everyday cooking.",
        "Barista-quality espresso at the touch of a button.",
        "Restaurant-grade sous vide cooking for home chefs.",
        "All-day comfort with temperature-regulating fill.",
        "Designed to simplify. Built to last.",
        "Your kitchen, perfected."
    ],
    "price_range": (9.99, 599.99),
}

BOOKS = {
    "authors": [
        "Dr. Priya Nair", "James Calloway", "Elena Vasquez", "Dr. Marcus Webb",
        "Aisha Okonkwo", "Thomas Lin", "Dr. Sarah Brennan", "Ravi Patel",
        "Claire Fontaine", "David Osei", "Maya Krishnamurthy", "Prof. Neil Hartley",
        "Jessica Chen", "Michael Torres", "Dr. Amelia Grant", "Robert Hayes",
        "Sophia Müller", "Dr. Kwame Asante", "Isabella Romano", "Dr. Viktor Petrov"
    ],
    "titles": [
        "The Architecture of Tomorrow", "Deep Roots: A Memoir of Soil and Silicon",
        "Thinking in Systems", "The Last Algorithm", "Untangling Complexity",
        "The Quiet Revolution", "Code & Culture", "Rethinking Intelligence",
        "The Design of Everyday Decisions", "When Data Speaks", "The Human Edge",
        "Patterns of Resilience", "Zero to One Hundred", "The Empathy Advantage",
        "Signals & Noise", "The Future of Work", "Mindful Machines",
        "The Storyteller's Science", "Digital Horizons", "Connected Lives"
    ],
    "subtitles": [
        "Building Tomorrow's World Today",
        "Lessons from Unexpected Places",
        "A Practical Guide",
        "The Last Stand Against Automation",
        "Finding Clarity in Chaos",
        "How Small Changes Create Big Impact",
        "The Intersection of Creation and Technology",
        "What AI Can and Cannot Do",
        "How Great Products Get Built",
        "The Truth Hidden in Numbers",
        "Why Humans Matter More Than Ever",
        "Stories of Survival and Growth",
        "From Startup to Scale",
        "Building Connection in a Digital World",
        "Reading Between the Data Points",
        "Remote, Async, and Thriving",
        "AI That Understands Humans",
        "The Art of Narrative in Business",
        "The Shift to Digital-First Living",
        "Building Community in Isolation"
    ],
    "forms": [
        "hardcover", "paperback", "audiobook", "e-book", "leather-bound edition",
        "illustrated edition", "collector's edition", "signed edition"
    ],
    "genres": [
        "Technology & Innovation", "Business & Economics", "Science & Nature",
        "Psychology & Behavior", "Self-Help & Personal Development", "Memoir & Biography",
        "Philosophy & Ethics", "Creative Writing", "History & Culture", "Health & Wellness"
    ],
    "features": [
        ["Foreword by industry leader", "Extensive bibliography"],
        ["350+ pages of research", "Practical case studies"],
        ["Interview sections", "Reflection questions"],
        ["Full-color illustrations", "Premium paper"],
        ["Audiobook narration by author", "Study guide included"],
        ["Companion website access", "Reading group materials"],
        ["Handwritten annotations", "Limited print run"]
    ],
    "taglines": [
        "'A landmark work for our times' — The Economist.",
        "'Essential reading for anyone curious about the future' — MIT Tech Review.",
        "'Transformative, urgent, and beautifully written' — The Guardian.",
        "'The definitive guide to navigating complexity' — Harvard Business Review.",
        "'A must-read that will change how you see the world' — Nature.",
        "'Gripping and revelatory' — The New York Times.",
        "'A triumph of storytelling and insight' — Financial Times."
    ],
    "price_range": (6.99, 79.99),
}

SPORTS = {
    "brands": [
        "Wilson", "Spalding", "Rawlings", "Louisville Slugger", "Ping", "TaylorMade",
        "Callaway", "Mizuno", "Yonex", "Head", "Babolat", "Prince", "Dunlop",
        "Evenflo", "Lifetime", "Spalding", "Spalding", "Huffy", "Mongoose"
    ],
    "lines": [
        "Champion", "Professional", "Elite", "Standard", "Training", "Recreational"
    ],
    "forms": [
        "Basketball", "Football", "Soccer Ball", "Baseball Glove", "Golf Club Set",
        "Tennis Racket", "Badminton Racket", "Skateboard", "Ice Hockey Stick",
        "Baseball Bat", "Cricket Bat", "Bowling Ball", "Ping Pong Paddle",
        "Yoga Mat", "Resistance Bands Set", "Dumbbells", "Jump Rope",
        "Inline Skates", "Bicycle Helmet", "Goalkeeper Gloves"
    ],
    "features": [
        ["Professional grade", "Tournament approved"],
        ["Lightweight design", "Enhanced grip"],
        ["Durable construction", "Weather resistant"],
        ["Comfort padding", "Adjustable fit"],
        ["High visibility", "Safety certified"]
    ],
    "details": [
        "Premium rubber construction", "Reinforced stitching",
        "Ergonomic design", "Weather-resistant material"
    ],
    "taglines": [
        "Built for champions.", "Performance guaranteed.",
        "Elevate your game.", "Where athletes train."
    ],
    "price_range": (9.99, 899.99),
}

BEAUTY = {
    "brands": [
        "MAC", "Sephora", "Urban Decay", "Too Faced", "Estée Lauder", "Lancome",
        "Dior", "Chanel", "L'Oréal", "Maybelline", "Charlotte Tilbury", "Fenty",
        "Nars", "Bobbi Brown", "Benefit", "Shu Uemura"
    ],
    "lines": [
        "Luxe", "Essential", "Classic", "Modern", "Natural", "Bold"
    ],
    "forms": [
        "Liquid Foundation", "Eyeshadow Palette", "Lipstick Collection", "Mascara",
        "Blush Palette", "Contour Kit", "Highlighter Set", "Face Serum",
        "Moisturizing Cream", "Face Mask", "Hair Conditioner", "Body Lotion",
        "Perfume", "Eyeliner Set", "Eyebrow Pencil", "Concealer"
    ],
    "features": [
        ["Long-wearing formula", "Waterproof"],
        ["Cruelty-free", "Vegan ingredients"],
        ["SPF protection", "Anti-aging"],
        ["Hydrating formula", "Lightweight"],
        ["Buildable coverage", "Shade range"]
    ],
    "details": [
        "Professional quality formula", "Dermatologist tested",
        "Hypoallergenic ingredients", "Luxurious texture"
    ],
    "taglines": [
        "Beauty elevated.", "Your best self awaits.",
        "Confidence starts here.", "Unlock your glow."
    ],
    "price_range": (8.99, 199.99),
}

TOYS = {
    "brands": [
        "LEGO", "Mattel", "Hasbro", "Funko", "Bandai", "Takara Tomy",
        "MelissaandDoug", "Playmobil", "Hot Wheels"
    ],
    "lines": [
        "Classic", "Deluxe", "Collection", "Premium", "Standard", "Special Edition"
    ],
    "forms": [
        "Building Blocks Set", "Action Figure", "Puzzle", "Board Game",
        "Plush Toy", "Collectible Figure", "Remote Control Car", "Dollhouse",
        "Model Kit", "Educational Toy", "Fidget Toy", "Building Kit"
    ],
    "features": [
        ["Age appropriate", "Educational value"],
        ["Durable materials", "Safety tested"],
        ["Collectible item", "Limited edition"],
        ["Interactive features", "Multiple pieces"]
    ],
    "details": [
        "Premium construction", "Safe for all ages",
        "Award-winning design", "Family-friendly"
    ],
    "taglines": [
        "Creativity unleashed.", "Play, learn, grow.",
        "Fun for the whole family.", "Imagination in every box."
    ],
    "price_range": (4.99, 299.99),
}

FOOD = {
    "brands": [
        "Organic Valley", "Annie's Homegrown", "Bob's Red Mill", "Nature's Path",
        "Dole", "Chiquita", "Vital Farms", "Barrington", "Newman's Own",
        "Starbucks", "Green Mountain", "Fair Trade"
    ],
    "lines": [
        "Organic", "Premium", "Pure", "Essential", "Select"
    ],
    "forms": [
        "Organic Coffee Beans", "Whole Grain Cereal", "Almond Butter", "Chia Seeds",
        "Organic Pasta", "Raw Almonds", "Coconut Oil", "Maple Syrup",
        "Dark Chocolate", "Herbal Tea Set", "Granola Mix", "Protein Powder"
    ],
    "features": [
        ["Organic certified", "Non-GMO"],
        ["Fair trade", "Sustainably sourced"],
        ["Gluten-free", "Vegan"],
        ["No artificial additives", "Cold-pressed"]
    ],
    "details": [
        "Premium sourced ingredients", "Ethically produced",
        "Natural formulation", "Sustainably grown"
    ],
    "taglines": [
        "Nourish naturally.", "Taste the difference.",
        "Farm to table quality.", "Health in every bite."
    ],
    "price_range": (2.99, 89.99),
}

CATEGORY_DATA = {
    "Electronics": ELECTRONICS,
    "Apparel": APPAREL,
    "Home": HOME,
    "Books": BOOKS,
    "Sports": SPORTS,
    "Beauty": BEAUTY,
    "Toys": TOYS,
    "Food": FOOD,
}

# ── User Personas (Better taste clusters) ───────────────────────────────────────

USER_PERSONAS = {
    "Tech Enthusiasts": list(range(1, 16)),          # 15 users
    "Fashion Conscious": list(range(16, 31)),        # 15 users
    "Home & Living": list(range(31, 46)),            # 15 users
    "Bookworms": list(range(46, 61)),                # 15 users
    "Sports & Fitness": list(range(61, 76)),         # 15 users
    "Beauty & Wellness": list(range(76, 91)),        # 15 users
    "Omnivores": list(range(91, 101)),               # 10 users (diverse tastes)
}

CROSS_USERS = list(range(91, 101))  # Users who rate across all categories


# ── Helper Functions ───────────────────────────────────────────────────────────

def _pick(lst):
    """Randomly pick from list."""
    return random.choice(lst)


def _generate_product_row(product_id, category):
    """Generate a realistic product row."""
    data = CATEGORY_DATA[category]

    if category == "Books":
        author = _pick(data["authors"])
        title = _pick(data["titles"])
        subtitle = _pick(data["subtitles"])
        form = _pick(data["forms"])
        genre = _pick(data["genres"])
        feature1, feature2 = _pick(data["features"])
        tagline = _pick(data["taglines"])

        full_title = f"{title}: {subtitle} ({form})"
        description = (
            f"By {author}.\n\n"
            f"Genre: {genre}\n\n"
            f"This compelling work explores {genre.lower()}. "
            f"Features {feature1} and {feature2}. "
            f"{tagline}"
        )
        price = round(random.uniform(*data["price_range"]), 2)

        return {
            "product_id": product_id,
            "title": full_title,
            "description": description,
            "category": category,
            "price": price,
        }

    else:
        # Electronics, Apparel, Home, etc.
        brand = _pick(data["brands"])
        line = _pick(data["lines"])
        form = _pick(data["forms"])
        feat1, feat2 = _pick(data["features"])
        detail = _pick(data.get("details", ["premium quality"]))
        tagline = _pick(data.get("taglines", ["Designed for you."]))

        title = f"{brand} {line} {form} – {feat1} & {feat2}"
        description = (
            f"The {brand} {line} {form} delivers {feat1} and {feat2}. "
            f"Crafted with {detail}, this product combines quality and performance. "
            f"{tagline}"
        )
        price = round(random.uniform(*data["price_range"]), 2)

        return {
            "product_id": product_id,
            "title": title,
            "description": description,
            "category": category,
            "price": price,
        }


def generate_products(n=NUM_PRODUCTS):
    """Generate n products with balanced category distribution."""
    per_cat = n // len(CATEGORIES)
    cats = CATEGORIES * per_cat
    leftover = n - len(cats)
    cats += random.choices(CATEGORIES, k=leftover)
    random.shuffle(cats)

    rows = [_generate_product_row(pid + 1, cat) for pid, cat in enumerate(cats)]
    df = pd.DataFrame(rows)
    return df


def _rating_for_user_product(user_id, category):
    """Generate biased rating based on user preference."""
    user_persona = next(
        (persona for persona, uids in USER_PERSONAS.items() if user_id in uids),
        None,
    )

    # Map personas to preferred categories
    persona_to_category = {
        "Tech Enthusiasts": "Electronics",
        "Fashion Conscious": "Apparel",
        "Home & Living": "Home",
        "Bookworms": "Books",
        "Sports & Fitness": "Sports",
        "Beauty & Wellness": "Beauty",
        "Omnivores": None,  # No preference
    }

    preferred_cat = persona_to_category.get(user_persona)

    if preferred_cat == category or user_id in CROSS_USERS:
        # Preferred category
        return int(np.clip(np.random.normal(loc=4.3, scale=0.6), 1, 5).round())
    else:
        # Non-preferred category
        return int(np.clip(np.random.normal(loc=2.8, scale=1.3), 1, 5).round())


def generate_ratings(products_df):
    """Generate sparse ratings matrix with taste clusters."""
    records = []

    cat_product_ids = {
        cat: products_df.loc[products_df["category"] == cat, "product_id"].tolist()
        for cat in CATEGORIES
    }
    all_product_ids = products_df["product_id"].tolist()

    base_n = max(1, int(NUM_PRODUCTS * BASE_SPARSITY))

    for user_id in range(1, NUM_USERS + 1):
        user_persona = next(
            (persona for persona, uids in USER_PERSONAS.items() if user_id in uids),
            None,
        )

        if user_id in CROSS_USERS:
            # Omnivores: broad sampling
            n_to_rate = base_n + random.randint(5, 20)
            sampled_ids = random.sample(all_product_ids, min(n_to_rate, NUM_PRODUCTS))
        else:
            # Build weighted pool (3x weight on preferred category)
            persona_to_category = {
                "Tech Enthusiasts": "Electronics",
                "Fashion Conscious": "Apparel",
                "Home & Living": "Home",
                "Bookworms": "Books",
                "Sports & Fitness": "Sports",
                "Beauty & Wellness": "Beauty",
            }
            preferred_cat = persona_to_category.get(user_persona)

            weighted_pool = []
            for cat, pids in cat_product_ids.items():
                weight = 3 if cat == preferred_cat else 1
                weighted_pool.extend(pids * weight)

            n_to_rate = base_n + random.randint(0, 15)
            seen = set()
            sampled_ids = []
            random.shuffle(weighted_pool)
            for pid in weighted_pool:
                if pid not in seen:
                    seen.add(pid)
                    sampled_ids.append(pid)
                if len(sampled_ids) >= n_to_rate:
                    break

        for pid in sampled_ids:
            cat = products_df.loc[products_df["product_id"] == pid, "category"].values[0]
            rating = _rating_for_user_product(user_id, cat)
            records.append(
                {"user_id": user_id, "product_id": pid, "rating": rating}
            )

    df = pd.DataFrame(records).drop_duplicates(subset=["user_id", "product_id"])
    df = df.sort_values(["user_id", "product_id"]).reset_index(drop=True)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Upgraded Dataset Generator — Multi-Modal Rec Engine v2")
    print(f"  Random seed: {RANDOM_SEED}")
    print("=" * 70)

    # ── Products ──────────────────────────────────────────────────────────────
    print("\n[1/2] Generating products …")
    products_df = generate_products(NUM_PRODUCTS)
    products_df.to_csv(PRODUCTS_FILE, index=False)

    print(f"      Saved {len(products_df)} rows → '{PRODUCTS_FILE}'")
    print("      Category distribution:")
    for cat, cnt in sorted(products_df["category"].value_counts().items()):
        pct = 100 * cnt / len(products_df)
        print(f"        {cat:<18} {cnt:>3} products ({pct:>5.1f}%)")
    print(f"      Price range: ${products_df['price'].min():.2f} – "
          f"${products_df['price'].max():.2f}")

    # ── Ratings ───────────────────────────────────────────────────────────────
    print("\n[2/2] Generating ratings …")
    ratings_df = generate_ratings(products_df)
    ratings_df.to_csv(RATINGS_FILE, index=False)

    total_possible = NUM_USERS * NUM_PRODUCTS
    density = len(ratings_df) / total_possible * 100

    print(f"      Saved {len(ratings_df)} rows → '{RATINGS_FILE}'")
    print(f"      Matrix density : {density:.2f}%  "
          f"({len(ratings_df)} / {total_possible} interactions)")
    print(f"      Users          : {ratings_df['user_id'].nunique()}")
    print(f"      Rated products : {ratings_df['product_id'].nunique()} "
          f"/ {NUM_PRODUCTS}")
    print(f"      Rating dist    :")
    for r in range(1, 6):
        cnt = (ratings_df["rating"] == r).sum()
        pct = 100 * cnt / len(ratings_df)
        bar = "█" * (cnt // 10)
        print(f"        ★{r}  {cnt:>5} ({pct:>5.1f}%)  {bar}")

    # ── User personas ──────────────────────────────────────────────────────────
    print(f"\n      User personas:")
    for persona, uids in USER_PERSONAS.items():
        print(f"        {persona:<20} users {min(uids):>3}-{max(uids):<3} ({len(uids):>2} users)")

    print("\n✓ Upgraded dataset ready!")
    print(f"  {NUM_PRODUCTS} products × {NUM_USERS} users with richer diversity\n")


if __name__ == "__main__":
    main()
