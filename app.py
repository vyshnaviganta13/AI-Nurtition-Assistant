# app.py
import io
import difflib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing.image import img_to_array


def generate_meal_plan(conditions):
    """Generate a simple meal plan for selected conditions."""
    meal_plans = {
        "kidney": {
            "Breakfast": ["Oatmeal", "Apple"],
            "Lunch": ["Grilled Chicken", "Cauliflower Rice"],
            "Dinner": ["Steamed Fish", "Cabbage Salad"]
        },
        "diabetes": {
            "Breakfast": ["Greek Yogurt", "Berries"],
            "Lunch": ["Quinoa Salad", "Leafy Greens"],
            "Dinner": ["Grilled Salmon", "Brown Rice"]
        },
        "hypertension": {
            "Breakfast": ["Banana Smoothie"],
            "Lunch": ["Oats with Vegetables"],
            "Dinner": ["Beetroot Soup", "Grilled Chicken"]
        },
        "obesity": {
            "Breakfast": ["Boiled Eggs", "Apple"],
            "Lunch": ["Grilled Vegetables", "Sweet Potato"],
            "Dinner": ["Broccoli Soup", "Steamed Chicken"]
        },
        "fitness": {
            "Breakfast": ["Egg Whites", "Oats"],
            "Lunch": ["Chicken Breast", "Quinoa"],
            "Dinner": ["Fish", "Greek Yogurt"]
        }
    }

    # For multiple conditions, merge plans
    final_plan = {}
    for condition in conditions:
        for meal, items in meal_plans.get(condition, {}).items():
            if meal not in final_plan:
                final_plan[meal] = []
            final_plan[meal].extend(items)

    return final_plan

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="AI Health Assistant", page_icon="🍴", layout="centered")
st.title("🍴 AI-based Health & Nutrition Assistant")
st.caption("Upload a food image → AI predicts the food → we show nutrition & condition-based advice.")

# Reduce TF logging noise (optional)
tf.get_logger().setLevel("ERROR")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return MobileNetV2(weights="imagenet")

@st.cache_data(show_spinner=False)
def load_nutrition_db():
    """
    Load a nutrition CSV and normalize columns to:
    ['Food','Calories','Protein','Fat','Carbs'].
    Accepts variations like Dish/Name, kcal/Energy, etc.
    Also adds 'food_norm' for case-insensitive matching.
    """
    # Try common filenames
    candidates = [
        Path("extended_nutrition.csv"),
        Path("nutrition.csv"),
        Path("food_nutrition_db.csv"),
    ]
    csv_path = next((p for p in candidates if p.exists()), None)

    if csv_path is None:
        # Minimal fallback so the app still runs
        data = [
            {"Food": "Pizza", "Calories": 266, "Protein": 11, "Fat": 10, "Carbs": 33},
            {"Food": "Carbonara", "Calories": 379, "Protein": 13, "Fat": 15, "Carbs": 52},
            {"Food": "Biryani", "Calories": 290, "Protein": 9, "Fat": 12, "Carbs": 35},
            {"Food": "Idli", "Calories": 70, "Protein": 2, "Fat": 0.5, "Carbs": 15},
            {"Food": "Dosa", "Calories": 170, "Protein": 3, "Fat": 6, "Carbs": 28},
        ]
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(csv_path)

    # Normalize column names
    lower_cols = {c.lower(): c for c in df.columns}
    def find_col(possible):
        for name in possible:
            if name in lower_cols:
                return lower_cols[name]
        return None

    food_col = find_col(["food", "dish", "name", "item"])
    cal_col  = find_col(["calories", "kcal", "energy"])
    pro_col  = find_col(["protein", "proteins"])
    fat_col  = find_col(["fat", "fats", "total_fat"])
    carb_col = find_col(["carbs", "carbohydrate", "carbohydrates", "total_carbohydrate"])

    # Build a normalized frame with safe defaults
    norm = pd.DataFrame()
    if food_col is None:
        # fabricate 'Food' if missing to avoid KeyError
        norm["Food"] = df.iloc[:, 0].astype(str)
    else:
        norm["Food"] = df[food_col].astype(str)

    # Coerce to numeric; invalid → NaN
    norm["Calories"] = pd.to_numeric(df[cal_col], errors="coerce") if cal_col else np.nan
    norm["Protein"]  = pd.to_numeric(df[pro_col], errors="coerce") if pro_col else np.nan
    norm["Fat"]      = pd.to_numeric(df[fat_col], errors="coerce") if fat_col else np.nan
    norm["Carbs"]    = pd.to_numeric(df[carb_col], errors="coerce") if carb_col else np.nan

    # Clean/trim names and add normalized search key
    norm["Food"] = norm["Food"].str.strip()
    norm["food_norm"] = norm["Food"].str.lower().str.replace(r"\s+", " ", regex=True)

    # Drop duplicates keeping first
    norm = norm.drop_duplicates(subset=["food_norm"], keep="first").reset_index(drop=True)
    return norm

@st.cache_data(show_spinner=False)
def get_close_foods(query: str, foods: list, n=5):
    return difflib.get_close_matches(query.lower(), [f.lower() for f in foods], n=n, cutoff=0.5)

def safe_float(x):
    """Convert to float if possible, else return None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def predict_food_from_image(uploaded_file) -> dict:
    """
    Returns dict:
    {
      "top1": "Pizza",
      "top3": ["Pizza", "Carbonara", "Hotdog"]
    }
    """
    model = load_model()
    # Read bytes and keep buffer for PIL
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Prepare for model
    resized = pil_img.resize((224, 224))
    x = img_to_array(resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    dec = decode_predictions(preds, top=3)[0]
    labels = [lbl for (_, lbl, _) in dec]

    # Map some ImageNet labels to common dish names
    label_map = {
        "hotdog": "Hot Dog",
        "cheeseburger": "Burger",
        "hamburger": "Burger",
        "carbonara": "Carbonara",
        "pizza": "Pizza",
        "spaghetti_squash": "Spaghetti",
        "ramen": "Ramen",
        "sushi": "Sushi",
        "guacamole": "Guacamole",
        "bagel": "Bagel",
        "cup": "Cup",  # usually not food, but included
    }
    top3 = [label_map.get(l, l.replace("_", " ").title()) for l in labels]
    return {"top1": top3[0], "top3": top3}

def lookup_food(food_name: str, db: pd.DataFrame):
    """
    Lookup nutrition by name (exact, then partial, then fuzzy).
    Returns dict:
    {
      "found": True/False,
      "food": "Pizza",
      "row": {"Calories":..., "Protein":..., "Fat":..., "Carbs":...},
      "suggestions": [...]
    }
    """
    if not isinstance(food_name, str) or not food_name.strip():
        return {"found": False, "food": "", "row": {}, "suggestions": []}

    q = food_name.strip().lower()
    # Exact match
    exact = db[db["food_norm"] == q]
    if not exact.empty:
        r = exact.iloc[0]
        row = {k: safe_float(r[k]) for k in ["Calories", "Protein", "Fat", "Carbs"]}
        return {"found": True, "food": r["Food"], "row": row, "suggestions": []}

    # Partial contains
    contains = db[db["food_norm"].str.contains(q, na=False)]
    if not contains.empty:
        r = contains.iloc[0]
        row = {k: safe_float(r[k]) for k in ["Calories", "Protein", "Fat", "Carbs"]}
        return {"found": True, "food": r["Food"], "row": row, "suggestions": []}

    # Fuzzy suggestions
    sugg = get_close_foods(q, db["Food"].tolist(), n=5)
    return {"found": False, "food": food_name, "row": {}, "suggestions": sugg}

# -----------------------------
# Health rules
# -----------------------------
HEALTH_RULES = {
    "diabetes": {"carbs_limit": 40, "advice": "High carbs may spike blood sugar. Prefer whole grains & fiber."},
    "hypertension": {"fat_limit": 15, "advice": "High fat/sodium may raise BP. Prefer grilled/boiled foods."},
    "obesity": {"calories_limit": 300, "advice": "High calories → weight gain risk. Try smaller portions."},
    "kidney": {"protein_limit": 20, "advice": "Excess protein stresses kidneys. Prefer balanced meals."},
    "fitness": {"balanced": True, "advice": "Target ~45–65% carbs, 10–35% protein, 20–35% fat."}
}

def check_health_conditions(food: str, nutrition: dict, user_conditions: list):
    """
    nutrition values may be None; only compare when numeric is available.
    Returns list of message strings.
    """
    results = []
    cal = safe_float(nutrition.get("Calories"))
    pro = safe_float(nutrition.get("Protein"))
    fat = safe_float(nutrition.get("Fat"))
    carb = safe_float(nutrition.get("Carbs"))

    for cond in user_conditions:
        rule = HEALTH_RULES.get(cond)
        if not rule:
            continue

        if cond == "diabetes" and carb is not None and carb > rule["carbs_limit"]:
            results.append(f"{food} has {carb:.1f}g carbs/100g. {rule['advice']}")

        if cond == "hypertension" and fat is not None and fat > rule["fat_limit"]:
            results.append(f"{food} has {fat:.1f}g fat/100g. {rule['advice']}")

        if cond == "obesity" and cal is not None and cal > rule["calories_limit"]:
            results.append(f"{food} has {cal:.0f} kcal/100g. {rule['advice']}")

        if cond == "kidney" and pro is not None and pro > rule["protein_limit"]:
            results.append(f"{food} has {pro:.1f}g protein/100g. {rule['advice']}")

        if cond == "fitness":
            # Macro balance (only if all present)
            vals = [v for v in [carb, pro, fat] if v is not None]
            if len(vals) == 3:
                total = carb + pro + fat
                if total > 0:
                    results.append(
                        f"ℹ️ Macro split (per 100g): {carb/total*100:.0f}% carbs, "
                        f"{pro/total*100:.0f}% protein, {fat/total*100:.0f}% fat. {rule['advice']}"
                    )

    return results
def suggest_alternatives(db: pd.DataFrame, current_food: str, row: dict, condition: str, top_n: int = 5):
    """
    If the current food violates a condition, suggest up to top_n better options
    from the same DB. Uses the same thresholds as HEALTH_RULES.
    Returns a simple list of food names.
    """
    # Current food macros (safe)
    cal = safe_float(row.get("Calories"))
    pro = safe_float(row.get("Protein"))
    fat = safe_float(row.get("Fat"))
    carb = safe_float(row.get("Carbs"))

    # Numeric-only copy
    df = db.copy()
    for col in ["Calories", "Protein", "Fat", "Carbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Exclude current food
    df = df[df["Food"].str.lower() != str(current_food).lower()]

    if condition == "diabetes":
        limit = HEALTH_RULES["diabetes"]["carbs_limit"]
        if carb is None or carb <= limit:
            return []
        # show items clearly below both the limit and current item
        cutoff = min(limit, carb * 0.8)
        ranked = df[df["Carbs"].notna() & (df["Carbs"] < cutoff)].sort_values("Carbs", ascending=True)

    elif condition == "hypertension":
        limit = HEALTH_RULES["hypertension"]["fat_limit"]
        if fat is None or fat <= limit:
            return []
        cutoff = min(limit, fat * 0.8)
        ranked = df[df["Fat"].notna() & (df["Fat"] < cutoff)].sort_values("Fat", ascending=True)

    elif condition == "obesity":
        limit = HEALTH_RULES["obesity"]["calories_limit"]
        if cal is None or cal <= limit:
            return []
        cutoff = min(limit, cal * 0.85)
        ranked = df[df["Calories"].notna() & (df["Calories"] < cutoff)].sort_values("Calories", ascending=True)

    elif condition == "kidney":
        limit = HEALTH_RULES["kidney"]["protein_limit"]
        if pro is None or pro <= limit:
            return []
        cutoff = min(limit, pro * 0.8)
        ranked = df[df["Protein"].notna() & (df["Protein"] < cutoff)].sort_values("Protein", ascending=True)

    else:
        return []

    return ranked["Food"].head(top_n).tolist()


def generate_meal_plan(db: pd.DataFrame, conditions: list, target_calories: int = 1800):
    """
    Simple day plan: Breakfast, Lunch, Snack, Dinner.
    Filters DB by selected conditions (AND logic). Picks one item per meal.
    Scales macros by reasonable default portions. Returns (plan_df, totals_series).
    """
    # Numeric copy
    df = db.copy()
    for col in ["Calories", "Protein", "Fat", "Carbs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply AND filters based on conditions
    filt = df.copy()
    if "diabetes" in conditions:
        filt = filt[filt["Carbs"].notna() & (filt["Carbs"] <= HEALTH_RULES["diabetes"]["carbs_limit"])]
    if "hypertension" in conditions:
        filt = filt[filt["Fat"].notna() & (filt["Fat"] <= HEALTH_RULES["hypertension"]["fat_limit"])]
    if "obesity" in conditions:
        filt = filt[filt["Calories"].notna() & (filt["Calories"] <= HEALTH_RULES["obesity"]["calories_limit"])]
    if "kidney" in conditions:
        filt = filt[filt["Protein"].notna() & (filt["Protein"] <= HEALTH_RULES["kidney"]["protein_limit"])]

    # Fallback if too strict
    if filt.empty:
        # choose lower-calorie options as a safe baseline
        filt = df.nsmallest(min(50, len(df)), "Calories", keep="all").dropna(subset=["Calories"])

    # Default portions (grams)
    meals = [("Breakfast", 250), ("Lunch", 350), ("Snack", 150), ("Dinner", 350)]

    # Random but avoid repeats
    rng = np.random.default_rng()
    used = set()
    plan_rows = []
    for m_name, grams in meals:
        pool = filt[~filt["Food"].str.lower().isin(used)]
        if pool.empty:
            pool = filt
        row = pool.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        used.add(row["Food"].lower())

        factor = grams / 100.0
        plan_rows.append({
            "Meal": m_name,
            "Food": row["Food"],
            "Portion (g)": grams,
            "Calories": round(row["Calories"] * factor, 1) if pd.notna(row["Calories"]) else None,
            "Protein": round(row["Protein"] * factor, 1) if pd.notna(row["Protein"]) else None,
            "Fat": round(row["Fat"] * factor, 1) if pd.notna(row["Fat"]) else None,
            "Carbs": round(row["Carbs"] * factor, 1) if pd.notna(row["Carbs"]) else None,
        })

    plan_df = pd.DataFrame(plan_rows)
    totals = plan_df[["Calories", "Protein", "Fat", "Carbs"]].sum(numeric_only=True)

    # Optional light normalization toward target calories (informational only)
    # (We simply display totals; no optimization in this simple version.)
    return plan_df, totals


# -----------------------------
# Sidebar – conditions
# -----------------------------
st.sidebar.header("🩺 Select Your Health Conditions")
conditions = st.sidebar.multiselect(
    "Choose conditions:",
    ["diabetes", "hypertension", "obesity", "kidney", "fitness"]
)

# Load DB once
db = load_nutrition_db()

# -----------------------------
# Main – uploader
# -----------------------------
uploaded = st.file_uploader("📸 Upload a food image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Show the image
    uploaded.seek(0)
    st.image(uploaded, caption="Uploaded Food Image", use_container_width=True)
    uploaded.seek(0)

    # Predict
    with st.spinner("🔎 Analyzing image..."):
        pred = predict_food_from_image(uploaded)
    st.success(f"Predicted Food: **{pred['top1']}**")
    st.caption(f"Other guesses: {', '.join(pred['top3'])}")

    # Let user correct/override the name if needed
    corrected = st.text_input("Edit food name if needed:", value=pred["top1"])

    # Lookup nutrition
    result = lookup_food(corrected, db)

    st.subheader("Nutrition Info (per 100g/ml)")


    if result["found"]:
        row = result["row"]
        
        # Fill missing values with 0 for safe display
        calories = row.get("Calories") or 0
        protein  = row.get("Protein") or 0
        fat      = row.get("Fat") or 0
        carbs    = row.get("Carbs") or 0

        # Create a DataFrame with explicit columns
        metrics_df = pd.DataFrame({
            "Nutrient": ["Calories (kcal)", "Protein (g)", "Fat (g)", "Carbs (g)"],
            "Value": [calories, protein, fat, carbs]
        })

        # Display the table
        st.table(metrics_df)

        # Portion scaling
        grams = st.slider("Portion size (grams)", min_value=50, max_value=500, value=200, step=25)
        factor = grams / 100.0

        portion_df = pd.DataFrame({
            "Nutrient": ["Calories (kcal)", "Protein (g)", "Fat (g)", "Carbs (g)"],
            "Value": [round(calories*factor, 1), round(protein*factor, 1), round(fat*factor, 1), round(carbs*factor, 1)]
        })

        st.subheader(f"Estimated for your portion ({grams}g/ml)")
        st.table(portion_df)

        # Health advice
        if conditions:
            st.subheader("Health Advice")
            warnings = check_health_conditions(result["food"], row, conditions)

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("Looks safe for your selected conditions.")

            # Recommended alternatives
            for condition in conditions:
                alternatives = suggest_alternatives(db,result["food"], row, condition)
                if alternatives:
                    st.subheader(f"Healthier Alternatives for {condition.capitalize()}:")
                    st.write(", ".join(alternatives))

            # Personalized daily meal plan
            st.subheader("Personalized Daily Meal Plan")
            plan_df, totals = generate_meal_plan(db, conditions)

        # Display full table
            st.table(plan_df)
        else:
            pass
