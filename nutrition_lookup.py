# nutrition_lookup.py
import os
import requests
import pandas as pd
import difflib
from typing import Optional, Dict, Any, List

# --------- Helper utils ---------
def normalize_label(label: str) -> str:
    """Normalize label from ImageNet/other (replace underscores, lowercase)."""
    return label.replace('_', ' ').strip().lower()

# small mapping to translate common ImageNet labels to friendly food names
COMMON_FOOD_MAP = {
    "pizza": "pizza",
    "cheeseburger": "cheeseburger",
    "hotdog": "hot dog",
    "french loaf": "bread",
    "bagel": "bagel",
    "burrito": "burrito",
    "guacamole": "guacamole",
    "cupcake": "cupcake",
    "ice cream": "ice cream",
    "samosa": "samosa",   # might not appear in ImageNet, but good to map if you add
    "rice": "white rice",
    "plate": None,  # non-food fallback
    "restaurant": None,
    "confectionery": "dessert"
    # extend this map as you test more images
}

# --------- Static DB loader & lookup ---------
def load_static_db(csv_path: str = "food_nutrition_db.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # ensure a lowercase 'food' column copy for safer matching
    df['food_norm'] = df['food'].astype(str).str.lower()
    return df

def lookup_nutrition_static(food_query: str, df: pd.DataFrame, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """Try exact then fuzzy matching in local CSV DB."""
    q = food_query.lower().strip()
    # exact substring match
    subset = df[df['food_norm'].str.contains(q, na=False)]
    if not subset.empty:
        row = subset.iloc[0]
        return dict(row.drop(labels=['food_norm']).to_dict(), source='static_db', matched=row['food'])
    # fuzzy match using difflib
    choices = df['food_norm'].tolist()
    matches = difflib.get_close_matches(q, choices, n=top_k, cutoff=0.6)
    if matches:
        match = matches[0]
        row = df[df['food_norm'] == match].iloc[0]
        return dict(row.drop(labels=['food_norm']).to_dict(), source='static_db', matched=row['food'])
    return None

# --------- USDA API lookup (optional) ---------
def get_nutrition_usda(food_name: str, api_key: str, retries: int = 1) -> Optional[Dict[str, Any]]:
    """
    Uses USDA FoodData Central's search endpoint to find nutrients for a food.
    If you have an API key, set USDA_API_KEY environment variable or pass it.
    """
    if not api_key:
        return None
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {'api_key': api_key, 'query': food_name, 'pageSize': 5}
    try:
        r = requests.get(search_url, params=params, timeout=8)
        if r.status_code != 200:
            # print("USDA search failed:", r.status_code, r.text)
            return None
        data = r.json()
        foods = data.get('foods', [])
        if not foods:
            return None

        # pick best candidate (first for simplicity)
        item = foods[0]
        nutrients = {}
        for n in item.get('foodNutrients', []):
            name = (n.get('nutrientName') or "").lower()
            val = n.get('value') or n.get('amount')
            unit = n.get('unitName') or n.get('unit') or ""
            if val is None:
                continue
            if 'energy' in name or 'calori' in name:
                nutrients['calories_kcal'] = f"{val} {unit}"
            elif 'protein' in name:
                nutrients['protein_g'] = f"{val} {unit}"
            elif 'carbohydrate' in name or 'carb' in name:
                nutrients['carbs_g'] = f"{val} {unit}"
            elif 'total lipid' in name or 'fat' in name:
                nutrients['fat_g'] = f"{val} {unit}"
            elif 'sugar' in name:
                nutrients['sugar_g'] = f"{val} {unit}"
            elif 'sodium' in name or 'salt' in name:
                nutrients['sodium_mg'] = f"{val} {unit}"
        # if energy not found, try detailed endpoint
        if 'calories_kcal' not in nutrients and 'fdcId' in item:
            detail_url = f"https://api.nal.usda.gov/fdc/v1/food/{item['fdcId']}"
            r2 = requests.get(detail_url, params={'api_key': api_key}, timeout=8)
            if r2.status_code == 200:
                fd = r2.json()
                for n in fd.get('foodNutrients', []):
                    name = (n.get('nutrientName') or "").lower()
                    val = n.get('value') or n.get('amount')
                    unit = n.get('unitName') or n.get('unit') or ""
                    if val is None:
                        continue
                    if 'energy' in name or 'calori' in name:
                        nutrients['calories_kcal'] = f"{val} {unit}"
                    elif 'protein' in name:
                        nutrients['protein_g'] = f"{val} {unit}"
                    elif 'carbohydrate' in name or 'carb' in name:
                        nutrients['carbs_g'] = f"{val} {unit}"
                    elif 'total lipid' in name or 'fat' in name:
                        nutrients['fat_g'] = f"{val} {unit}"
                    elif 'sugar' in name:
                        nutrients['sugar_g'] = f"{val} {unit}"
                    elif 'sodium' in name or 'salt' in name:
                        nutrients['sodium_mg'] = f"{val} {unit}"
        if nutrients:
            nutrients['matched_name'] = item.get('description') or item.get('description', '')
            nutrients['source'] = 'usda'
            return nutrients
    except Exception as e:
        # print("USDA lookup error:", e)
        return None
    return None

# --------- High-level lookup that tries USDA then static DB ---------
def lookup_nutrition(food_label: str,
                     use_usda: bool = False,
                     usda_api_key: Optional[str] = None,
                     static_csv_path: str = "nutrition_db.csv",
                     df_static: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Try to find nutrition info for a normalized food_label.
    Returns a dict or None.
    """
    q = normalize_label(food_label)

    # map via COMMON_FOOD_MAP
    mapped = COMMON_FOOD_MAP.get(q, q)
    if mapped is None:
        return None

    # First try USDA if requested
    if use_usda and usda_api_key:
        res = get_nutrition_usda(mapped, usda_api_key)
        if res:
            return res

    # load static db if not provided
    if df_static is None:
        try:
            df_static = load_static_db(static_csv_path)
        except FileNotFoundError:
            return None

    res_static = lookup_nutrition_static(mapped, df_static)
    if res_static:
        return res_static

    # As last resort, try fuzzy query using the label itself (without mapping)
    res_static = lookup_nutrition_static(q, df_static)
    if res_static:
        return res_static

    return None

# --------- Integration helper for predictions (top-k) ---------
def get_nutrition_from_predictions(predictions: List[tuple],
                                   use_usda: bool = False,
                                   usda_api_key: Optional[str] = None,
                                   static_csv_path: str = "nutrition_db.csv") -> Optional[Dict[str, Any]]:
    """
    predictions: list of tuples as returned by decode_predictions: [(imagenetID, label, prob), ...]
    returns first successful nutrition dict or None
    """
    df = None
    for (_, label, prob) in predictions:
        norm = normalize_label(label)
        # try mapped and raw forms
        for candidate in [norm, COMMON_FOOD_MAP.get(norm, norm)]:
            if candidate is None:
                continue
            info = lookup_nutrition(candidate, use_usda=use_usda, usda_api_key=usda_api_key, static_csv_path=static_csv_path, df_static=df)
            if info:
                info['prediction_label'] = label
                info['prediction_confidence'] = float(prob)
                return info
    return None