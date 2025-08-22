# Health condition rules and checker

# Step A: Define rules
health_rules = {
    "diabetes": {
        "carbs_limit": 40,
        "advice": "High carbs may spike blood sugar. Prefer whole grains & fiber-rich foods."
    },
    "hypertension": {
        "fat_limit": 15,
        "advice": "High fat & sodium may increase BP. Prefer grilled/boiled foods."
    },
    "obesity": {
        "calories_limit": 300,
        "advice": "High calories can cause weight gain. Try smaller portions."
    },
    "kidney": {
        "protein_limit": 20,
        "advice": "Excess protein stresses kidneys. Prefer balanced meals."
    },
    "fitness": {
        "balanced": True,
        "advice": "Aim for a balanced diet: carbs 45–65%, protein 10–35%, fat 20–35%."
    }
}

# Step B: Function to apply rules
def check_health_conditions(food, nutrition, user_conditions):
    results = []
    
    for condition in user_conditions:
        rule = health_rules.get(condition)
        if not rule:
            continue
        
        # Apply condition-specific checks
        if condition == "diabetes" and nutrition["Carbs"] > rule["carbs_limit"]:
            results.append(f"⚠️ {food} has {nutrition['Carbs']}g carbs. {rule['advice']}")
        
        if condition == "hypertension" and nutrition["Fat"] > rule["fat_limit"]:
            results.append(f"⚠️ {food} has {nutrition['Fat']}g fat. {rule['advice']}")
        
        if condition == "obesity" and nutrition["Calories"] > rule["calories_limit"]:
            results.append(f"⚠️ {food} has {nutrition['Calories']} calories. {rule['advice']}")
        
        if condition == "kidney" and nutrition["Protein"] > rule["protein_limit"]:
            results.append(f"⚠️ {food} has {nutrition['Protein']}g protein. {rule['advice']}")
        
        if condition == "fitness":
            total = nutrition["Carbs"] + nutrition["Protein"] + nutrition["Fat"]
            if total > 0:
                carb_pct = (nutrition["Carbs"] / total) * 100
                prot_pct = (nutrition["Protein"] / total) * 100
                fat_pct = (nutrition["Fat"] / total) * 100
                results.append(
                    f"ℹ️ Macro balance: {carb_pct:.1f}% carbs, "
                    f"{prot_pct:.1f}% protein, {fat_pct:.1f}% fat. {rule['advice']}"
                )
    
    return results