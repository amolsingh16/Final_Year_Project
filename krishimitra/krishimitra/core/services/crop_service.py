"""
Crop Recommendation and Fertilizer Requirement Advisory Engine.
Based on ICAR soil testing standards and agronomic crop guidelines for Indian agriculture.
"""

# Crop agronomic requirements: (N_min, N_max, P_min, P_max, K_min, K_max, pH_min, pH_max, ideal_soil)
CROP_DATABASE = [
    {
        "name": "Tomato",
        "n_range": (60, 150),
        "p_range": (30, 80),
        "k_range": (40, 120),
        "ph_range": (6.0, 7.5),
        "suitable_soils": ["Loamy", "Clay Loam", "Sandy Loam"],
        "season": "All Seasons / Rabi & Kharif",
        "rec_npk_per_acre": (40, 20, 20),  # N, P2O5, K2O in kg per acre
        "tips": "Requires well-drained soil rich in organic matter. Avoid waterlogging."
    },
    {
        "name": "Rice / Wheat",
        "n_range": (80, 160),
        "p_range": (40, 90),
        "k_range": (40, 100),
        "ph_range": (5.5, 7.5),
        "suitable_soils": ["Clay", "Clay Loam", "Loamy"],
        "season": "Kharif (Rice) / Rabi (Wheat)",
        "rec_npk_per_acre": (48, 24, 24),
        "tips": "Requires abundant water during early growth phase and balanced basal NPK application."
    },
    {
        "name": "Maize (Corn)",
        "n_range": (70, 140),
        "p_range": (35, 75),
        "k_range": (30, 80),
        "ph_range": (5.8, 7.2),
        "suitable_soils": ["Sandy Loam", "Loamy", "Silt Loam"],
        "season": "Kharif & Rabi",
        "rec_npk_per_acre": (48, 24, 20),
        "tips": "High nitrogen responder. Ensure soil aeration and proper drainage."
    },
    {
        "name": "Cotton",
        "n_range": (90, 170),
        "p_range": (40, 80),
        "k_range": (40, 90),
        "ph_range": (6.0, 8.0),
        "suitable_soils": ["Black Cotton Soil", "Deep Clay", "Loamy"],
        "season": "Kharif",
        "rec_npk_per_acre": (40, 20, 20),
        "tips": "Thrives in deep black soil with good moisture retention capacity."
    },
    {
        "name": "Potato",
        "n_range": (80, 150),
        "p_range": (50, 100),
        "k_range": (80, 150),
        "ph_range": (5.0, 6.5),
        "suitable_soils": ["Sandy Loam", "Loose Loam"],
        "season": "Rabi",
        "rec_npk_per_acre": (60, 30, 40),
        "tips": "Prefers slightly acidic, loose, friable soil for tuber development."
    },
    {
        "name": "Pulses (Chickpea/Pigeonpea)",
        "n_range": (20, 50),
        "p_range": (40, 80),
        "k_range": (20, 60),
        "ph_range": (6.0, 7.8),
        "suitable_soils": ["Sandy Loam", "Loamy"],
        "season": "Rabi / Kharif",
        "rec_npk_per_acre": (10, 20, 10),
        "tips": "Leguminous crop fixing atmospheric nitrogen. Requires phosphatic basal fertilizer."
    }
]


def recommend_crops(n: float, p: float, k: float, ph: float, soil_type: str = "") -> list:
    """
    Evaluates soil parameters N-P-K (kg/ha) and pH to match suitable crop choices.
    """
    scores = []
    
    for crop in CROP_DATABASE:
        score = 0
        reasons = []
        
        # N score
        if crop["n_range"][0] <= n <= crop["n_range"][1]:
            score += 25
            reasons.append("Optimal Nitrogen level")
        elif abs(n - sum(crop["n_range"])/2) <= 50:
            score += 15
            reasons.append("Acceptable Nitrogen level")
            
        # P score
        if crop["p_range"][0] <= p <= crop["p_range"][1]:
            score += 25
            reasons.append("Optimal Phosphorus level")
        elif abs(p - sum(crop["p_range"])/2) <= 40:
            score += 15
            reasons.append("Acceptable Phosphorus level")

        # K score
        if crop["k_range"][0] <= k <= crop["k_range"][1]:
            score += 25
            reasons.append("Optimal Potassium level")
        elif abs(k - sum(crop["k_range"])/2) <= 40:
            score += 15
            reasons.append("Acceptable Potassium level")

        # pH score
        if crop["ph_range"][0] <= ph <= crop["ph_range"][1]:
            score += 25
            reasons.append("Ideal soil pH range")
        elif abs(ph - 6.5) <= 1.5:
            score += 15
            reasons.append("Manageable soil pH")

        match_pct = min(100, max(40, score))
        scores.append({
            "name": crop["name"],
            "match_pct": match_pct,
            "season": crop["season"],
            "tips": crop["tips"],
            "reasons": reasons,
            "rec_npk": crop["rec_npk_per_acre"]
        })

    # Sort highest match percentage first
    scores.sort(key=lambda x: x["match_pct"], reverse=True)
    return scores


def calculate_fertilizers(crop_name: str, area: float, area_unit: str = "Acre") -> dict:
    """
    Calculates estimated Urea, DAP, and MOP commercial fertilizer bags (50kg) required.
    
    Commercial Formulations:
    - DAP (18% N, 46% P2O5)
    - Urea (46% N)
    - MOP / Muriate of Potash (60% K2O)
    """
    # Convert area to acres
    acres = area * 2.47105 if area_unit.lower() in ["hectare", "ha"] else area

    # Default NPK recommendation per acre (N, P2O5, K2O in kg)
    target_crop = next((c for c in CROP_DATABASE if c["name"].lower() in crop_name.lower()), CROP_DATABASE[0])
    rec_n, rec_p, rec_k = target_crop["rec_npk_per_acre"]

    total_n_req = rec_n * acres
    total_p_req = rec_p * acres
    total_k_req = rec_k * acres

    # Step 1: P requirement met via DAP (46% P2O5)
    dap_kg = total_p_req / 0.46
    dap_n_supplied = dap_kg * 0.18

    # Step 2: Remaining N requirement met via Urea (46% N)
    remaining_n_req = max(0.0, total_n_req - dap_n_supplied)
    urea_kg = remaining_n_req / 0.46

    # Step 3: K requirement met via MOP (60% K2O)
    mop_kg = total_k_req / 0.60

    return {
        "crop_name": target_crop["name"],
        "acres": round(acres, 2),
        "dap_kg": round(dap_kg, 1),
        "dap_bags": round(dap_kg / 50.0, 1),
        "urea_kg": round(urea_kg, 1),
        "urea_bags": round(urea_kg / 50.0, 1),
        "mop_kg": round(mop_kg, 1),
        "mop_bags": round(mop_kg / 50.0, 1),
        "disclaimer": "Advisory Note: These calculated quantities are general recommendations based on ICAR guidelines. Please consult your local Krishi Vigyan Kendra (KVK) or soil-testing laboratory for exact field recommendations."
    }
