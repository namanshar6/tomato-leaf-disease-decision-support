"""
treatment_recommendations.py
-----------------------------
Implements the "Treatment Recommendation" module from the architecture
diagram: given a predicted disease class, return general treatment and
preventive guidance.

Class names here follow the PlantVillage tomato naming convention
(after cleanup in data_prep.py, underscores instead of triple-underscores).
If your class folder names differ slightly, adjust the keys below to match
the output of `sorted(os.listdir('data/train'))`.
"""

RECOMMENDATIONS = {
    "Tomato_Bacterial_spot": {
        "description": "Bacterial infection causing small, dark, water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper-based bactericides; remove and destroy heavily infected leaves.",
        "prevention": "Use disease-free seeds/transplants, avoid overhead irrigation, rotate crops.",
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease (Alternaria solani) causing concentric 'target-spot' brown lesions on older leaves.",
        "treatment": "Apply chlorothalonil or mancozeb-based fungicides at first sign of symptoms.",
        "prevention": "Stake plants for airflow, mulch to prevent soil splash, rotate crops yearly.",
    },
    "Tomato_Late_blight": {
        "description": "Aggressive fungal-like disease (Phytophthora infestans) causing large, dark, water-soaked lesions that spread rapidly.",
        "treatment": "Apply fungicides containing chlorothalonil or copper immediately; remove infected plants.",
        "prevention": "Avoid overhead watering, ensure good drainage, plant resistant varieties.",
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease (Passalora fulva) causing pale green/yellow spots on top of leaves with olive-green mold underneath.",
        "treatment": "Improve ventilation, apply fungicides such as chlorothalonil.",
        "prevention": "Reduce humidity in greenhouse settings, avoid leaf wetness, space plants adequately.",
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease causing small circular spots with dark borders and grey centers, usually on lower leaves.",
        "treatment": "Apply fungicides (chlorothalonil, mancozeb); remove infected lower leaves.",
        "prevention": "Avoid overhead irrigation, mulch soil, rotate crops, remove plant debris after harvest.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Pest infestation causing stippled, yellowing leaves and fine webbing; not a fungal/bacterial disease.",
        "treatment": "Apply miticides or insecticidal soap; introduce natural predators like ladybugs.",
        "prevention": "Keep plants well-watered (mites thrive in dry conditions), monitor regularly.",
    },
    "Tomato_Target_Spot": {
        "description": "Fungal disease (Corynespora cassiicola) causing brown lesions with concentric rings, similar to early blight.",
        "treatment": "Apply fungicides such as azoxystrobin or chlorothalonil.",
        "prevention": "Improve air circulation, avoid leaf wetness, rotate crops.",
    },
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies, causing upward curling and yellowing of leaves, stunted growth.",
        "treatment": "No cure once infected; remove and destroy infected plants to limit spread.",
        "prevention": "Control whitefly populations, use reflective mulches, plant resistant varieties.",
    },
    "Tomato_Tomato_mosaic_virus": {
        "description": "Viral disease causing mottled light/dark green patterns, leaf curling, and stunted growth.",
        "treatment": "No chemical cure; remove and destroy infected plants, disinfect tools.",
        "prevention": "Use virus-free seeds, wash hands/tools between plants, control aphids.",
    },
    "Tomato_healthy": {
        "description": "No disease detected — the leaf appears healthy.",
        "treatment": "No treatment necessary.",
        "prevention": "Continue regular watering, balanced fertilization, and routine monitoring.",
    },
}


def get_recommendation(class_name: str) -> dict:
    """Look up treatment guidance for a predicted class name.

    Falls back to a generic message if the exact key isn't found
    (handles minor naming differences between dataset versions).
    """
    if class_name in RECOMMENDATIONS:
        return RECOMMENDATIONS[class_name]

    # Fuzzy fallback: match on lowercase substring
    key_lower = class_name.lower()
    for k, v in RECOMMENDATIONS.items():
        if k.lower() in key_lower or key_lower in k.lower():
            return v

    return {
        "description": "No detailed information available for this class.",
        "treatment": "Consult a local agricultural extension officer for confirmation and guidance.",
        "prevention": "Practice crop rotation, proper spacing, and regular field monitoring.",
    }
