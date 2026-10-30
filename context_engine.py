"""Explainable, rule-based context, risk, treatment, and feedback logic.

This is a decision-support prototype, not a validated agronomic model. Risk
scores and image-based severity proxies are transparent heuristics and should
not be presented as clinical/field-validated predictions.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

FEEDBACK_PATH = Path("model/treatment_feedback.jsonl")
UNCERTAINTY_THRESHOLD = 0.70
AMBIGUITY_MARGIN = 0.12


def estimate_visual_severity(image):
    """Estimate discoloration among green-ish pixels as a rough visual proxy."""
    rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    leaf_mask = (green > 0.10) & (green >= red * 0.75) & (green >= blue * 0.75)
    # A coarse proxy: less-green yellow/brown/pale pixels within tomato-colored pixels.
    healthy_green = (green > red * 1.12) & (green > blue * 1.03)
    discolored = leaf_mask & ~healthy_green
    leaf_pixels = int(leaf_mask.sum())
    ratio = float(discolored.sum() / leaf_pixels) if leaf_pixels else 0.0
    if leaf_pixels < 250:
        level = "Unavailable"
    elif ratio < 0.12:
        level = "Low"
    elif ratio < 0.30:
        level = "Moderate"
    else:
        level = "High"
    return {"level": level, "discoloration_ratio": ratio, "leaf_pixel_count": leaf_pixels}


def estimate_risk(disease, severity, weather):
    """Return a 0-100 rule score from entered environment; not a calibrated forecast."""
    score, reasons = 15, []
    if disease == "Tomato_healthy":
        score = 0
        return {"score": score, "band": "Low", "reasons": ["No disease class was predicted."], "coverage": "Not applicable"}
    if severity == "High":
        score += 25
        reasons.append("High visible-symptom proxy raises concern.")
    elif severity == "Moderate":
        score += 12
        reasons.append("Moderate visible-symptom proxy raises concern.")
    elif severity == "Unknown":
        reasons.append("Severity is unavailable; no severity points were added.")
    else:
        reasons.append("Low visible-symptom proxy.")

    moist_diseases = ("blight", "spot", "mold", "septoria", "target")
    disease_lower = disease.lower()
    if any(term in disease_lower for term in moist_diseases):
        for key, threshold, points, label in (
            ("humidity", 80, 15, "High humidity"),
            ("leaf_wetness", 8, 20, "Long leaf-wetness period"),
            ("rainfall", 10, 10, "Recent rainfall"),
        ):
            value = weather.get(key)
            if value is not None and value >= threshold:
                score += points
                reasons.append(f"{label} increases favorable conditions for this disease group.")
        temp = weather.get("temperature")
        if temp is not None and 15 <= temp <= 25:
            score += 8
            reasons.append("Entered temperature is within a broad favorable range for several tomato pathogens.")
    if "mite" in disease_lower and weather.get("humidity") is not None and weather["humidity"] < 45:
        score += 12
        reasons.append("Low humidity can favor spider-mite pressure.")
    if weather.get("soil_moisture") is not None and weather["soil_moisture"] < 20:
        reasons.append("Low soil moisture: address plant water stress with careful soil-level irrigation.")
    score = min(score, 100)
    band = "High" if score >= 65 else "Moderate" if score >= 35 else "Low"
    observed = sum(weather.get(k) is not None for k in ("temperature", "humidity", "soil_moisture", "leaf_wetness", "rainfall"))
    return {"score": score, "band": band, "reasons": reasons,
            "coverage": f"{observed}/5 environmental inputs supplied"}


def build_contextual_recommendation(class_name, base, severity, risk, crop_stage, history):
    if class_name == "Tomato_healthy":
        return {"priority": "Monitor", "options": ["Continue routine scouting and balanced crop care."],
                "explanation": "The classifier selected the healthy class; avoid treatment unless symptoms appear."}
    options = [
        "Cultural: remove heavily affected leaves when appropriate, avoid handling wet plants, improve airflow, and sanitize tools.",
        "Biological: consider a locally registered biological control suitable for the suspected disease and crop stage; follow its label.",
        "Chemical: use only a locally registered product for tomato and this disease, following its label, pre-harvest interval, and protective-equipment directions. Confirm with an extension officer.",
    ]
    if risk["band"] == "High":
        priority = "Prompt scouting and confirmation"
    elif severity == "Low":
        priority = "Preventive cultural measures and close monitoring"
    else:
        priority = "Integrated management with local confirmation"
    explanation = f"Priority reflects {severity.lower()} visual severity, {risk['band'].lower()} rule-based risk, and {crop_stage.lower()} crop stage."
    if history == "Previously ineffective":
        explanation += " A previously ineffective treatment was reported, so reassess the diagnosis and avoid repeating it without expert advice."
        priority = "Reassess diagnosis and treatment history"
    return {"priority": priority, "options": options, "explanation": explanation,
            "base_treatment": base.get("treatment", ""), "prevention": base.get("prevention", "")}


def feedback_followup(previous_severity, current_severity, outcome):
    if outcome == "Improved" or current_severity < previous_severity:
        return "Reported improvement: continue monitoring and follow the product label or cultural plan; avoid unnecessary repeat applications."
    if outcome == "No change" or current_severity == previous_severity:
        return "No clear change: capture a comparable image and review conditions/treatment with a local extension officer before changing products."
    return "Worsened: request expert diagnosis promptly; check for disease spread, look-alike symptoms, and treatment timing."


def save_feedback(record, path=FEEDBACK_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(record)
    entry["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
