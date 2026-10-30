"""Streamlit interface for diagnosis, context-aware guidance, and follow-up."""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from context_engine import (
    AMBIGUITY_MARGIN,
    UNCERTAINTY_THRESHOLD,
    build_contextual_recommendation,
    estimate_risk,
    estimate_visual_severity,
    feedback_followup,
    save_feedback,
)
from treatment_recommendations import get_recommendation

IMG_SIZE = (224, 224)
MODEL_PATH = "model/tomato_disease_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, encoding="utf-8") as f:
        class_indices = json.load(f)
    return model, {int(v): k for k, v in class_indices.items()}


def preprocess_image(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB").resize(IMG_SIZE), dtype="float32")
    return np.expand_dims(preprocess_input(arr), axis=0)


def main():
    st.set_page_config(page_title="Tomato Leaf Decision Support", page_icon="🍅", layout="wide")
    st.title("🍅 Tomato Leaf Diagnosis and Treatment Decision Support")
    st.caption("MobileNetV2 classification with a transparent context and follow-up prototype. Not a substitute for local diagnosis.")

    with st.expander("Review demo: saved validation results", expanded=False):
        metrics_path = "model/eval_metrics.json"
        matrix_path = "model/confusion_matrix.png"
        if os.path.exists(metrics_path) and os.path.exists(matrix_path):
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            st.caption("These are the validation results produced earlier by evaluate.py and included with the trained model. They are not recalculated in this session because the validation image dataset is not included.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            m2.metric("Macro precision", f"{metrics['precision_macro']:.2%}")
            m3.metric("Macro recall", f"{metrics['recall_macro']:.2%}")
            m4.metric("Macro F1", f"{metrics['f1_macro']:.2%}")
            st.image(matrix_path, caption="Saved confusion matrix from the validation evaluation", use_container_width=True)
        else:
            st.warning("Saved evaluation results are missing. Run `python evaluate.py` with data/val available to generate them.")

        st.markdown("#### Run a live mini-evaluation")
        st.caption("Upload a few labeled leaf photos. The app will run the saved classifier now and generate a new confusion matrix from only these samples. Select each photo's known class correctly; this small demonstration is not a replacement for the full validation evaluation.")
        demo_files = st.file_uploader("Choose labeled tomato leaf photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="review_eval_files")
        if demo_files:
            demo_classes = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]
            with st.form("live_mini_evaluation"):
                actual_labels = []
                for i, sample in enumerate(demo_files):
                    actual_labels.append(st.selectbox(f"Known class for {sample.name}", demo_classes, format_func=lambda name: name.replace("_", " "), key=f"demo_class_{i}"))
                run_demo = st.form_submit_button("Run evaluation and generate matrix")
            if run_demo:
                try:
                    demo_model, demo_idx_to_class = load_model_and_classes()
                    y_true, y_pred = [], []
                    progress = st.progress(0, text="Evaluating uploaded samples...")
                    for i, (sample, actual_name) in enumerate(zip(demo_files, actual_labels)):
                        sample.seek(0)
                        sample_image = Image.open(sample).convert("RGB")
                        scores = demo_model.predict(preprocess_image(sample_image), verbose=0)[0]
                        y_true.append(next(index for index, name in demo_idx_to_class.items() if name == actual_name))
                        y_pred.append(int(np.argmax(scores)))
                        progress.progress((i + 1) / len(demo_files), text=f"Evaluated {i + 1} of {len(demo_files)} photos")
                    shown_ids = sorted(set(y_true) | set(y_pred))
                    shown_names = [demo_idx_to_class[index].replace("_", " ") for index in shown_ids]
                    matrix = confusion_matrix(y_true, y_pred, labels=shown_ids)
                    st.success(f"Generated a new confusion matrix from {len(demo_files)} uploaded photos.")
                    score_cols = st.columns(4)
                    score_cols[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                    score_cols[1].metric("Macro precision", f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.2%}")
                    score_cols[2].metric("Macro recall", f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.2%}")
                    score_cols[3].metric("Macro F1", f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.2%}")
                    fig, ax = plt.subplots(figsize=(max(5, len(shown_ids) * 1.1), max(4, len(shown_ids) * 0.85)))
                    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=shown_names, yticklabels=shown_names, ax=ax)
                    ax.set_xlabel("Predicted class")
                    ax.set_ylabel("Known class")
                    ax.set_title("Live mini-evaluation confusion matrix")
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption("This matrix summarizes only the photos uploaded for this demonstration. It should not be reported as the model's full validation performance.")
                except Exception as exc:
                    st.error(f"Could not run the live mini-evaluation: {exc}")

    try:
        model, idx_to_class = load_model_and_classes()
    except (OSError, IOError):
        st.error("Model files are missing. Prepare the dataset and run `python train.py` first.")
        st.stop()

    uploaded = st.file_uploader("Upload a clear, well-lit tomato leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("The system requests another image when confidence is low or the two leading classes are too close.")
        st.stop()
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)
    with st.spinner("Analyzing image..."):
        probs = model.predict(preprocess_image(image), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top_idx, second_idx = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    confidence = float(probs[top_idx])
    margin = confidence - float(probs[second_idx])
    disease = idx_to_class[top_idx]
    st.subheader("Diagnosis")
    st.write(f"**Top class:** {disease.replace('_', ' ')} · **confidence:** {confidence:.1%}")
    st.progress(min(confidence, 1.0))
    uncertain = confidence < UNCERTAINTY_THRESHOLD or margin < AMBIGUITY_MARGIN
    if uncertain:
        st.warning(f"Uncertain result (top-two gap {margin:.1%}). Please take another in-focus image in natural light, including the whole leaf and a close-up. No treatment recommendation is issued from this result.")
        with st.expander("See model probabilities"):
            for idx in order[:5]:
                st.write(f"{idx_to_class[int(idx)].replace('_', ' ')}: {float(probs[idx]):.1%}")
        st.caption("The confidence threshold is a configurable prototype threshold, not a calibrated guarantee of correctness.")
        st.stop()

    severity_estimate = estimate_visual_severity(image)
    st.subheader("Visible severity estimate")
    if severity_estimate["level"] == "Unavailable":
        st.warning("Could not isolate enough green leaf pixels for a visual severity estimate. Treat severity as unknown and seek another photo or expert review.")
        severity = "Unknown"
    else:
        st.write(f"**{severity_estimate['level']}** · estimated discolored fraction within a coarse green-pixel mask: {severity_estimate['discoloration_ratio']:.1%}")
        st.caption("This is a simple image-color proxy, not a validated lesion segmentation or severity model.")
        severity = st.selectbox("Severity used by the decision rules (edit if the estimate looks wrong)", ["Low", "Moderate", "High"], index=["Low", "Moderate", "High"].index(severity_estimate["level"]))

    st.subheader("Growing conditions")
    c1, c2, c3 = st.columns(3)
    with c1:
        has_temperature = st.checkbox("I have temperature data", key="has_temperature")
        temperature_value = st.number_input("Air temperature (°C)", -10.0, 60.0, value=25.0, disabled=not has_temperature)
        has_humidity = st.checkbox("I have humidity data", key="has_humidity")
        humidity_value = st.number_input("Relative humidity (%)", 0.0, 100.0, value=70.0, disabled=not has_humidity)
    with c2:
        has_soil_moisture = st.checkbox("I have soil moisture data", key="has_soil_moisture")
        soil_moisture_value = st.number_input("Soil moisture (%)", 0.0, 100.0, value=45.0, disabled=not has_soil_moisture)
        has_leaf_wetness = st.checkbox("I have leaf-wetness data", key="has_leaf_wetness")
        leaf_wetness_value = st.number_input("Leaf wetness (hours/day)", 0.0, 24.0, value=0.0, disabled=not has_leaf_wetness)
    with c3:
        has_rainfall = st.checkbox("I have rainfall data", key="has_rainfall")
        rainfall_value = st.number_input("Rainfall in last 24 h (mm)", 0.0, 500.0, value=0.0, disabled=not has_rainfall)
        crop_stage = st.selectbox("Crop stage", ["Seedling", "Vegetative", "Flowering", "Fruiting"])
    history = st.selectbox("Previous treatment response", ["No prior treatment", "Previously effective", "Previously ineffective"])
    weather = {
        "temperature": temperature_value if has_temperature else None,
        "humidity": humidity_value if has_humidity else None,
        "soil_moisture": soil_moisture_value if has_soil_moisture else None,
        "leaf_wetness": leaf_wetness_value if has_leaf_wetness else None,
        "rainfall": rainfall_value if has_rainfall else None,
    }
    risk = estimate_risk(disease, severity, weather)
    base = get_recommendation(disease)
    recommendation = build_contextual_recommendation(disease, base, severity, risk, crop_stage, history)

    st.subheader("Contextual assessment")
    st.metric("Progression risk heuristic", f"{risk['band']} · {risk['score']}/100")
    st.caption(f"Rule-based indicator, not a trained or field-validated progression forecast. Environmental coverage: {risk['coverage']}.")
    for reason in risk["reasons"]:
        st.write(f"- {reason}")
    st.subheader("Treatment options")
    st.write(f"**Priority:** {recommendation['priority']}")
    st.write(recommendation["explanation"])
    st.write(f"**Class guidance:** {base['description']}")
    if recommendation.get("base_treatment"):
        st.write(f"**General reference:** {recommendation['base_treatment']}")
    st.write("**Cultural option:** " + recommendation["options"][0].split(": ", 1)[1])
    st.write("**Biological option:** " + recommendation["options"][1].split(": ", 1)[1])
    st.write("**Chemical option:** " + recommendation["options"][2].split(": ", 1)[1])
    st.write(f"**Prevention:** {base['prevention']}")
    st.info("Use only products registered for tomato and the suspected problem in your location. Follow the label, protective-equipment directions, and pre-harvest interval; ask an agricultural extension officer to confirm diagnosis and product choice.")

    st.subheader("Post-treatment feedback")
    st.caption("After applying a treatment, return with a comparable photo. Feedback is recorded locally as a simple log; it does not train the classifier.")
    with st.form("feedback_form"):
        followup_severity = st.selectbox("Severity on follow-up photo", ["Low", "Moderate", "High"])
        outcome = st.selectbox("Farmer-reported response", ["Improved", "No change", "Worsened"])
        applied = st.text_input("Treatment applied (optional; avoid personal information)")
        submitted = st.form_submit_button("Save follow-up")
    if submitted:
        ordinal = {"Low": 1, "Moderate": 2, "High": 3}
        guidance = feedback_followup(ordinal.get(severity, 0), ordinal[followup_severity], outcome)
        try:
            save_feedback({"disease_class": disease, "initial_severity": severity, "followup_severity": followup_severity,
                           "reported_outcome": outcome, "treatment_applied": applied[:200], "guidance": guidance})
            st.success(guidance)
        except OSError:
            st.error("Could not save feedback to model/treatment_feedback.jsonl.")

    with st.expander("Model probabilities"):
        for idx in order[:5]:
            st.write(f"{idx_to_class[int(idx)].replace('_', ' ')}: {float(probs[idx]):.1%}")
    st.caption("Prototype limitations: PlantVillage benchmark images have controlled backgrounds; visual severity and risk rules require field validation. Feedback is not evidence that a treatment caused an outcome.")


if __name__ == "__main__":
    main()
