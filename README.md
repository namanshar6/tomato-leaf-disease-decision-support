# Tomato Leaf Disease Decision Support

A Streamlit prototype that classifies tomato leaf images with a trained MobileNetV2 model and presents uncertainty-aware, context-dependent decision support.

## Run on Windows

1. Install Python 3.10–3.14 and ensure it is available on PATH. Python 3.14 requires the TensorFlow 2.22 release-candidate-or-newer dependency specified in `requirements.txt`.
2. Double-click `Launch Tomato Leaf App.bat` (or the `Tomato Leaf App` shortcut when present locally). On first launch, the script creates `.venv` and installs dependencies. Keep the launcher window open while using the browser app.
3. Or run manually:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The model must be present at `model/tomato_disease_model.h5`, with class names in `model/class_indices.json`.

## What the app does

- Classifies one uploaded tomato-leaf JPG or PNG into one of ten trained classes.
- Requests another image when top confidence is below 70% or the margin between the top two classes is below 12 percentage points.
- Shows an adjustable green-pixel/discoloration severity proxy.
- Uses optional temperature, humidity, soil moisture, leaf wetness, rainfall, crop stage, and prior response to produce a transparent rule-based risk score and contextual treatment options.
- Records user-entered follow-up outcomes locally in `model/treatment_feedback.jsonl`. Feedback does not retrain the classifier.
- Provides a saved full-validation report view and a small live mini-evaluation for manually labeled photos.

## Model and evaluation

`train.py` trains a MobileNetV2 transfer-learning classifier; `evaluate.py` calculates classification metrics and saves the confusion matrix and training metrics; `data_prep.py` filters tomato classes from a source dataset and prepares train/validation folders.

The package includes trained weights, class mapping, saved evaluation metrics, confusion matrix, and training curves. It does **not** include raw PlantVillage images or the validation split. The metrics in `model/eval_metrics.json` (accuracy 92.03%, macro precision 91.03%, macro recall 90.18%, macro F1 90.04%) are previously saved validation results, not a result regenerated from the images included here. To reproduce evaluation, create `data/train/<class>/` and `data/val/<class>/` using the original dataset and run the scripts.

## Prototype limitations

The disease classifier is trained. Severity and environmental risk are heuristic prototypes, not trained or validated predictors. The post-treatment feedback is a local log and does not establish treatment effectiveness. Treat recommendations as general decision support and check local product labels and agricultural extension guidance. Field performance has not been established.

See `output/pdf/Tomato_Leaf_Project_Implementation.pdf` for the implementation report, configuration, code map, screenshots, saved evaluation artifacts, and live demonstration steps.
