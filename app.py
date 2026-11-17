import os
import re
import json
import joblib
from flask import Flask, request, jsonify, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

app = Flask(__name__)

# Load model, vectorizer, and feature weights
model_path = os.path.join(MODELS_DIR, 'model.pkl')
vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
weights_path = os.path.join(MODELS_DIR, 'feature_weights.json')
eval_path = os.path.join(LOGS_DIR, 'evaluation.json')

model = None
tfidf = None
feature_weights = {}
eval_metrics = None

try:
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        tfidf = joblib.load(vec_path)
        print("Model and vectorizer loaded successfully.")
    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            feature_weights = json.load(f)
        print(f"Loaded {len(feature_weights)} feature weights for XAI.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

try:
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_metrics = json.load(f)
except Exception as e:
    print(f"Warning: Could not load evaluation metrics: {e}")


def clean_text(text: str) -> str:
    """Preprocess text: remove URLs, punctuation, and lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def explain_prediction(cleaned_text: str, top_n: int = 5):
    """Identify top influential keywords in input using model weights (XAI)."""
    words = cleaned_text.split()
    matched_features = []

    # Check unigrams and bigrams
    n_words = len(words)
    candidates = set(words)
    for i in range(n_words - 1):
        candidates.add(f"{words[i]} {words[i+1]}")

    for candidate in candidates:
        if candidate in feature_weights:
            weight = feature_weights[candidate]
            matched_features.append({
                'token': candidate,
                'weight': round(float(weight), 4),
                'indicator': 'fake' if weight > 0 else 'real',
                'magnitude': abs(weight)
            })

    # Sort by magnitude of influence
    matched_features.sort(key=lambda x: x['magnitude'], reverse=True)
    return matched_features[:top_n]


@app.route('/')
def home():
    return render_template('index.html', metrics=eval_metrics)


@app.route('/metrics', methods=['GET'])
def get_metrics():
    if eval_metrics:
        return jsonify(eval_metrics)
    return jsonify({'status': 'no_metrics_found'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tfidf is None:
        return jsonify({'error': 'Model is not trained or loaded yet. Run train_model.py first.'}), 500

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Parameter "text" is required.'}), 400

    raw_text = data.get('text', '').strip()
    if len(raw_text) < 10:
        return jsonify({'error': 'Input text is too short. Please provide at least 10 characters.'}), 400

    cleaned = clean_text(raw_text)
    if not cleaned:
        return jsonify({'error': 'Input contains only symbols or numbers. Please provide valid text.'}), 400

    # Vectorize
    vec = tfidf.transform([cleaned])

    # Predict
    pred = int(model.predict(vec)[0])
    label = 'Fake' if pred == 1 else 'Real'

    # Probabilities
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        prob_real = float(proba[0]) * 100
        prob_fake = float(proba[1]) * 100
        confidence = prob_fake if pred == 1 else prob_real
    else:
        confidence = 100.0
        prob_real = 0.0 if pred == 1 else 100.0
        prob_fake = 100.0 if pred == 1 else 0.0

    # Explainability (XAI)
    explanation = explain_prediction(cleaned)

    return jsonify({
        'prediction': label,
        'label_code': pred,
        'confidence': round(confidence, 2),
        'probability_real': round(prob_real, 2),
        'probability_fake': round(prob_fake, 2),
        'snippet': raw_text[:140] + ('...' if len(raw_text) > 140 else ''),
        'character_count': len(raw_text),
        'word_count': len(raw_text.split()),
        'key_influences': explanation
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': tfidf is not None,
        'feature_weights_loaded': len(feature_weights) > 0
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
