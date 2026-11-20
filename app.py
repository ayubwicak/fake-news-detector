import os
import re
import csv
import json
import joblib
import unicodedata
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FEEDBACK_PATH = os.path.join(DATA_DIR, 'feedback.csv')

app = Flask(__name__)

# Rate limiting: 30 requests/minute/IP by default, overridable via env var.
# A callable limit value allows tests to lower the limit without re-importing.
RATE_LIMIT = os.environ.get('RATE_LIMIT_PER_MINUTE', '30 per minute')
app.config['RATE_LIMIT'] = RATE_LIMIT
app.config.setdefault(
    'RATELIMIT_STORAGE_URI',
    os.environ.get('RATE_LIMIT_STORAGE_URI', 'memory://'),
)
limiter = Limiter(app=app, key_func=get_remote_address)


def current_rate_limit() -> str:
    """Return the active rate limit (config first, env/default as fallback)."""
    return app.config.get('RATE_LIMIT', RATE_LIMIT)


@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

# Load model, vectorizer, and feature weights
model_path = os.path.join(MODELS_DIR, 'model.pkl')
vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
weights_path = os.path.join(MODELS_DIR, 'feature_weights.json')
threshold_path = os.path.join(MODELS_DIR, 'threshold.json')
eval_path = os.path.join(LOGS_DIR, 'evaluation.json')

model = None
tfidf = None
feature_weights = {}
eval_metrics = None
DECISION_THRESHOLD = 0.5

try:
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            DECISION_THRESHOLD = float(json.load(f).get('threshold', 0.5))
        print(f"Decision threshold loaded: {DECISION_THRESHOLD}")
except Exception as e:
    print(f"Warning: Could not load threshold: {e}")

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

from id_preprocessing import detect_language, clean_text as id_clean_text


def clean_text(text: str) -> str:
    """Preprocess text: remove URLs, punctuation, and lowercase."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def parse_feedback_label(value):
    """Strictly parse a feedback label.

    Accept only 0/1 integers or the strings '0'/'1'.  Booleans and floats are
    rejected even though ``True == 1`` and ``1.0 == 1`` in Python, because
    silently accepting them makes the saved label ambiguous.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in {'0', '1'}:
        return int(value.strip())
    return None


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
@limiter.limit(current_rate_limit)
def get_metrics():
    if eval_metrics:
        return jsonify(eval_metrics)
    return jsonify({'status': 'no_metrics_found'}), 404


@app.route('/predict', methods=['POST'])
@limiter.limit(current_rate_limit)
def predict():
    if model is None or tfidf is None:
        return jsonify({'error': 'Model is not trained or loaded yet. Run train_model.py first.'}), 500

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Parameter "text" is required.'}), 400

    # Validate input type - must be a string
    raw_text = data.get('text')
    if not isinstance(raw_text, str):
        return jsonify({'error': 'Parameter "text" must be a string.'}), 400

    raw_text = raw_text.strip()
    # Normalize Unicode (NFKC): convert full-width chars to standard, remove compatibility
    raw_text = unicodedata.normalize('NFKC', raw_text)
    if len(raw_text) < 10:
        return jsonify({'error': 'Input text is too short. Please provide at least 10 characters.'}), 400

    # Detect language using lexicon-based approach
    lang_result = detect_language(raw_text)
    lang_code = lang_result['code']
    lang_name = lang_result['name']
    supported = lang_result['supported']

    # Clean text using Indonesian preprocessing (consistent with training)
    cleaned = id_clean_text(raw_text)
    if not cleaned:
        return jsonify({'error': 'Input contains only symbols or numbers. Please provide valid text.'}), 400

    # Vectorize
    vec = tfidf.transform([cleaned])

    # Probabilities (calibrated)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vec)[0]
        prob_real = float(proba[0]) * 100
        prob_fake = float(proba[1]) * 100
    else:
        prob_real, prob_fake = 0.0, 100.0

    # Decision uses the tuned threshold (optimized on validation split)
    pred = 1 if (prob_fake / 100.0) >= DECISION_THRESHOLD else 0
    label = 'Fake' if pred == 1 else 'Real'
    confidence = prob_fake if pred == 1 else prob_real

    # Explainability (XAI) - get top influential features.  When the input has
    # no vocabulary known to the model, surface an explicit note instead of
    # silently returning an empty list.
    explanation = explain_prediction(cleaned)
    explanation_note = None
    if not explanation:
        explanation_note = (
            'Tidak ada kata kunci yang dikenali pada teks ini. '
            'Coba gunakan kalimat berita yang lebih lengkap.'
        )

    # Language-aware handling.  The primary model artifact (models/model.pkl)
    # is trained primarily on English news; key-influence weights come from its
    # vocabulary.  Regional-language / unknown inputs are flagged so callers can
    # show a user-visible warning instead of trusting the score blindly.
    language_warning = None
    if not supported:
        language_warning = {
            'code': lang_code,
            'name': lang_name,
            'message': (
                f'Prediction for {lang_name} may be less accurate. '
                f'The primary model is trained mainly on English news, and key '
                f'indicators are drawn from that vocabulary.'
            ),
        }

    return jsonify({
        'prediction': label,
        'label_code': pred,
        'confidence': round(confidence, 2),
        'probability_real': round(prob_real, 2),
        'probability_fake': round(prob_fake, 2),
        'decision_threshold': DECISION_THRESHOLD,
        'snippet': raw_text[:140] + ('...' if len(raw_text) > 140 else ''),
        'character_count': len(raw_text),
        'word_count': len(raw_text.split()),
        'key_influences': explanation,
        'key_influences_note': explanation_note,
        'language_detection': {
            'code': lang_code,
            'name': lang_name,
            'supported': supported,
            'warning': language_warning['message'] if language_warning else None
        }
    })


@app.route('/feedback', methods=['POST'])
@limiter.limit(current_rate_limit)
def feedback():
    """Record human-verified labels for continuous retraining."""
    data = request.get_json(silent=True)
    if not data or 'text' not in data or 'label' not in data:
        return jsonify({'error': 'Parameters "text" and "label" are required.'}), 400

    label = parse_feedback_label(data['label'])
    if label is None:
        return jsonify({'error': 'Label must be 0 (Real) or 1 (Fake).'}), 400

    text = str(data['text']).strip()
    text = unicodedata.normalize('NFKC', text)

    if len(text) < 10:
        return jsonify({'error': 'Feedback text is too short (minimum 10 characters).'}), 400

    os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(FEEDBACK_PATH)

    try:
        with open(FEEDBACK_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['text', 'label'])
            writer.writerow([text, label])
    except Exception as e:
        return jsonify({'error': f'Could not save feedback: {e}'}), 500

    try:
        with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
            total = max(sum(1 for _ in f) - 1, 0)
    except Exception:
        total = 1

    return jsonify({
        'status': 'saved',
        'message': 'Feedback recorded. Run "python train_model.py" to retrain and improve the model.',
        'total_feedback_samples': total
    })


@app.route('/health', methods=['GET'])
def health():
    feedback_count = 0
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
                feedback_count = max(sum(1 for _ in f) - 1, 0)
        except Exception:
            feedback_count = 0

    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': tfidf is not None,
        'feature_weights_loaded': len(feature_weights) > 0,
        'decision_threshold': DECISION_THRESHOLD,
        'feedback_samples_pending_retrain': feedback_count
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
