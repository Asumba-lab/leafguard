import os
import io
import logging

from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from kaggle.api.kaggle_api_extended import KaggleApi

# ————— Setup —————
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Paths & credentials from environment
MODEL_FILE      = os.getenv('MODEL_FILE', 'plant_disease_model.pth')
MODEL_PATH      = os.path.join(BASE_DIR, MODEL_FILE)
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY      = os.getenv('KAGGLE_KEY')
DATASET_ID      = 'plantdoc/plantdoc-dataset'

# Image processing settings
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# Your disease classes
CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust",
    "Healthy"
]

# ————— Flask app —————
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

# DummyModel matches your CLASS_NAMES length
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input feature size of 10 (placeholder), output = number of classes
        self.fc = nn.Linear(10, len(CLASS_NAMES))

    def forward(self, x):
        return self.fc(x)

def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None
    try:
        model = DummyModel()
        state = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        logger.info("Dummy model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

model = load_model()
transforms_ = get_transforms()


# ————— Routes —————
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/download-dataset')
def download_dataset_route():
    """Downloads the Kaggle dataset into ./datasets folder."""
    try:
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY']      = KAGGLE_KEY

        download_path = os.path.join(BASE_DIR, 'datasets')
        os.makedirs(download_path, exist_ok=True)

        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading Kaggle dataset: {DATASET_ID}")
        api.dataset_download_files(DATASET_ID, path=download_path, unzip=True)
        logger.info("Download complete.")
        return jsonify({'message': f'Dataset downloaded at {download_path}'})
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return jsonify({'error': 'Failed to download dataset'}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Receives an image and optional lat/lon, returns disease + risk JSON."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Invalid file type'}), 400

    # Load and ignore the real image for dummy model
    try:
        Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    if model is None:
        return jsonify({'error': 'Model not available'}), 500

    # Dummy prediction: random tensor of correct shape
    inp = torch.randn(1, 10)
    with torch.no_grad():
        out = model(inp)
        _, idx = torch.max(out, 1)
    disease = CLASS_NAMES[idx.item()]

    # Simple placeholder risk
    lat = request.form.get('latitude')
    lon = request.form.get('longitude')
    risk = "No weather risk (dummy)"

    return jsonify({'disease': disease, 'risk': risk})


@app.route('/result')
def result():
    """Renders results page with disease + risk passed via query params."""
    disease = request.args.get('disease', 'No disease detected')
    risk    = request.args.get('risk', 'Environmental risk not available')
    return render_template('result.html', disease=disease, risk=risk)


# ————— Run —————
if __name__ == '__main__':
    env   = os.getenv('FLASK_ENV', 'development')
    debug = (env != 'production')
    logger.info(f"Starting LeafGuard in {env} mode; debug={debug}")
    app.run(debug=debug, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
