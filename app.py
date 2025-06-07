from flask import Flask, request, render_template, jsonify, url_for, redirect
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
import io
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL_FILE = os.getenv('MODEL_FILE', 'plant_disease_model.pth')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', 'YOUR_OPENWEATHERMAP_API_KEY')
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

app = Flask(__name__)

# Define transforms for image preprocessing
def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    ])

# Load the pre-trained model
def load_model():
    try:
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace the final layer with our classifier
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 38)  # 38 classes in PlantVillage
        
        # Check if model file exists
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model weights file not found: {MODEL_FILE}")
            
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Initialize model and transforms
try:
    model = load_model()
    transforms = get_transforms()
except Exception as e:
    print(f"Failed to initialize: {str(e)}")
    model = None
    transforms = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    # Get parameters from URL
    disease = request.args.get('disease', 'No disease detected')
    risk = request.args.get('risk', 'Environmental risk not available')
    
    return render_template('result.html', disease=disease, risk=risk)

@app.route('/api/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Process image upload and return disease prediction."""
    try:
        if not model:
            return jsonify({
                'error': f'Model not loaded. Please make sure the model weights file ({MODEL_FILE}) exists.',
                'disease': 'Model not available',
                'risk': 'Model not available'
            })

        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Process image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        if transforms:
            img_tensor = transforms(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
            disease = CLASS_NAMES[predicted.item()]
        else:
            disease = "Model not available"
        
        # Get weather risk if location is provided
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        weather_risk = "Enable location for weather alerts."
        weather_data = None
        
        if lat and lon:
            try:
                weather_data = get_weather_risk(float(lat), float(lon))
                if weather_data['status'] == 'error':
                    weather_risk = weather_data['message']
                else:
                    weather_risk = weather_data['message']
            except ValueError:
                weather_risk = "Invalid location coordinates"
                logger.error("Invalid location coordinates provided")
        
        response = {
            'disease': disease,
            'risk': weather_risk
        }
        
        # Add weather data if available
        if weather_data and weather_data['status'] != 'error':
            response['weather_data'] = weather_data
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500

if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Log application startup
    logger.info("Starting LeafGuard application...")
    logger.info(f"Model status: {'Ready' if model else 'Not loaded'}")
    logger.info(f"Using model file: {MODEL_FILE}")
    logger.info(f"Environment: {os.getenv('FLASK_ENV', 'development')}")
    
    # Run the application
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    else:
        app.run(debug=True)
def result():
    # Get parameters from URL
    disease = request.args.get('disease', 'No disease detected')
    risk = request.args.get('risk', 'Environmental risk not available')
    
    return render_template('result.html', disease=disease, risk=risk)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Process image
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            disease = predict_image(img)
            
            # Get weather risk if location is provided
            lat = request.form.get('latitude')
            lon = request.form.get('longitude')
            weather_risk = "Enable location for weather alerts."
            if lat and lon:
                weather_risk = get_weather_risk(float(lat), float(lon))
            
            return jsonify({
                'disease': disease,
                'risk': weather_risk
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
