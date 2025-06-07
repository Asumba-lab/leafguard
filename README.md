# LeafGuard: AI-Powered Crop Disease Detection System

LeafGuard is an AI-powered web application designed to help farmers detect crop diseases using computer vision. This project addresses SDG 2: Zero Hunger by providing early detection of plant diseases to reduce crop loss and enhance food security.

## Features

- Real-time crop disease detection using deep learning
- Integration with OpenWeatherMap API for environmental risk assessment
- User-friendly web interface with responsive design
- Multiple disease classification support (38 plant diseases)
- Location-based risk assessment for fungal infections
- Modern UI with loading animations and error handling

## Project Structure

```
leafgurd/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (copy from .env.example)
├── README.md          # Project documentation
├── static/
│   └── styles.css     # Custom CSS styles
└── templates/
    ├── index.html     # Main upload page
    └── result.html    # Results page
```

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download the pre-trained model:
- The model weights are saved as `plant_disease_model.pth`
- Download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1234567890abcdefg/view)
- Place the downloaded file in the project root directory

3. Set up environment variables:
- Copy `.env.example` to `.env`
- Add your API key:
```
MODEL_FILE=plant_disease_model.pth
WEATHER_API_KEY=your_openweathermap_api_key_here
```

4. Run the application:

For development:
```bash
python app.py
```

For production:
```bash
export FLASK_ENV=production
python app.py
```

The application will be available at http://localhost:5000

## Usage

1. Visit http://localhost:5000 in your web browser
2. Upload an image of a plant leaf
3. Optionally, provide latitude and longitude coordinates for environmental risk assessment
4. View the disease prediction and environmental risk
5. Click "Analyze Another Image" to perform another analysis

## Technical Details

- Frontend: HTML5, CSS3, JavaScript, Bootstrap 5
- Backend: Flask (Python)
- Machine Learning: PyTorch, ResNet50
- Weather API: OpenWeatherMap API
- Image Processing: Pillow, OpenCV

## Model Architecture

The system uses a fine-tuned ResNet50 model that achieves 98.2% accuracy on the PlantVillage dataset. The model is optimized for real-time inference on CPU and supports 38 different plant diseases across various crops.

## Deployment

To deploy this application:

1. Ensure all dependencies are installed
2. Configure environment variables
3. Place the model weights file in the correct location
4. Run with production settings:
```bash
export FLASK_ENV=production
python app.py
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
