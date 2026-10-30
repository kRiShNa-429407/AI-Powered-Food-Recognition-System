# AI-Powered Food Recognition System

An AI-powered web app that identifies food items from images and shows their nutritional information (calories, protein, fat, and carbohydrates). It helps users track their diet without manual calorie counting.

## Features

- Upload a food image or use a live camera feed
- Top-5 food predictions with confidence scores
- Nutrition details (calories, protein, fat, carbs) for the predicted food
- Simple, interactive Streamlit interface

## How It Works

1. **Input:** the user uploads an image or captures one with the camera.
2. **Preprocessing:** the image is resized and normalized.
3. **Prediction:** a pretrained ResNet-18 model fine-tuned on Food-101 predicts the top-5 food classes.
4. **Nutrition mapping:** the predicted food is mapped to its calories, protein, fat, and carbs.
5. **Output:** predictions and nutrition info are shown in the app.

## Tech Stack

- Python
- PyTorch (ResNet-18, transfer learning)
- OpenCV
- Streamlit

## Model Training

- **Dataset:** Food-101
- **Model:** pretrained ResNet-18
- **Loss / optimizer:** Cross-Entropy Loss with Adam/SGD
- **Split:** 80% training, 20% validation
- **Augmentation:** rotation, flip, crop
- **Trained for:** 5 epochs (`food101_after5epochs.pth`)

Training is done in the `AI_Powered.ipynb` notebook.

## Project Structure

```
.
├── app.py                      # Streamlit app
├── AI_Powered.ipynb            # Model training notebook
├── food_classes.json           # List of food class names
├── food101_after5epochs.pth    # Trained model weights
└── .gitignore
```

## Getting Started

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. Install dependencies
pip install torch torchvision opencv-python streamlit pillow

# 3. Run the app
streamlit run app.py
```

## Applications

- Healthcare and diet monitoring (e.g., for diabetic or obese patients)
- Fitness and lifestyle planning
- Education on AI and health awareness
- Research in computer vision and nutrition science

## Future Scope

- Add more food categories
- Android/iOS app
- Diet recommendations with healthier alternatives
- Voice assistance for nutrition details
- Smartwatch and fitness tracker integration
