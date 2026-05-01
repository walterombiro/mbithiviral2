# VL Prediction App

This Streamlit web application predicts Viral Load (VL) suppression status using a pre-trained machine learning model.

## Features

- Web interface for manual input of patient features
- Prediction of Suppressed or Unsuppressed VL
- Display of probabilities for both classes
- Shows the input features used for prediction

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the model file `model.pkl` and `Clean_Data.csv` in the same directory.

## Running the App

Run the application:
```
streamlit run app.py
```

Or:
```
python -m streamlit run app.py
```

Open your browser and go to the provided local URL (usually `http://localhost:8501`)

## Usage

Fill in the sidebar with patient information and click "Predict" to get the results.

## Model Details

The model is a stacked ensemble classifier that predicts VL suppression based on various patient attributes.