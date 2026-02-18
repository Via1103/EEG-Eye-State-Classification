**EEG Eye State Classification using Deep Learning**

**Overview**

This project explores deep learning architectures for classifying eye state (Open vs Closed) using EEG time-series signals.

The objective was to build a clean temporal modeling pipeline and compare different recurrent architectures for sequence classification.

**Dataset**

- 14 EEG signal channels

- Binary classification (0 = Open, 1 = Closed)

- Time-ordered signal samples

**Methodology**
Sequence Generation

- Sliding window approach (window size = 20)

- Converts raw data into sequences of shape (20, 14)

**Data Splitting**

- 80–20 stratified train-test split

- Maintains class balance

**Feature Scaling**

- StandardScaler fitted only on training data

- Test data transformed using training statistics

- 3D sequence data reshaped for proper scaling (2D → scale → 3D)

**Models Implemented**

- Single LSTM

- Stacked LSTM

- CNN–LSTM Hybrid

All models were implemented using TensorFlow/Keras with early stopping.

**Results**

| Model        | Test Accuracy |
|--------------|--------------|
| Single LSTM  | 99.43% |
| Stacked LSTM | **99.80%** |
| CNN-LSTM     | 99.26% |

Confusion Matrix (Stacked LSTM)

Training Curve (Stacked LSTM)


**Reproducibility**

To load the trained model:

from tensorflow.keras.models import load_model
import joblib

model = load_model("best_stacked_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

**Tools Used**

- Python

- TensorFlow / Keras

- NumPy

- Scikit-learn

- Matplotlib

**Key Takeaways**

- Proper temporal modeling significantly improves EEG classification.

- Leakage-aware preprocessing is critical for reliable evaluation.

- Stacked LSTM achieved the best performance due to hierarchical temporal feature extraction.
