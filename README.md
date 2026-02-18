EEG Eye State Classification using Deep Learning
Overview

This project implements temporal deep learning architectures for EEG-based eye state classification using LSTM and CNN-LSTM models.

Methodology

Sliding window sequence generation (window size = 20)

Stratified 80–20 train-test split

StandardScaler fitted only on training data (leakage prevention)

Reshaping for proper scaling of 3D sequence data

Models Implemented

Single LSTM

Stacked LSTM

CNN-LSTM

Results
Model	Test Accuracy
Single LSTM	99.43%
Stacked LSTM	99.80%
CNN-LSTM	99.26%
Key Learning

This project highlights the importance of proper temporal modeling and leakage-free preprocessing in EEG-based classification tasks.