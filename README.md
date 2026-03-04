# Sleep Apnea Detection: Health Sensing Pipeline

This repository contains the complete pipeline for analyzing, preprocessing, and modeling overnight polysomnography (PSG) data to detect breathing irregularities. Developed for the IITG SRIP 2026 Assessment.

## Repository Structure

\`\`\`text
Project Root/ 
|-- Data/                       # Raw participant data (Ignored in Git for privacy/size)
|-- Dataset/                    # Processed & windowed datasets
|   |-- breathing_dataset.pkl
|-- Visualizations/             # Generated 30-minute paginated PDFs
|-- models/                     # Model architectures
|   |-- __init__.py
|   |-- cnn_model.py
|-- scripts/                    # Execution scripts
|   |-- vis.py                  # EDA & Plotting
|   |-- create_dataset.py       # Preprocessing & Windowing
|   |-- train_model.py          # 1D CNN Training & LOOCV
|-- README.md
|-- requirements.txt
|-- report.pdf                  # Final analysis and results discussion
\`\`\`

## Installation

Ensure you have Python 3.10+ installed. Clone this repository and install the dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage and Execution

### 1. Data Visualization
Generates a paginated PDF plotting Nasal Airflow, Thoracic Movement, and SpO2 for the entire 8-hour session, overlaid with annotated respiratory events.
\`\`\`bash
python scripts/vis.py -name "Data/AP02"
\`\`\`

### 2. Signal Preprocessing & Dataset Creation
Applies a 0.17-0.4 Hz Butterworth bandpass filter, slices data into 30-second windows (50% overlap), upsamples SpO2, and exports a unified `.pkl` dataset.
\`\`\`bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
\`\`\`

### 3. Model Training & Evaluation
Trains a 1D Convolutional Neural Network on the processed dataset using Leave-One-Participant-Out Cross-Validation (LOOCV).
\`\`\`bash
python scripts/train_model.py -dataset "Dataset/breathing_dataset.pkl"
\`\`\`

## Results Overview
The baseline 1D CNN achieved an overall accuracy of **91.37%**. However, analysis of the confusion matrix indicates a severe class imbalance favoring normal breathing windows over apnea/hypopnea events. For a detailed discussion of the methodology and results, please refer to `report.pdf`.