# Sleep Apnea Detection: Health Sensing Pipeline

**Author:** Pranav Shukla (Vellore Institute of Technology, Bhopal)

Hi! This repository contains my submission for the IITG SRIP Assessment (AI for Health). It is a complete data pipeline for analyzing, preprocessing, and modeling overnight polysomnography (PSG) data to detect breathing irregularities like sleep apnea.

## Repository Structure

```text
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
Installation
To run this pipeline on your local machine, you will need Python 3.10 or higher. Clone the repository and install the required dependencies:

Bash

pip install -r requirements.txt
How to Run the Code
1. Data Visualization
This script generates a paginated PDF plotting Nasal Airflow, Thoracic Movement, and SpO2 for the entire 8-hour session, complete with overlaid respiratory event annotations.

Bash

python scripts/vis.py -name "Data/AP02"
2. Signal Preprocessing & Dataset Creation
This applies a 0.17-0.4 Hz Butterworth bandpass filter to isolate breathing frequencies, slices the continuous data into 30-second windows (with a 50% overlap), upsamples the SpO2 data, and exports a unified .pkl dataset for training.

Bash

python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
3. Model Training & Evaluation
This trains a baseline 1D Convolutional Neural Network on the processed dataset using Leave-One-Participant-Out Cross-Validation (LOOCV).

Bash

python scripts/train_model.py -dataset "Dataset/breathing_dataset.pkl"
Results Overview
The baseline 1D CNN achieved an overall accuracy of 91.37%. However, looking closely at the confusion matrix reveals a severe class imbalance favoring normal breathing windows (creating an "Accuracy Paradox").

For a much deeper dive into the methodology, the class imbalance issue, and the results, please check out report.pdf.
