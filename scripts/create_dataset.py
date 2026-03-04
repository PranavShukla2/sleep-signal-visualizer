import os
import argparse
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, medfilt

def load_signal(file_path, col_name): 
    if not os.path.exists(file_path):
        return pd.DataFrame()

    df = pd.read_csv(
        file_path,
        sep=';',
        skiprows=7,
        engine="python",
        names=["timestamp", col_name],
        header=None
    )

    df["timestamp"] = df["timestamp"].astype(str).str.replace(",", ".")
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], 
        format="%d.%m.%Y %H:%M:%S.%f", 
        errors="coerce"
    )
    
    if df[col_name].dtype == object:
        df[col_name] = df[col_name].str.replace(",", ".")
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    df = df.dropna()
    df.set_index("timestamp", inplace=True)

    return df

def preprocess_respiratory(df, col_name, fs=32):
    if df.empty:
        return df
    nyq = 0.5 * fs
    low = 0.1 / nyq
    high = 3.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    
    signal = df[col_name].fillna(0).values
    filtered_signal = filtfilt(b, a, signal)
    
    mean_val = np.mean(filtered_signal)
    std_val = np.std(filtered_signal)
    
    if std_val != 0:
        df[col_name] = (filtered_signal - mean_val) / std_val
    else:
        df[col_name] = filtered_signal - mean_val
        
    return df

def preprocess_spo2(df, col_name, fs=4):
    if df.empty:
        return df
    df.loc[(df[col_name] < 50) | (df[col_name] > 100), col_name] = np.nan
    df[col_name] = df[col_name].interpolate(method='linear').bfill().ffill()
    
    window_size = int(5 * fs)
    if window_size % 2 == 0:
        window_size += 1
        
    df[col_name] = medfilt(df[col_name].values, kernel_size=window_size)
    return df

def process_participant(participant_dir, output_dir):
    participant_name = os.path.basename(os.path.normpath(participant_dir))
    
    nasal_path = thoracic_path = spo2_path = ""
    
    if os.path.exists(participant_dir):
        for f in os.listdir(participant_dir):
            lower_f = f.lower()
            if "flow" in lower_f and "events" not in lower_f:
                nasal_path = os.path.join(participant_dir, f)
            elif "thorac" in lower_f:
                thoracic_path = os.path.join(participant_dir, f)
            elif "spo2" in lower_f:
                spo2_path = os.path.join(participant_dir, f)
    
    nasal_df = load_signal(nasal_path, "Nasal Airflow")
    thoracic_df = load_signal(thoracic_path, "Thoracic Movement")
    spo2_df = load_signal(spo2_path, "SpO2")

    nasal_df = preprocess_respiratory(nasal_df, "Nasal Airflow", fs=32)
    thoracic_df = preprocess_respiratory(thoracic_df, "Thoracic Movement", fs=32)
    spo2_df = preprocess_spo2(spo2_df, "SpO2", fs=4)

    participant_out_dir = os.path.join(output_dir, participant_name)
    os.makedirs(participant_out_dir, exist_ok=True)

    if not nasal_df.empty:
        nasal_df.to_csv(os.path.join(participant_out_dir, "clean_nasal_airflow.csv"))
    if not thoracic_df.empty:
        thoracic_df.to_csv(os.path.join(participant_out_dir, "clean_thoracic.csv"))
    if not spo2_df.empty:
        spo2_df.to_csv(os.path.join(participant_out_dir, "clean_spo2.csv"))

    print(f"Successfully processed and saved dataset for {participant_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", default="Dataset") 
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        return

    process_participant(args.dir, args.out)

if __name__ == "__main__":
    main()