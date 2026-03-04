import os
import argparse
import pandas as pd
import numpy as np
import pickle
from scipy.signal import butter, filtfilt

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

def load_events(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    
    events = pd.read_csv(
        file_path,
        sep=';',
        skiprows=5,
        engine="python",
        names=["time_range", "duration", "event_type", "sleep_stage"],
        header=None
    )
    
    events = events.dropna(subset=["time_range"])
      
    time_split = events["time_range"].str.split("-", expand=True)
    start_raw = time_split[0].str.strip().str.replace(",", ".")
    end_raw_time = time_split[1].str.strip().str.replace(",", ".")
    
    events["start_time"] = pd.to_datetime(start_raw, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
        
    date_str = start_raw.str.split(" ", expand=True)[0]
    events["end_time"] = pd.to_datetime(date_str + " " + end_raw_time, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
 
    crossover_mask = events["end_time"] < events["start_time"]
    events.loc[crossover_mask, "end_time"] += pd.Timedelta(days=1)
    
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
        
    return events

def filter_signal(df, col_name, fs):
    if df.empty:
        return df
    nyq = 0.5 * fs
    low = 0.17 / nyq
    high = 0.4 / nyq
    b, a = butter(4, [low, high], btype='band')
    
    signal = df[col_name].fillna(0).values
    df[col_name] = filtfilt(b, a, signal)
    return df

def get_window_label(w_start, w_end, events_df):
    if events_df.empty or "start_time" not in events_df.columns:
        return "normal"
        
    for _, event in events_df.iterrows():
        e_start = event["start_time"]
        e_end = event["end_time"]
        
        overlap_start = max(w_start, e_start)
        overlap_end = min(w_end, e_end)
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        
        if overlap_duration > 15:
            e_type = event["event_type"]
            if "apnea" in e_type:
                return "apnea"
            elif "hypopnea" in e_type:
                return "hypopnea"
            else:
                return e_type
                
    return "normal"

def process_all_participants(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dataset = []

    for participant_name in os.listdir(in_dir):
        participant_dir = os.path.join(in_dir, participant_name)
        
        if not os.path.isdir(participant_dir):
            continue
            
        nasal_path = thoracic_path = spo2_path = events_path = ""
        
        for f in os.listdir(participant_dir):
            lower_f = f.lower()
            if "flow events" in lower_f:
                events_path = os.path.join(participant_dir, f)
            elif "flow" in lower_f and "events" not in lower_f:
                nasal_path = os.path.join(participant_dir, f)
            elif "thorac" in lower_f:
                thoracic_path = os.path.join(participant_dir, f)
            elif "spo2" in lower_f:
                spo2_path = os.path.join(participant_dir, f)
                
        nasal_df = load_signal(nasal_path, "Nasal Airflow")
        thoracic_df = load_signal(thoracic_path, "Thoracic Movement")
        spo2_df = load_signal(spo2_path, "SpO2")
        events_df = load_events(events_path)

        if nasal_df.empty or thoracic_df.empty or spo2_df.empty:
            continue

        nasal_df = filter_signal(nasal_df, "Nasal Airflow", 32)
        thoracic_df = filter_signal(thoracic_df, "Thoracic Movement", 32)

        start_time = max(nasal_df.index.min(), thoracic_df.index.min(), spo2_df.index.min())
        end_time = min(nasal_df.index.max(), thoracic_df.index.max(), spo2_df.index.max())

        current_time = start_time
        window_duration = pd.Timedelta(seconds=30)
        step_duration = pd.Timedelta(seconds=15)

        while current_time + window_duration <= end_time:
            w_end = current_time + window_duration
            
            n_window = nasal_df.loc[current_time:w_end - pd.Timedelta(milliseconds=1)]["Nasal Airflow"].values
            t_window = thoracic_df.loc[current_time:w_end - pd.Timedelta(milliseconds=1)]["Thoracic Movement"].values
            s_window = spo2_df.loc[current_time:w_end - pd.Timedelta(milliseconds=1)]["SpO2"].values
            
            if len(n_window) >= 960 and len(t_window) >= 960 and len(s_window) >= 120:
                label = get_window_label(current_time, w_end, events_df)
                
                dataset.append({
                    "participant": participant_name,
                    "nasal": n_window[:960],
                    "thoracic": t_window[:960],
                    "spo2": s_window[:120],
                    "label": label
                })
                
            current_time += step_duration
            
        print(f"Processed {participant_name}")

    output_path = os.path.join(out_dir, "breathing_dataset.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Dataset successfully saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()

    process_all_participants(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()