import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
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
    
    events["event_type"] = events["event_type"].astype(str).str.strip()
        
    return events

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

def create_visualization(participant_dir):
    participant_name = os.path.basename(os.path.normpath(participant_dir))
    
    nasal_path = thoracic_path = spo2_path = events_path = ""
    
    if os.path.exists(participant_dir):
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

    nasal_df = preprocess_respiratory(nasal_df, "Nasal Airflow", fs=32)
    thoracic_df = preprocess_respiratory(thoracic_df, "Thoracic Movement", fs=32)
    spo2_df = preprocess_spo2(spo2_df, "SpO2", fs=4)

    os.makedirs("Visualizations", exist_ok=True)
    output_path = os.path.join("Visualizations", f"{participant_name}_visualization.pdf")

    all_times = []
    for df in [nasal_df, thoracic_df, spo2_df]:
        if not df.empty:
            all_times.append(df.index.min())
            all_times.append(df.index.max())
            
    if not all_times:
        return

    recording_start = min(all_times)
    recording_end = max(all_times)
    
    window_duration = pd.Timedelta(minutes=30)
    current_time = recording_start

    with PdfPages(output_path) as pdf:
        while current_time < recording_end:
            window_end = current_time + window_duration
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
            
            n_chunk = nasal_df.loc[current_time:window_end] if not nasal_df.empty else pd.DataFrame()
            t_chunk = thoracic_df.loc[current_time:window_end] if not thoracic_df.empty else pd.DataFrame()
            s_chunk = spo2_df.loc[current_time:window_end] if not spo2_df.empty else pd.DataFrame()

            if not n_chunk.empty:
                axes[0].plot(n_chunk.index, n_chunk["Nasal Airflow"], color="#1f77b4", linewidth=0.8, rasterized=True)
            axes[0].set_title("Nasal Airflow (Filtered & Normalized)")
            axes[0].set_ylabel("Z-Score")

            if not t_chunk.empty:
                axes[1].plot(t_chunk.index, t_chunk["Thoracic Movement"], color="#ff7f0e", linewidth=0.8, rasterized=True)
            axes[1].set_title("Thoracic Movement (Filtered & Normalized)")
            axes[1].set_ylabel("Z-Score")

            if not s_chunk.empty:
                axes[2].plot(s_chunk.index, s_chunk["SpO2"], color="#2ca02c", linewidth=1.5, rasterized=True)
                axes[2].set_ylim(75, 100) 
            axes[2].set_title("SpO₂ (%) (Cleaned)")
            axes[2].set_ylabel("SpO₂")
            axes[2].set_xlabel("Clock Time")

            axes[2].set_xlim(current_time, window_end)
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            event_colors = {"apnea": "red", "hypopnea": "orange", "arousal": "purple", "desaturation": "brown"}
            legend_patches = {}

            if not events_df.empty and "start_time" in events_df.columns:
                for _, event in events_df.iterrows():
                    start = event["start_time"]
                    end = event["end_time"]
                    
                    if start < window_end and end > current_time:
                        e_type = str(event.get("event_type", "unknown")).lower()
                        base_type = "apnea" if "apnea" in e_type else ("hypopnea" if "hypopnea" in e_type else e_type)
                        color = event_colors.get(base_type, "blue")

                        for ax in axes:
                            ax.axvspan(start, end, color=color, alpha=0.3)
                        
                        if base_type not in legend_patches:
                            legend_patches[base_type] = mpatches.Patch(color=color, alpha=0.3, label=base_type.capitalize())

            if legend_patches:
                axes[0].legend(handles=list(legend_patches.values()), loc="upper right")

            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            
            current_time = window_end

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        return

    create_visualization(args.dir)

if __name__ == "__main__":
    main()