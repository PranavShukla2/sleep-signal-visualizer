import matplotlib.dates as mdates
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

def load_signal(file_path, col_name): 
    if not os.path.exists(file_path):
        print(f"Warning: Signal file not found: {file_path}")
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
        print(f"Warning: Events file not found: {file_path}")
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

    os.makedirs("Visualizations", exist_ok=True)
    output_path = os.path.join(
        "Visualizations", f"{participant_name}_visualization.pdf"
    )

    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
       
        if not nasal_df.empty:
            axes[0].plot(nasal_df.index, nasal_df["Nasal Airflow"], rasterized=True)
        axes[0].set_title("Nasal Airflow (32 Hz)")
        axes[0].set_ylabel("Amplitude")

        if not thoracic_df.empty:
            axes[1].plot(thoracic_df.index, thoracic_df["Thoracic Movement"], rasterized=True)
        axes[1].set_title("Thoracic Movement (32 Hz)")
        axes[1].set_ylabel("Amplitude")

        if not spo2_df.empty:
            axes[2].plot(spo2_df.index, spo2_df["SpO2"], rasterized=True)
        axes[2].set_title("SpO₂ (4 Hz)")
        axes[2].set_ylabel("SpO₂ (%)")
        axes[2].set_xlabel("Clock Time")

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        event_colors = {"apnea": "red", "hypopnea": "orange", "arousal": "purple"}
        legend_patches = {}

        if not events_df.empty and "start_time" in events_df.columns:
            for _, event in events_df.iterrows():
                start = event["start_time"]
                end = event["end_time"]
                
                e_type = str(event.get("event_type", "unknown")).lower()
                color = event_colors.get(e_type, "blue")

                for ax in axes:
                    ax.axvspan(start, end, color=color, alpha=0.3)
                
                if e_type not in legend_patches:
                    legend_patches[e_type] = mpatches.Patch(color=color, alpha=0.3, label=e_type.capitalize())

        if legend_patches:
            axes[0].legend(handles=list(legend_patches.values()), loc="upper right")

        plt.tight_layout()
        pdf.savefig(fig, dpi=300) 
        plt.close(fig)

    print(f"Visualization saved at: {output_path}")
def main():
    parser = argparse.ArgumentParser(description="Generate sleep signal visualization PDF")
   
    parser.add_argument("--dir", required=True, help="Path to participant folder (e.g., Data/AP20)")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Provided participant folder '{args.dir}' does not exist.")
        return

    create_visualization(args.dir)

if __name__ == "__main__":
    main()