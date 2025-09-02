#!/usr/bin/env python3
# df360_plot_csv.py — quick SNR/RSSI/Bearing plots from DF360 logs
# Usage:
#   python -m pip install matplotlib pandas
#   python df360_plot_csv.py --csv path/to/log.csv
#   python df360_plot_csv.py --csv path/to/log.csv --live  # tail the file as DF360 writes

import argparse, time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_df(csv_path: Path, last_n=1000):
    try:
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    except Exception as e:
        print(f"[error] Could not read {csv_path}: {e}"); return None
    if last_n and len(df) > last_n:
        df = df.tail(last_n).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, type=Path)
    ap.add_argument('--live', action='store_true', help='Tail the CSV for updates')
    args = ap.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        print(f"No such file: {csv_path}"); return

    plt.ion()
    fig1 = plt.figure(); ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111)

    last_mtime = 0
    while True:
        if not args.live and last_mtime: break
        try:
            mtime = csv_path.stat().st_mtime
        except FileNotFoundError:
            time.sleep(0.5); continue
        if mtime != last_mtime:
            last_mtime = mtime
            df = load_df(csv_path)
            if df is None or df.empty:
                time.sleep(0.5); continue
            # Plot SNR and RSSI over time
            ax1.clear()
            ax1.plot(df['timestamp'], df['snr_db'], label='SNR (dB)')
            ax1.plot(df['timestamp'], df['rssi_dbm'], label='RSSI (dBm)')
            ax1.set_title('SNR / RSSI over time')
            ax1.legend(); ax1.grid(True); fig1.autofmt_xdate()

            # Plot bearing over time (wrap 0..360)
            ax2.clear()
            if 'bearing_deg' in df.columns:
                ax2.plot(df['timestamp'], df['bearing_deg'], label='Bearing (deg)')
                ax2.set_ylim(0, 360)
            ax2.set_title('Bearing over time')
            ax2.grid(True); fig2.autofmt_xdate()

            plt.pause(0.01)
        time.sleep(0.5)

if __name__ == '__main__':
    main()
