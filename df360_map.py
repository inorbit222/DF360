#!/usr/bin/env python3
# df360_map.py — generate an interactive HTML map from a DF360 CSV log
# Usage:
#   python -m pip install folium pandas
#   python df360_map.py --csv path/to/log.csv --out map.html

import argparse, math
import pandas as pd
from pathlib import Path

def color_for_snr(snr):
    if pd.isna(snr): return 'gray'
    if snr >= 20: return 'green'
    if snr >= 10: return 'orange'
    return 'red'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, type=Path)
    ap.add_argument('--out', default='df360_map.html', type=Path)
    ap.add_argument('--step', type=int, default=5, help='Plot every Nth point to avoid clutter')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['timestamp'])
    if df.empty:
        print('No data'); return

    import folium
    center = [df['latitude'].mean(), df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')

    for i in range(0, len(df), args.step):
        row = df.iloc[i]
        lat, lon = float(row['latitude']), float(row['longitude'])
        snr = float(row['snr_db']) if not pd.isna(row['snr_db']) else None
        color = color_for_snr(snr)
        text = f"{row['timestamp']}\nSNR: {snr:.1f} dB\nBearing: {row.get('bearing_deg', float('nan')):.1f}°"
        folium.CircleMarker(location=[lat, lon], radius=4, color=color, fill=True, fill_opacity=0.8, popup=text).add_to(m)
    m.save(str(args.out))
    print('Wrote', args.out)

if __name__ == '__main__':
    main()
