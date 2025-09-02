
# DF360 Operator Guide

This guide covers setup, operating the GUI, CLI usage, logging modes, mapping, and troubleshooting.

---

## 1) Install & Requirements

### Windows (recommended steps)
1. Install Python 3.11+.
2. In PowerShell:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install PySide6 matplotlib folium pandas numpy scipy colorama pynmea2 pyyaml pyrtlsdr
   ```
3. For **non-RTL** radios, install **SoapySDR** runtime and your device plugin (Airspy/HackRF/Lime/SDRplay).  
   - Windows: PothosSDR bundle; ensure `SoapySDR` is on PATH.
4. **RTL-SDR (USB)** on Windows: run **Zadig** and select **WinUSB** driver.
5. Place these files together in a working folder:
   - `DF360.py` (the logger/DF engine)
   - `sdr_backend.py` (multi-SDR abstraction)
   - `df360_gui.py` (the GUI)
   - Optional: `config.yaml` (defaults)

### macOS / Linux
- Install Homebrew/APT packages for SoapySDR & your device plugin (only if using Soapy).
- Install Python deps as above.

---

## 2) Quick Start (GUI)

```powershell
python df360_gui.py
```
1. **DF360.py**: browse to your DF360 script.
2. (Optional) **config.yaml**: select a preset.
3. Choose SDR **Kind** and **Device** (empty = auto/local).
4. Set **Center (MHz)**, **Sample rate**, **Gain**, **Interval**.
5. Enter GPS **IP**/**Port** (your phone/GPS tether).
6. (Optional) Enable **Gate on TX**, set **SNR threshold** and **Hang**.
7. Click **Start**.

- The **Logs** tab shows live console output + parsed SNR/RSSI/Noise/Bearing.
- The **Charts** tab tails the CSV and draws SNR/RSSI & Bearing vs time.
- The **Map** tab draws recent points colored by SNR with a short **bearing arrow**.

> The GUI auto-detects which CLI flags your DF360 supports by probing `-h`, so it won’t pass unsupported flags to older builds.

---

## 3) Logging Modes

### Continuous
Default behavior — every interval a row is logged.

### Gate on TX (sit-and-wait)
Enable gating to only log while **SNR ≥ threshold**. A **hang time** keeps logging briefly after the signal dips to avoid gaps.

**CLI flags (if supported by your DF360):**
- `--gate-logging` / `--no-gate-logging`
- `--snr-threshold-db 12.0`
- `--hang-s 2.0`

**Example:**
```powershell
python DF360.py --noninteractive --center-mhz 170.0 --interval 5 --gate-logging --snr-threshold-db 12 --hang-s 2
```

---

## 4) SDR Backends

- **RTL (USB):** `--sdr-kind rtl`
- **RTL-TCP:** `--sdr-kind rtltcp --sdr-device 192.168.4.1:1234`
- **SoapySDR (Airspy/HackRF/Lime/SDRplay):** `--sdr-kind soapy --sdr-device "driver=airspy"`
- **Auto:** `--sdr-kind auto`

List devices (if supported): `--list-sdr`

---

## 5) GPS Input

- DF360 expects NMEA from a **TCP server** (e.g., GPS Tether on phone).
- Set **IP**/**Port** in GUI or CLI: `--ip 192.168.1.247 --port 8080`.

---

## 6) Output & Files

- Logs: `logs/<trip_slug>/rf_log_<trip>_<session>.csv`
- Resolved run config saved alongside the CSV: `resolved_config.json`
- IQ samples: `iq_samples/` (falls back to `~/DF360/iq_samples` on permission errors)

**CSV schema** (highlights; may grow over time):
- `timestamp, latitude, longitude, altitude_m, speed_kph, course_deg, num_satellites, hdop`
- `rssi_dbm, snr_db, noise_floor_db, peak_freq_mhz, bandwidth_khz, freq_offset_hz`
- `carrier_freq_mhz, resonant_freq_mhz, tuning_error_mhz, antenna_gain_db, adjusted_rssi_dbm`
- `spectral_entropy, spectral_flatness, papr_db, power_std_db, k_factor_db`
- `bearing_deg, bearing_std_deg`
- `triangulated_lat, triangulated_lon, triangulation_range_m, triangulation_bearing_deg` (when available)
- `event_detected, schema_version`

---

## 7) Charts & Map

- **Charts** tab shows **SNR/RSSI** and **Bearing** vs time, tailing the most recent ~1000 rows.
- **Map** shows last ~1500 points, colored by **SNR** with **bearing arrows**.
- Use **Refresh Map** to regenerate on demand; it auto-refreshes when the tab is visible.

---

## 8) CLI Cookbook (headless)

```powershell
# List radios (newer DF360)
python DF360.py --list-sdr

# 3-minute run at 155.250 MHz, 10s interval
python DF360.py --noninteractive --center-mhz 155.25 --interval 10 --duration-s 180

# Airspy via Soapy
python DF360.py --noninteractive --sdr-kind soapy --sdr-device "driver=airspy" --center-mhz 146.52 --interval 5

# RTL-TCP radio on network
python DF360.py --noninteractive --sdr-kind rtltcp --sdr-device 192.168.4.1:1234 --center-mhz 462.5625 --interval 3
```

---

## 9) Troubleshooting

- **“Unrecognized arguments”**: Your DF360 is older. Replace it with the latest or rely on the GUI’s auto-detection (it won’t pass unknown flags).
- **PermissionError on `iq_samples`**: DF360 now falls back to `~/DF360/iq_samples`. Or set `--iq-dir` to a writable path.
- **No devices found**: Install drivers (Zadig for RTL), SoapySDR runtime for non-RTL; try `--list-sdr`.
- **GPS timeout**: Verify IP/port and that your phone is on the same network; check firewall rules.
- **Soapy init fails**: Ensure the correct driver string, e.g., `driver=hackrf`, and that the plugin is installed.

---

## 10) Packaging (optional)

To ship a single EXE of the **GUI** (includes DF360 + backend):
```powershell
python -m pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed df360_gui.py
```
Include `DF360.py`, `sdr_backend.py`, and any YAMLs next to the EXE or embed via a spec file (advanced).

---

## 11) Safety & Best Practices

- Observe local radio regulations; only monitor allowed spectrum.
- Keep antennas clear of power lines and moving parts.
- Stabilize the laptop on a flat surface; don’t operate while driving.
- Use reasonable gain to avoid overload.
