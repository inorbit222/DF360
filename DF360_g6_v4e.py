
# DF360_g6 v4e — Windows-safe atomic JSON + decoupled spectrum worker
import os, re, sys, json, glob, time, math, argparse, threading
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

try:
    from scipy.signal import welch as _scipy_welch  # type: ignore
except Exception:
    _scipy_welch = None

from rtlsdr import RtlSdr
from colorama import Fore, Style, init as colorama_init

try:
    import yaml
except Exception:
    yaml = None

from gps_mux import GPSMux

colorama_init(autoreset=True)

DEFAULT_CONFIG: Dict[str, Any] = {
    "center_mhz": 173.35,
    "sample_rate": 2_400_000,
    "gain": "auto",
    "interval_s": 10,          # CSV/log interval
    "spectrum": {
        "out": "auto",         # <trip>/live_spectrum.json
        "interval_s": 0.25,    # spectrum refresh
        "nperseg": 1024,
        "avg_frames": 1
    },
    "gps": {
        "source": "auto",
        "priority": ["tcp","udp","serial","gpsd","manual"],
        "host": "192.168.1.156", "port": 8080,
        "udp_port": 10110,
        "gpsd_host": "127.0.0.1",
        "gpsd_port": 2947,
        "serial_port": "COM5",
        "serial_baud": 9600,
        "manual_lat": 40.6000, "manual_lon": -124.1500,
        "fix_timeout_s": 15.0, "stale_age_s": 30.0,
        "min_hdop": 4.0,
        "prefer_talkers": ["GN","GP","GL","GA"],
        "last_fix_cache": "df_last_fix.json",
        "allow_no_fix": True, "manual_fallback": True, "debug_lines": 120
    },
    "log": {"out_dir": "logs"},
    "trip": {"slug": ""},
}

def welch_psd(x: np.ndarray, fs: float, nperseg: int = 1024, overlap: float = 0.5):
    if _scipy_welch:
        return _scipy_welch(x, fs=fs, nperseg=nperseg, noverlap=int(nperseg*overlap))
    step = int(nperseg * (1 - overlap)); step = max(1, step)
    if len(x) < nperseg:
        x = np.pad(x, (0, nperseg - len(x)), mode="constant")
    window = np.hanning(nperseg)
    scale = (np.sum(window**2) * fs)
    segments = []
    for start in range(0, len(x) - nperseg + 1, step):
        seg = x[start:start+nperseg] * window
        spec = np.fft.rfft(seg)
        psd = (np.abs(spec)**2) / scale
        segments.append(psd)
    if not segments:
        return np.fft.rfftfreq(nperseg, d=1/fs), np.zeros(nperseg//2+1)
    Pxx = np.mean(np.vstack(segments), axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1/fs)
    return freqs, Pxx

def load_config(path: Path) -> Dict[str, Any]:
    if path and path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in (".yaml", ".yml") and yaml:
            raw = yaml.safe_load(text) or {}
            return {**DEFAULT_CONFIG, **raw}
        try:
            raw = json.loads(text) or {}
            return {**DEFAULT_CONFIG, **raw}
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CONFIG))

def build_parser():
    p = argparse.ArgumentParser(description="DF360 – RF logger (decoupled spectrum)")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--noninteractive", action="store_true")
    p.add_argument("--center-mhz", type=float, dest="center_mhz")
    p.add_argument("--sample-rate", type=float, dest="sample_rate")
    p.add_argument("--gain", type=str)
    p.add_argument("--interval", type=int, dest="interval_s")
    p.add_argument("--trip", type=str)
    p.add_argument("--outdir", type=Path)
    # GPS quick opts
    p.add_argument("--gps-source", choices=["auto","tcp","udp","serial","gpsd","manual"])
    p.add_argument("--ip", type=str, dest="gps_host")
    p.add_argument("--port", type=int, dest="gps_port")
    # Spectrum controls
    p.add_argument("--spec-interval", type=float, dest="spec_interval_s")
    p.add_argument("--spec-nperseg", type=int, dest="spec_nperseg")
    p.add_argument("--spec-avg", type=int, dest="spec_avg_frames")
    p.add_argument("--allow-preflight-fail", action="store_true")
    return p

def merge_cli_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    if args.center_mhz:   cfg["center_mhz"] = float(args.center_mhz)
    if args.sample_rate:  cfg["sample_rate"] = float(args.sample_rate)
    if args.gain is not None: cfg["gain"] = args.gain
    if args.interval_s:   cfg["interval_s"] = int(args.interval_s)
    if args.trip:         cfg["trip"]["slug"] = args.trip
    if args.outdir:       cfg["log"]["out_dir"] = str(args.outdir)
    if args.gps_source:   cfg["gps"]["source"] = args.gps_source
    if args.gps_host:     cfg["gps"]["host"] = args.gps_host
    if args.gps_port:     cfg["gps"]["port"] = int(args.gps_port)
    if args.spec_interval_s is not None: cfg["spectrum"]["interval_s"] = float(args.spec_interval_s)
    if args.spec_nperseg: cfg["spectrum"]["nperseg"] = int(args.spec_nperseg)
    if args.spec_avg_frames: cfg["spectrum"]["avg_frames"] = max(1,int(args.spec_avg_frames))
    cfg["allow_fail"] = bool(args.allow_preflight_fail)
    return cfg

def _check_sdr(cfg) -> bool:
    try:
        sdr = RtlSdr()
        sdr.sample_rate = float(cfg["sample_rate"])
        sdr.center_freq = float(cfg["center_mhz"]) * 1e6
        g = cfg["gain"]
        sdr.gain = 'auto' if str(g).lower() == 'auto' else float(g)
        sdr.close()
        return True
    except Exception as e:
        print(f"{Fore.RED}[CHECK] SDR error: {e}{Style.RESET_ALL}")
        return False

def preflight_checks(cfg, out_dir: Path, noninteractive=False, allow_fail=False) -> bool:
    print(f"{Fore.CYAN}[CHECK] Freq={cfg['center_mhz']} MHz  SR={cfg['sample_rate']}  Gain={cfg['gain']}  Interval={cfg['interval_s']}s{Style.RESET_ALL}")
    sdr_ok = _check_sdr(cfg)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".write_test").write_text("ok", encoding="utf-8")
        (out_dir / ".write_test").unlink(missing_ok=True)
        out_ok = True
    except Exception as e:
        print(f"{Fore.RED}[CHECK] Cannot write to {out_dir}: {e}{Style.RESET_ALL}")
        out_ok = False
    all_ok = sdr_ok and out_ok
    if not all_ok and noninteractive and not allow_fail:
        print(f"{Fore.RED}[CHECK] Preflight failed in noninteractive mode{Style.RESET_ALL}")
    elif not all_ok:
        print(f"{Fore.YELLOW}[CHECK] One or more checks failed; continuing...{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}[CHECK] All good{Style.RESET_ALL}")
    return all_ok or (not noninteractive) or allow_fail

def _atomic_write_json(path: Path, obj: dict):
    # Robust atomic replace for Windows: retry when destination is temporarily locked.
    import time, os, json
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{int(time.time()*1000)}")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    delay = 0.01
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(delay); delay = min(0.10, delay * 1.5)
        except OSError as e:
            if getattr(e, "winerror", None) in (5, 32):  # Access denied / sharing violation
                time.sleep(delay); delay = min(0.10, delay * 1.5)
            else:
                raise
    try:
        path.write_text(json.dumps(obj), encoding="utf-8")
    except Exception:
        pass
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

class SDRWorker(threading.Thread):
    """Continuously reads small chunks from RTL-SDR and emits PSD + live_spectrum.json."""
    def __init__(self, sdr: RtlSdr, center_mhz: float, sample_rate: float, out_path: Path,
                 spec_interval: float, nperseg: int, avg_frames: int):
        super().__init__(daemon=True)
        self.sdr = sdr
        self.center_mhz = center_mhz
        self.sample_rate = float(sample_rate)
        self.out_path = out_path
        self.spec_interval = max(0.05, float(spec_interval))
        self.nperseg = int(nperseg)
        self.avg_frames = max(1, int(avg_frames))
        self._stop = threading.Event()
        self.latest = None  # dict with 'freqs','psd','noise','snr','peak'

    def stop(self):
        self._stop.set()

    def run(self):
        nsamp = int(min(self.sample_rate * 0.2, 262144))  # ~0.2s or 256k
        overlap = 0.5
        last_emit = 0.0
        while not self._stop.is_set():
            try:
                x = self.sdr.read_samples(nsamp)
                P = None
                for _ in range(self.avg_frames):
                    f, psd = welch_psd(x, fs=self.sample_rate, nperseg=self.nperseg, overlap=overlap)
                    P = psd if P is None else (P + psd)
                psd = P / float(self.avg_frames)
                peak_idx = int(np.argmax(psd))
                peak_freq_mhz = float(f[peak_idx] / 1e6)
                noise = float(np.mean(psd))
                noise_db = 10*np.log10(max(noise,1e-20))
                snr_db = 10*np.log10(max(float(np.max(psd)),1e-20)/max(noise,1e-20))
                psd_db = (10*np.log10(psd + 1e-20)).astype(float)
                self.latest = {
                    "t": datetime.utcnow().isoformat(),
                    "center_mhz": self.center_mhz,
                    "sample_rate": float(self.sample_rate),
                    "freqs_mhz": (f/1e6).astype(float),
                    "psd_db": psd_db,
                    "peak_freq_mhz": peak_freq_mhz,
                    "noise_floor_db": noise_db,
                    "snr_db": float(snr_db),
                }
                now = time.time()
                if now - last_emit >= self.spec_interval:
                    _atomic_write_json(self.out_path, {
                        "t": self.latest["t"],
                        "center_mhz": self.center_mhz,
                        "sample_rate": float(self.sample_rate),
                        "freqs_mhz": self.latest["freqs_mhz"].tolist(),
                        "psd_db": self.latest["psd_db"].tolist(),
                        "peak_freq_mhz": peak_freq_mhz,
                        "noise_floor_db": noise_db,
                        "snr_db": float(snr_db),
                    })
                    last_emit = now
            except Exception as e:
                print(f"{Fore.YELLOW}[SDRWorker] read error: {e}{Style.RESET_ALL}")
                time.sleep(self.spec_interval)

def _rtl_init(center_mhz: float, sample_rate: float, gain):
    sdr = RtlSdr()
    sdr.sample_rate = float(sample_rate)
    sdr.center_freq = float(center_mhz) * 1e6
    try:
        sdr.gain = 'auto' if str(gain).lower() == 'auto' else float(gain)
    except Exception:
        sdr.gain = 'auto'
    return sdr

def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    cfg["trip"]["slug"] = (cfg["trip"].get("slug") or datetime.utcnow().strftime("trip-%Y%m%d-%H%M%S"))
    out_dir = Path(cfg["log"]["out_dir"]) / cfg["trip"]["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if not preflight_checks(cfg, out_dir, noninteractive=args.noninteractive, allow_fail=cfg.get("allow_fail", False)):
        sys.exit(2)

    spec_path = cfg.get("spectrum",{}).get("out","auto")
    if not spec_path or spec_path == "auto":
        spec_path = str(out_dir / "live_spectrum.json")
    spec_path = Path(spec_path)

    # RTL-SDR
    try:
        sdr = _rtl_init(center_mhz=float(cfg["center_mhz"]), sample_rate=float(cfg["sample_rate"]), gain=cfg["gain"])
        print(f"[SDR] center={cfg['center_mhz']} MHz | sr={cfg['sample_rate']} | gain={cfg['gain']}")
    except Exception as e:
        print(f"{Fore.RED}[SDR] failed to initialize: {e}{Style.RESET_ALL}")
        sys.exit(3)

    # GPS mux
    mux_cfg = {
        "source": cfg["gps"].get("source", "auto"),
        "priority": cfg["gps"].get("priority", ["tcp","udp","serial","gpsd","manual"]),
        "tcp_host": cfg["gps"].get("host","127.0.0.1"),
        "tcp_port": int(cfg["gps"].get("port",8080)),
        "udp_port": int(cfg["gps"].get("udp_port",10110)),
        "gpsd_host": cfg["gps"].get("gpsd_host","127.0.0.1"),
        "gpsd_port": int(cfg["gps"].get("gpsd_port",2947)),
        "serial_port": cfg["gps"].get("serial_port","COM5"),
        "serial_baud": int(cfg["gps"].get("serial_baud",9600)),
        "manual_lat": float(cfg["gps"].get("manual_lat",0.0)),
        "manual_lon": float(cfg["gps"].get("manual_lon",0.0)),
        "stale_age_s": float(cfg["gps"].get("stale_age_s",30.0)),
        "fix_timeout_s": float(cfg["gps"].get("fix_timeout_s",15.0)),
        "min_hdop": float(cfg["gps"].get("min_hdop",0.0)),
        "last_fix_cache": cfg["gps"].get("last_fix_cache","df_last_fix.json"),
        "allow_no_fix": bool(cfg["gps"].get("allow_no_fix", True)),
        "manual_fallback": bool(cfg["gps"].get("manual_fallback", True)),
        "prefer_talkers": cfg["gps"].get("prefer_talkers",["GN","GP","GL","GA"]),
        "debug_lines": int(cfg["gps"].get("debug_lines",120)),
    }
    gps_mux = GPSMux(mux_cfg)

    # Start SDR worker
    sw = SDRWorker(
        sdr=sdr,
        center_mhz=float(cfg["center_mhz"]),
        sample_rate=float(cfg["sample_rate"]),
        out_path=spec_path,
        spec_interval=float(cfg.get("spectrum",{}).get("interval_s", 0.25)),
        nperseg=int(cfg.get("spectrum",{}).get("nperseg", 1024)),
        avg_frames=int(cfg.get("spectrum",{}).get("avg_frames", 1)),
    )
    sw.start()
    print(f"[SDR] spectrum @ {cfg.get('spectrum',{}).get('interval_s',0.25)}s -> {spec_path}")

    # CSV logging loop (uses latest PSD from worker, non-blocking)
    session_id = str(uuid4())[:8]
    trip_slug  = re.sub(r'[^a-zA-Z0-9_]+', '_', cfg["trip"]["slug"].strip().lower())[:32]
    csv_path = out_dir / f"rf_log_{trip_slug}_{session_id}.csv"

    LOG_INTERVAL = int(cfg["interval_s"])
    last_log = 0.0

    try:
        while True:
            time.sleep(0.05)
            now = time.time()
            if (now - last_log) < LOG_INTERVAL:
                continue

            spec = sw.latest
            if not spec:
                continue

            peak_freq_mhz = float(spec["peak_freq_mhz"])
            noise_floor_db = float(spec["noise_floor_db"])
            snr_db = float(spec["snr_db"])

            # GPS snapshot
            fix, gps_state, fix_age = gps_mux.get_last_fix()
            lat=lon=alt=spd=crs=sats=hdop=talker=None; fix_quality=None
            if fix:
                lat=getattr(fix,"lat",None); lon=getattr(fix,"lon",None)
                alt=getattr(fix,"alt_m",None); spd=getattr(fix,"speed_kph",None)
                crs=getattr(fix,"course_deg",None); hdop=getattr(fix,"hdop",None)
                sats=getattr(fix,"sats",None); talker=getattr(fix,"talker",None)
                fix_quality=getattr(fix,"fix_type",None)

            entry = dict(
                timestamp=datetime.utcnow().isoformat(),
                latitude=lat, longitude=lon, altitude_m=alt,
                speed_kph=spd, course_deg=crs, num_satellites=sats, hdop=hdop,
                rssi_dbm=None,
                noise_floor_db=noise_floor_db,
                snr_db=snr_db,
                freq_mhz=float(cfg["center_mhz"]),
                peak_freq_mhz=peak_freq_mhz,
                bandwidth_khz=None,
                freq_offset_hz=(peak_freq_mhz - float(cfg["center_mhz"])) * 1e6,
                gps_state=gps_state,
                fix_age_s=(round(float(fix_age),2) if fix_age != 1e9 else ""),
                talker=talker or "", fix_quality=fix_quality or "",
                gps_provider=getattr(gps_mux,"provider_name",""),
            )

            pd.DataFrame([entry]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            print(f"[LOG] CSV tick @ {entry['timestamp']} | SNR={snr_db:.1f} dB | GPS={gps_state}")
            last_log = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try: sw.stop()
        except Exception: pass
        try: sw.join(timeout=1.0)
        except Exception: pass
        try: sdr.close()
        except Exception: pass
        try: gps_mux.stop()
        except Exception: pass
        print(f"[INFO] Final log saved to {csv_path}")

if __name__ == '__main__':
    main()
