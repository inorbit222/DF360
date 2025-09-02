# df360_gui_g15 — Full GUI (profiles, SDR, charts, map) + embedded Spectrum tab (auto-discovers live_spectrum.json)
from __future__ import annotations

import csv
import io
import json
import math
import os
import socket
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

# ---------- Optional dependencies ----------
try:
    import yaml  # PyYAML for profiles
except Exception:  # pragma: no cover
    yaml = None

try:
    from PyQt5 import QtWebEngineWidgets  # optional
    WEBENGINE_OK = True
except Exception:
    WEBENGINE_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # We'll embed via canvas; backend set later on canvas creation
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MPL_OK = True
except Exception:
    MPL_OK = False

# Optional: spectrum viewer
try:
    import pyqtgraph as pg
    PG_OK = True
except Exception:
    PG_OK = False

import webbrowser
import tempfile
import time

# ---------- App constants & dirs ----------
APP_NAME = "DF360"
CFG_DIR = Path.home() / f".{APP_NAME.lower()}"
PROFILES_DIR = CFG_DIR / "profiles"


def _ensure_dirs():
    CFG_DIR.mkdir(exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def safe_iq_root() -> Path:
    # Prefer Documents\\DF360\\iq; fallback to ~/.df360/iq
    candidates = [
        Path.home() / "Documents" / APP_NAME / "iq",
        CFG_DIR / "iq",
    ]
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".writetest"
            t.write_text("ok", encoding="utf-8")
            t.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    return candidates[-1]


def safe_logs_root() -> Path:
    # Prefer Documents\\DF360\\logs; fallback to ~/.df360/logs
    candidates = [
        Path.home() / "Documents" / APP_NAME / "logs",
        CFG_DIR / "logs",
    ]
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            continue
    return candidates[-1]


# ---------- Small helpers ----------
class HelpCache:
    """Cache DF360 --help output to detect supported flags quickly."""

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def supports_flag(self, df360_path: str, flag: str) -> bool:
        if not df360_path:
            return False
        if df360_path in self._cache:
            out = self._cache[df360_path]
        else:
            try:
                out = subprocess.check_output(
                    [sys.executable, df360_path, "--help"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=4,
                )
            except Exception:
                out = ""
            self._cache[df360_path] = out
        return flag in out


@dataclass
class SessionProfile:
    center_mhz: Union[float, str] = ""
    sample_rate: Union[float, str] = ""
    gain: str = ""
    interval_s: Union[int, str] = ""
    gps_ip: str = ""
    gps_port: Union[int, str] = ""
    iq_dir: str = ""
    out_dir: str = ""
    gate_enable: bool = False
    snr_threshold_db: Union[float, str, None] = None
    hang_s: Union[float, str, None] = None

    @staticmethod
    def from_main(win: QtWidgets.QWidget) -> "SessionProfile":
        g = lambda n, d="": getattr(win, n).text() if hasattr(win, n) else d
        b = lambda n, d=False: getattr(win, n).isChecked() if hasattr(win, n) else d
        iq = getattr(win, "iqDirEdit", None)
        outd = getattr(win, "outDirEdit", None)
        return SessionProfile(
            center_mhz=g("centerFreqEdit"),
            sample_rate=g("sampleRateEdit"),
            gain=g("gainEdit"),
            interval_s=g("intervalEdit"),
            gps_ip=g("ipEdit"),
            gps_port=g("portEdit"),
            iq_dir=iq.text() if isinstance(iq, QtWidgets.QLineEdit) else str(safe_iq_root()),
            out_dir=outd.text() if isinstance(outd, QtWidgets.QLineEdit) else str(safe_logs_root()),
            gate_enable=b("gateCheck"),
            snr_threshold_db=(g("gateThreshEdit", "") or None),
            hang_s=(g("gateHangEdit", "") or None),
        )

    def apply_to_main(self, win: QtWidgets.QWidget) -> None:
        def set_text(name: str, val):
            if hasattr(win, name) and val not in (None, ""):
                getattr(win, name).setText(str(val))
        set_text("centerFreqEdit", self.center_mhz)
        set_text("sampleRateEdit", self.sample_rate)
        set_text("gainEdit", self.gain)
        set_text("intervalEdit", self.interval_s)
        set_text("ipEdit", self.gps_ip)
        set_text("portEdit", self.gps_port)
        if hasattr(win, "iqDirEdit") and isinstance(win.iqDirEdit, QtWidgets.QLineEdit) and self.iq_dir:
            win.iqDirEdit.setText(self.iq_dir)
        if hasattr(win, "outDirEdit") and isinstance(win.outDirEdit, QtWidgets.QLineEdit) and self.out_dir:
            win.outDirEdit.setText(self.out_dir)
        if hasattr(win, "gateCheck"):
            win.gateCheck.setChecked(bool(self.gate_enable))
        set_text("gateThreshEdit", self.snr_threshold_db)
        set_text("gateHangEdit", self.hang_s)


class ProfilesStore:
    def __init__(self):
        _ensure_dirs()

    def list_names(self) -> List[str]:
        return sorted(p.stem for p in PROFILES_DIR.glob("*.yml"))

    def load(self, name: str) -> SessionProfile:
        p = PROFILES_DIR / f"{name}.yml"
        if not p.exists() or yaml is None:
            return SessionProfile()
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        keys = asdict(SessionProfile()).keys()
        return SessionProfile(**{k: data.get(k, getattr(SessionProfile, k, "")) for k in keys})

    def save(self, name: str, prof: SessionProfile) -> None:
        if yaml is None:
            raise RuntimeError("PyYAML is required for profiles support.")
        (PROFILES_DIR / f"{name}.yml").write_text(yaml.safe_dump(asdict(prof), sort_keys=False), encoding="utf-8")

    def delete(self, name: str) -> None:
        p = PROFILES_DIR / f"{name}.yml"
        if p.exists():
            p.unlink()


class DeviceTools:
    @staticmethod
    def test_rtl_tcp(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, str]:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True, f"Connected to {host}:{port}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @staticmethod
    def lazy_import_soapy():
        try:
            import SoapySDR  # type: ignore
            return True, SoapySDR
        except Exception as e:
            return False, f"SoapySDR not available: {e}"

    @staticmethod
    def enumerate_soapy():
        ok, s = DeviceTools.lazy_import_soapy()
        if not ok:
            return s
        try:
            return s.Device.enumerate()
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def test_soapy_device(dev_args: dict) -> Tuple[bool, str]:
        ok, s = DeviceTools.lazy_import_soapy()
        if not ok:
            return False, s  # type: ignore
        try:
            d = s.Device(dev_args)
            d.listAntennas(s.Direction.RX, 0)
            d.close()
            return True, f"Opened {dev_args}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


# ---------- CSV utilities ----------
def find_latest_csv(outdir: Path, trip_hint: str = "") -> Optional[Path]:
    """Return the newest CSV under outdir, searching recursively and preferring trip matches.
    Also handles engines that create a nested 'logs' directory under the chosen outdir.
    """
    if not outdir.exists():
        return None

    search_roots = [outdir]
    nested = outdir / "logs"
    if nested.exists():
        search_roots.insert(0, nested)  # prefer nested/logs first

    candidates = []
    for root in search_roots:
        candidates.extend(root.rglob("*.csv"))

    if not candidates:
        return None

    if trip_hint:
        trip_lower = trip_hint.lower()
        candidates = sorted(
            candidates,
            key=lambda p: (trip_lower in p.name.lower(), p.stat().st_mtime),
            reverse=True,
        )
    else:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0] if candidates else None


def tail_lines(path: Path, max_lines: int = 500) -> List[str]:
    # Efficient-ish tail; reads from the end in blocks
    lines: List[str] = []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            block = 4096
            data = b""
            while len(lines) <= max_lines and f.tell() > 0:
                seek = max(0, f.tell() - block)
                f.seek(seek)
                data = f.read(filesize - seek) + data
                f.seek(seek)
                filesize = seek
                lines = data.splitlines()
                if seek == 0:
                    break
            text_lines = [line.decode("utf-8", errors="replace") for line in lines[-max_lines:]]
            return text_lines
    except Exception:
        return []
    return []


def parse_csv_rows(lines: List[str]) -> Tuple[List[dict], Dict[str, int]]:
    """Return list of row dicts and a header index map."""
    if not lines:
        return [], {}
    # find header (first line that looks like csv with commas)
    header = None
    for i, line in enumerate(lines):
        if "," in line:
            header = line
            data_start = i + 1
            break
    if header is None:
        return [], {}
    header_cols = [h.strip().lower() for h in header.split(",")]
    idx = {name: i for i, name in enumerate(header_cols)}
    rows = []
    for line in lines[data_start:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header_cols):
            continue
        row = {name: parts[i] for name, i in idx.items()}
        rows.append(row)
    return rows, idx


def to_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def deglen_lat(lat_deg: float) -> float:
    # degrees to kilometers (~111km per degree latitude)
    return 111.32


def deglen_lon(lat_deg: float) -> float:
    # degrees to kilometers varies with latitude
    return 111.32 * math.cos(math.radians(lat_deg))


def project_bearing(lat: float, lon: float, bearing_deg: float, length_km: float) -> Tuple[float, float]:
    """Return endpoint lat2, lon2 given start lat/lon, bearing, and length (km)."""
    # Simple equirectangular approximation for short distances
    dlat_km = math.cos(math.radians(90 - bearing_deg)) * length_km
    dlon_km = math.sin(math.radians(bearing_deg)) * length_km
    lat2 = lat + dlat_km / deglen_lat(lat)
    lon2 = lon + dlon_km / deglen_lon(lat)
    return lat2, lon2


# ---------- Matplotlib canvas ----------
class ChartsCanvas(FigureCanvas):
    def __init__(self, parent=None):
        if not MPL_OK:
            raise RuntimeError("matplotlib is not available")
        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax1 = self.fig.add_subplot(211)  # SNR
        self.ax2 = self.fig.add_subplot(212)  # RSSI/Noise
        self.snrs = deque(maxlen=1000)
        self.rssis = deque(maxlen=1000)
        self.noises = deque(maxlen=1000)
        self.bearings = deque(maxlen=1000)

        self.ln_snr, = self.ax1.plot([], [], label="SNR (dB)")
        self.ax1.set_ylabel("SNR (dB)")
        self.ax1.grid(True)
        self.ax1.legend(loc="upper right")

        self.ln_rssi, = self.ax2.plot([], [], label="RSSI (dB)")
        self.ln_noise, = self.ax2.plot([], [], label="Noise (dB)")
        self.ax2.set_xlabel("Samples")
        self.ax2.set_ylabel("Level (dB)")
        self.ax2.grid(True)
        self.ax2.legend(loc="upper right")

    def update_series(self, new_snrs: List[float], new_rssis: List[float], new_noises: List[float]):
        # Append new values into deques
        for v in new_snrs:
            self.snrs.append(v)
        for v in new_rssis:
            self.rssis.append(v)
        for v in new_noises:
            self.noises.append(v)

        # Update lines safely
        n1 = self._safe_set(self.ln_snr, self.snrs)
        # Axis 1 relim after setting SNR
        self.ax1.relim(); self.ax1.autoscale_view()

        n2 = self._safe_set(self.ln_rssi, self.rssis)
        n3 = self._safe_set(self.ln_noise, self.noises)
        # Axis 2 relim after setting both
        self.ax2.relim(); self.ax2.autoscale_view()

        self.draw_idle()

    def _safe_set(self, line, seq):
        import numpy as _np
        # Convert to float array
        try:
            y = _np.asarray(list(seq), dtype=float)
        except Exception:
            y = _np.asarray([], dtype=float)
        if y.size == 0:
            line.set_data(_np.empty(0, dtype=float), _np.empty(0, dtype=float))
            return 0
        # Replace NaN/inf to keep Matplotlib happy
        if not _np.isfinite(y).all():
            y = _np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x = _np.arange(y.size, dtype=float)
        line.set_data(x, y)
        return y.size
        if len(y) == 0:
            line.set_data([], [])
            return
        x = list(range(len(y)))
        line.set_data(x, y)

        x2 = range(len(self.rssis))
        self.ln_rssi.set_data(list(x2), list(self.rssis))
        self.ln_noise.set_data(list(x2), list(self.noises))
        self.ax2.relim(); self.ax2.autoscale_view()

        self.draw_idle()


# ---------- Dashboard metric widgets & math ----------
FIELD_MODE_QSS = """
*[role="metricTitle"] { font-size: 12pt; opacity: 0.75; }
*[role="metricValue"] { font-size: 28pt; font-weight: 700; }
*[role="metricSub"]   { font-size: 11pt; opacity: 0.8; }
QToolButton, QPushButton { padding: 8px 14px; font-size: 12pt; }
"""

class MetricCard(QtWidgets.QWidget):
    """Simple metric card used on the dashboard tab."""
    def __init__(self, title: str, unit: str = "", parent=None):
        super().__init__(parent)
        self._unit = unit
        self._title = QtWidgets.QLabel(title)
        self._value = QtWidgets.QLabel("—")
        self._sub   = QtWidgets.QLabel("")
        for w in (self._title, self._value, self._sub):
            w.setAlignment(QtCore.Qt.AlignCenter)
        self._title.setProperty("role", "metricTitle")
        self._value.setProperty("role", "metricValue")
        self._sub.setProperty("role", "metricSub")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self._title)
        lay.addWidget(self._value)
        lay.addWidget(self._sub)

    def set_value(self, val, subtext: str = ""):
        if val is None:
            self._value.setText("—")
            self._sub.setText("")
            return
        if self._unit == "MHz":
            self._value.setText(f"{float(val):,.3f} {self._unit}")
        elif self._unit == "dB":
            self._value.setText(f"{float(val):+.1f} {self._unit}")
        elif self._unit == "deg":
            self._value.setText(f"{float(val):03.0f}°")
        else:
            self._value.setText(str(val))
        self._sub.setText(subtext)

def _cardinal16(deg: float) -> str:
    dirs = [
        "N","NNE","NE","ENE","E","ESE","SE","SSE",
        "S","SSW","SW","WSW","W","WNW","NW","NNW",
    ]
    try:
        i = int(((deg + 11.25) % 360) // 22.5)
    except Exception:
        i = 0
    return dirs[i]

@dataclass
class MetricsSnapshot:
    freq_mhz: float | None
    snr_db: float | None
    bearing_deg: float | None
    centroid_lat: float | None
    centroid_lon: float | None

class MetricsAdapter:
    """Collects recent rows, computes SNR EMA and bearing → centroid."""
    def __init__(self, window_secs: int = 120, min_snr: float = 6.0):
        self.window_secs = int(window_secs)
        self.min_snr = float(min_snr)
        self._ring = deque(maxlen=5000)  # (ts, lat, lon, snr)
        self._ema_snr = None
        self._ema_bearing_xy = None

    def ingest_row(self, row: dict):
        lat = to_float(row.get("lat") or row.get("latitude"))
        lon = to_float(row.get("lon") or row.get("lng") or row.get("longitude"))
        snr = to_float(row.get("snr") or row.get("snr_db") or row.get("snr(dB)") or row.get("snr_dbfs"))
        if (lat is not None) and (lon is not None) and (snr is not None):
            self._ring.append((time.monotonic(), lat, lon, snr))
        if snr is not None:
            self._ema_snr = snr if self._ema_snr is None else (0.35 * snr + 0.65 * self._ema_snr)

    def snapshot(self, current_fix: tuple[float, float] | None, freq_hz: float | None) -> MetricsSnapshot:
        # purge old
        cutoff = time.monotonic() - self.window_secs
        while self._ring and self._ring[0][0] < cutoff:
            self._ring.popleft()

        freq_mhz = (freq_hz / 1e6) if (freq_hz is not None) else None
        snr_card = None if self._ema_snr is None else round(self._ema_snr, 1)

        if not current_fix or not self._ring:
            return MetricsSnapshot(freq_mhz, snr_card, None, None, None)

        pts = [(lat, lon, snr) for (_, lat, lon, snr) in self._ring if snr >= self.min_snr]
        if len(pts) < 3:
            return MetricsSnapshot(freq_mhz, snr_card, None, None, None)

        c_lat, c_lon = self._weighted_geo_centroid(pts)
        bearing = self._initial_bearing(current_fix[0], current_fix[1], c_lat, c_lon)

        # EMA on unit circle
        import math
        x, y = math.cos(math.radians(bearing)), math.sin(math.radians(bearing))
        if self._ema_bearing_xy is None:
            self._ema_bearing_xy = (x, y)
        else:
            ax, ay = self._ema_bearing_xy
            self._ema_bearing_xy = (0.4 * x + 0.6 * ax, 0.4 * y + 0.6 * ay)
        ax, ay = self._ema_bearing_xy
        smoothed_bearing = (math.degrees(math.atan2(ay, ax)) + 360.0) % 360.0

        return MetricsSnapshot(freq_mhz, snr_card, round(smoothed_bearing), c_lat, c_lon)

    @staticmethod
    def _weighted_geo_centroid(pts):
        import math
        x = y = z = w_sum = 0.0
        for lat, lon, snr in pts:
            w = 1.0 + (snr / 20.0)
            latr = math.radians(lat); lonr = math.radians(lon)
            x += w * math.cos(latr) * math.cos(lonr)
            y += w * math.cos(latr) * math.sin(lonr)
            z += w * math.sin(latr)
            w_sum += w
        x /= w_sum; y /= w_sum; z /= w_sum
        lon = math.degrees(math.atan2(y, x))
        hyp = math.sqrt(x * x + y * y)
        lat = math.degrees(math.atan2(z, hyp))
        return lat, lon

    @staticmethod
    def _initial_bearing(lat1, lon1, lat2, lon2) -> float:
        import math
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        Δλ = math.radians(lon2 - lon1)
        y = math.sin(Δλ) * math.cos(φ2)
        x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
        θ = math.degrees(math.atan2(y, x))
        return (θ + 360.0) % 360.0

# ---------- Spectrum tab (embedded widget) ----------
def _read_json_robust(path, _max_tries=8, _sleep=0.03):
    """Windows-safe reader: tolerate PermissionError and partial writes."""
    last_exc = None
    for _ in range(_max_tries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if not txt:
                    time.sleep(_sleep); continue
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    last = txt.rfind("}")
                    if last != -1:
                        return json.loads(txt[:last+1])
                    raise
        except (PermissionError, OSError, json.JSONDecodeError) as e:
            last_exc = e
            time.sleep(_sleep)
    if last_exc:
        raise last_exc
    return {}

class SpectrumWidget(QtWidgets.QWidget):
    """Embeddable spectrum viewer widget with plot + waterfall and auto-discovery of live_spectrum.json."""
    def __init__(self, outdir_getter=None, parent=None):
        super().__init__(parent)
        self._get_outdir = outdir_getter  # callable returning Path or str
        self._wf_buf = None
        self._nf_ema = None
        self._spec_mtime = None
        self._autoload_seen = False
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        self.pathEdit = QtWidgets.QLineEdit()
        self.browseBtn = QtWidgets.QPushButton("Browse…")
        self.browseBtn.clicked.connect(self._choose_folder)
        row.addWidget(QtWidgets.QLabel("Trip folder:"))
        row.addWidget(self.pathEdit, 1)
        row.addWidget(self.browseBtn)
        lay.addLayout(row)

        if PG_OK:
            self.plot = pg.PlotWidget(title="Spectrum (PSD dB)")
            self.plot.setLabel('bottom', 'Frequency', units='MHz')
            self.curve = self.plot.plot([], [], pen=pg.mkPen(width=2))
            self.noise = self.plot.plot([], [], pen=pg.mkPen(style=QtCore.Qt.DashLine))
            lay.addWidget(self.plot, 2)

            self.waterfall = pg.ImageView(view=pg.PlotItem())
            self.waterfall.ui.roiBtn.hide(); self.waterfall.ui.menuBtn.hide()
            lay.addWidget(self.waterfall, 3)
        else:
            lab = QtWidgets.QLabel("pyqtgraph not installed. Run: pip install pyqtgraph")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lay.addWidget(lab)

        self.status = QtWidgets.QLabel("Waiting for live_spectrum.json …")
        self.status.setStyleSheet("color: #888;")
        lay.addWidget(self.status)

    # ---------- Discovery helpers ----------
    def _choose_folder(self):
        start = None
        try:
            if callable(self._get_outdir):
                start = str(Path(self._get_outdir() or '').expanduser())
        except Exception:
            start = None
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose trip folder", start or "")
        if d:
            self.pathEdit.setText(d)
            self._autoload_seen = False
            self._spec_mtime = None
            self.status.setText("Waiting for live_spectrum.json …")

    def _candidate_roots(self):
        roots = []
        # include current Out Dir from main window if available
        try:
            if callable(self._get_outdir):
                r = Path(str(self._get_outdir())).expanduser()
                if r.exists():
                    roots.append(r)
                if (r / "logs").exists():
                    roots.insert(0, r / "logs")
        except Exception:
            pass

        try:
            s = QtCore.QSettings("DF360", "GUI")
            last = s.value("last_root", type=str)
            if last:
                roots.append(Path(last))
        except Exception:
            pass

        roots += [
            Path.cwd() / "logs",
            Path.home() / "Documents/DF360/logs",
            Path.home() / "DF360/logs",
        ]
        out, seen = [], set()
        for r in roots:
            try:
                rp = r.resolve()
            except Exception:
                continue
            if rp in seen:
                continue
            seen.add(rp)
            if r.exists():
                out.append(r)
        return out

    def _find_latest_spec(self):
        latest_path, latest_mtime = None, -1
        for root in self._candidate_roots():
            try:
                for p in root.rglob("live_spectrum.json"):
                    try:
                        m = p.stat().st_mtime
                        if m > latest_mtime:
                            latest_mtime, latest_path = m, p
                    except FileNotFoundError:
                        continue
            except Exception:
                continue
        if latest_path:
            try:
                QtCore.QSettings("DF360", "GUI").setValue("last_root", str(latest_path.parent))
            except Exception:
                pass
            return str(latest_path)
        return None

    def _spec_path(self):
        # 1) If user chose a folder, prefer that location
        p = self.pathEdit.text().strip()
        if p:
            sp = Path(p) / "live_spectrum.json"
            if sp.exists():
                return str(sp)
            # If folder chosen but file not there yet, allow autoload once
            if not self._autoload_seen:
                found = self._find_latest_spec()
                if found:
                    self._autoload_seen = True
                    return found
            return None
        # 2) Otherwise, try auto-discovery in common roots
        return self._find_latest_spec()

    # ---------- Timer tick ----------
    def _tick(self):
        if not PG_OK:
            return
        spec_path = self._spec_path()
        if not spec_path:
            self.status.setText("Waiting for live_spectrum.json …")
            return
        try:
            mtime = os.path.getmtime(spec_path)
            if self._spec_mtime == mtime:
                return
            self._spec_mtime = mtime
            js = _read_json_robust(spec_path)
            import numpy as np
            f = np.array(js.get("freqs_mhz") or [], dtype=float)
            y = np.array(js.get("psd_db")    or [], dtype=float)
            if f.size and y.size and f.size == y.size:
                # spectrum line
                self.curve.setData(f, y)
                # EMA baseline
                if self._nf_ema is None or self._nf_ema.shape != y.shape:
                    self._nf_ema = y.copy()
                alpha = 0.9
                self._nf_ema = alpha * self._nf_ema + (1-alpha) * y
                self.noise.setData(f, self._nf_ema)

                # waterfall (first frame with autolevels; then fixed to reduce flicker)
                H = 150
                if self._wf_buf is None or self._wf_buf.shape[1] != y.size:
                    self._wf_buf = np.tile(y, (H,1))
                    self.waterfall.setImage(self._wf_buf, autoLevels=True, autoRange=False)
                else:
                    self._wf_buf = np.roll(self._wf_buf, -1, axis=0)
                    self._wf_buf[-1,:] = y
                    self.waterfall.setImage(self._wf_buf, autoLevels=False, autoRange=False)

                self.status.setText(f"Updated • {Path(spec_path).parent.name}")
            else:
                self.status.setText("Waiting for data in live_spectrum.json …")
        except Exception:
            # ignore transient read/parse/shape issues
            pass

# ---------- Main Window ----------
class BearingDial(QtWidgets.QWidget):
    """Minimal bearing dial for in-vehicle view."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._bearing = None  # degrees 0..360 or None
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.setToolTip("Bearing to centroid")

    def setBearing(self, deg):
        try:
            if deg is None:
                self._bearing = None
            else:
                d = float(deg) % 360.0
                self._bearing = d
        except Exception:
            self._bearing = None
        self.update()

    def paintEvent(self, ev):
        import math
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)

        w = self.width(); h = self.height()
        d = min(w, h) - 12
        cx = w//2; cy = h//2
        rect = QtCore.QRectF(cx - d/2, cy - d/2, d, d)

        # Colors
        fg = self.palette().windowText().color()
        muted = QtGui.QColor(fg); muted.setAlpha(120)

        # Outer circle
        pen = QtGui.QPen(muted if self._bearing is None else fg, 2)
        qp.setPen(pen)
        qp.setBrush(QtCore.Qt.NoBrush)
        qp.drawEllipse(rect)

        # Cardinal ticks
        qp.save(); qp.translate(cx, cy)
        for i in range(0, 360, 30):
            angle = math.radians(i)
            inner = d*0.42 if i % 90 == 0 else d*0.46
            x1 = math.cos(angle) * inner; y1 = math.sin(angle) * inner
            x2 = math.cos(angle) * (d*0.5); y2 = math.sin(angle) * (d*0.5)
            qp.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))
        qp.restore()

        # North marker
        qp.save(); qp.translate(cx, cy - d*0.5 + 6)
        north_brush = QtGui.QBrush(QtGui.QColor(fg))
        qp.setBrush(north_brush if self._bearing is not None else QtGui.QBrush(muted))
        qp.setPen(QtCore.Qt.NoPen)
        tri = [QtCore.QPointF(0, 0), QtCore.QPointF(-6, 10), QtCore.QPointF(6, 10)]
        qp.drawPolygon(QtGui.QPolygonF(tri))
        qp.restore()

        # Bearing arrow
        if self._bearing is not None:
            qp.save(); qp.translate(cx, cy); qp.rotate(self._bearing - 90)
            arrow_pen = QtGui.QPen(fg, 3, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            qp.setPen(arrow_pen)
            qp.drawLine(QtCore.QPointF(-d*0.10, 0), QtCore.QPointF(d*0.40, 0))
            path = QtGui.QPainterPath()
            path.moveTo(d*0.40, 0); path.lineTo(d*0.30, -8); path.lineTo(d*0.30, 8); path.closeSubpath()
            qp.fillPath(path, QtGui.QBrush(fg))
            qp.restore()

        # Label text
        qp.setPen(fg if self._bearing is not None else muted)
        font = qp.font(); font.setPointSizeF(max(8.0, d/10.0)); font.setBold(True)
        qp.setFont(font)
        if self._bearing is None:
            txt = "—"
        else:
            cardinal = _cardinal16(self._bearing)
            txt = f"{int(round(self._bearing))}° {cardinal}"
        qp.drawText(self.rect(), QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter, txt)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DF360 GUI")
        self.resize(1200, 820)
        self._help = HelpCache()
        _ensure_dirs()
        # Remember where the app was started (for resolving relative DF360.py)
        self._app_start_dir = Path.cwd()

        # State for charts/map
        self.current_csv: Optional[Path] = None
        self.last_tail_count = 0
        self.chart_timer = QtCore.QTimer(self)
        self.chart_timer.setInterval(1000)  # 1s
        self.chart_timer.timeout.connect(self._charts_tick)

        # Central widget & root layout
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # --- Top controls area (two columns) ---
        topBox = QtWidgets.QGroupBox("Session Controls")
        topLay = QtWidgets.QHBoxLayout(topBox)
        root.addWidget(topBox)

        # Left column (general controls)
        left = QtWidgets.QWidget()
        self.generalLayout = QtWidgets.QVBoxLayout(left)  # name expected by patches
        topLay.addWidget(left, 1)

        # Right column (SDR controls)
        right = QtWidgets.QWidget()
        self.sdrLayout = QtWidgets.QVBoxLayout(right)  # name expected by patches
        topLay.addWidget(right, 1)

        # Build general controls block
        self._build_general_controls()
        self._init_iq_and_profiles_ui()

        # Build SDR block
        self._init_sdr_ui()

        # --- Tabs ---
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        # --- Dashboard tab (first) ---
        self.dashboardTab = QtWidgets.QWidget()
        _dbLay = QtWidgets.QVBoxLayout(self.dashboardTab)
        _cards = QtWidgets.QHBoxLayout()
        self.card_freq = MetricCard("Frequency", "MHz")
        self.card_snr  = MetricCard("SNR", "dB")
        self.card_bear = MetricCard("Bearing → Centroid", "deg")
        _cards.addWidget(self.card_freq)
        _cards.addWidget(self.card_snr)
        _cards.addWidget(self.card_bear)
        _dbLay.addLayout(_cards)

        # Bearing dial widget
        self.bearingDial = BearingDial()
        _dbLay.addWidget(self.bearingDial, 0, QtCore.Qt.AlignCenter)

        # Metrics engine
        self.metrics = MetricsAdapter(window_secs=120, min_snr=6.0)

        # Place Dashboard as first tab
        self.tabs.addTab(self.dashboardTab, "Dashboard")
        self.tabs.setCurrentIndex(0)

        # --- Logs tab ---
        self.logsEdit = QtWidgets.QPlainTextEdit()
        self.logsEdit.setReadOnly(True)
        self.logsEdit.setMaximumBlockCount(5000)
        logsTab = QtWidgets.QWidget()
        logsLay = QtWidgets.QVBoxLayout(logsTab)
        logsLay.addWidget(self.logsEdit)
        self.tabs.addTab(logsTab, "Logs")

        # --- Charts tab ---
        chartsTab = QtWidgets.QWidget()
        chartsLay = QtWidgets.QVBoxLayout(chartsTab)
        if MPL_OK:
            self.chartCanvas = ChartsCanvas(chartsTab)
            chartsLay.addWidget(self.chartCanvas, 1)
            infoRow = QtWidgets.QHBoxLayout()
            self.chartStatus = QtWidgets.QLabel("Waiting for CSV…")
            self.stabilityLabel = QtWidgets.QLabel("")
            infoRow.addWidget(self.stabilityLabel)
            infoRow.addWidget(self.chartStatus)
            infoRow.addStretch(1)
            chartsLay.addLayout(infoRow)
        else:
            chartsLay.addWidget(QtWidgets.QLabel("matplotlib not installed. Install with: pip install matplotlib"))
        self.tabs.addTab(chartsTab, "Charts")

        # --- Spectrum tab (embedded) ---
        if PG_OK:
            self.spectrumTab = SpectrumWidget(self._get_out_dir)
        else:
            stub = QtWidgets.QWidget()
            L = QtWidgets.QVBoxLayout(stub)
            lab = QtWidgets.QLabel("pyqtgraph not installed. Run: pip install pyqtgraph")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            L.addWidget(lab)
            self.spectrumTab = stub
        self.tabs.addTab(self.spectrumTab, "Spectrum")

        # --- Map tab ---
        mapTab = QtWidgets.QWidget()
        mapLay = QtWidgets.QVBoxLayout(mapTab)
        ctrlRow = QtWidgets.QHBoxLayout()
        self.mapRefreshBtn = QtWidgets.QPushButton("Refresh Map")
        self.mapOpenBtn = QtWidgets.QPushButton("Open in Browser")
        ctrlRow.addWidget(self.mapRefreshBtn)
        ctrlRow.addWidget(self.mapOpenBtn)
        self.manualYagiCheck = QtWidgets.QCheckBox("Manual Yagi")
        self.manualYagiSpin = QtWidgets.QSpinBox()
        self.manualYagiSpin.setRange(0, 359)
        self.manualYagiSpin.setSuffix("°")
        self.manualYagiSpin.setValue(0)
        ctrlRow.addWidget(self.manualYagiCheck)
        ctrlRow.addWidget(self.manualYagiSpin)
        ctrlRow.addStretch(1)
        mapLay.addLayout(ctrlRow)

        self.mapStatus = QtWidgets.QLabel("Map ready.")
        mapLay.addWidget(self.mapStatus)

        if WEBENGINE_OK:
            self.webView = QtWebEngineWidgets.QWebEngineView()
            mapLay.addWidget(self.webView, 1)
        else:
            self.webView = None
            mapLay.addWidget(QtWidgets.QLabel("PyQtWebEngine not installed. Use 'Open in Browser' to view the map."), 1)

        self.tabs.addTab(mapTab, "Map")

        # Status bar
        self.status = self.statusBar()
        self.procState = QtWidgets.QLabel("Idle")
        self.status.addPermanentWidget(self.procState)

        # Process
        self.proc: QtCore.QProcess | None = None

        # Connect map buttons
        self.mapRefreshBtn.clicked.connect(self._refresh_map)
        self.mapOpenBtn.clicked.connect(self._open_map_external)

        # Style
        self.setStyleSheet(FIELD_MODE_QSS)

    # ----- UI builders -----
    def _build_general_controls(self):
        grid = QtWidgets.QGridLayout()
        self.generalLayout.addLayout(grid)

        # Row 0
        self.centerFreqEdit = QtWidgets.QLineEdit("155.25")
        self.centerFreqEdit.setToolTip("Center frequency in MHz (e.g., 155.25)")
        self.sampleRateEdit = QtWidgets.QLineEdit("2400000")
        self.sampleRateEdit.setToolTip("Sample rate in Hz (e.g., 2400000)")
        grid.addWidget(QtWidgets.QLabel("Center MHz"), 0, 0)
        grid.addWidget(self.centerFreqEdit, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Sample Rate"), 0, 2)
        grid.addWidget(self.sampleRateEdit, 0, 3)

        # Row 1
        self.gainEdit = QtWidgets.QLineEdit("auto")
        self.gainEdit.setToolTip("Gain (number in dB or 'auto')")
        self.intervalEdit = QtWidgets.QLineEdit("10")
        self.intervalEdit.setToolTip("CSV write interval (seconds)")
        grid.addWidget(QtWidgets.QLabel("Gain"), 1, 0)
        grid.addWidget(self.gainEdit, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Interval (s)"), 1, 2)
        grid.addWidget(self.intervalEdit, 1, 3)

        # Row 2
        self.ipEdit = QtWidgets.QLineEdit("192.168.1.247")
        self.ipEdit.setToolTip("GPS/telemetry source IP")
        self.portEdit = QtWidgets.QLineEdit("8080")
        self.portEdit.setToolTip("GPS/telemetry port")
        grid.addWidget(QtWidgets.QLabel("GPS IP"), 2, 0)
        grid.addWidget(self.ipEdit, 2, 1)
        grid.addWidget(QtWidgets.QLabel("GPS Port"), 2, 2)
        grid.addWidget(self.portEdit, 2, 3)

        # Row 3
        self.tripNameEdit = QtWidgets.QLineEdit("home1")
        self.tripNameEdit.setToolTip("Trip/session name used in CSV and maps")
        self.df360PathEdit = QtWidgets.QLineEdit("DF360_g4.py")
        pathBtn = QtWidgets.QPushButton("…")
        pathBtn.setToolTip("Choose the DF360.py engine script")
        pathBtn.clicked.connect(self._choose_df360_path)
        grid.addWidget(QtWidgets.QLabel("Trip Name"), 3, 0)
        grid.addWidget(self.tripNameEdit, 3, 1)
        grid.addWidget(QtWidgets.QLabel("DF360 Path"), 3, 2)
        pwrap = QtWidgets.QHBoxLayout()
        pwrap_widget = QtWidgets.QWidget()
        pwrap.addWidget(self.df360PathEdit, 1)
        pwrap.addWidget(pathBtn)
        pwrap.setContentsMargins(0, 0, 0, 0)
        pwrap_widget.setLayout(pwrap)
        grid.addWidget(pwrap_widget, 3, 3)

        # Row 4: Output dir
        self.outDirEdit = QtWidgets.QLineEdit(str(safe_logs_root()))
        outBtn = QtWidgets.QPushButton("…")
        outBtn.setToolTip("Choose output logs directory (CSV, maps)")
        grid.addWidget(QtWidgets.QLabel("Output Dir"), 4, 0)
        grid.addWidget(self.outDirEdit, 4, 1, 1, 2)
        grid.addWidget(outBtn, 4, 3)

        def pick_out_dir():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Output Directory", self.outDirEdit.text())
            if d:
                self.outDirEdit.setText(d)

        outBtn.clicked.connect(pick_out_dir)

        # Gate controls
        gateRow = QtWidgets.QHBoxLayout()
        self.gateCheck = QtWidgets.QCheckBox("Gate on TX")
        self.gateCheck.setChecked(False)
        self.gateCheck.setToolTip("Enable TX-gated logging (sit & wait)")
        self.gateThreshEdit = QtWidgets.QLineEdit("8")
        self.gateThreshEdit.setToolTip("SNR threshold (dB) to start an event")
        self.gateHangEdit = QtWidgets.QLineEdit("3")
        self.gateHangEdit.setToolTip("Hang time (seconds) to keep event open")
        gateRow.addWidget(self.gateCheck)
        gateRow.addWidget(QtWidgets.QLabel("SNR Threshold (dB)"))
        gateRow.addWidget(self.gateThreshEdit)
        gateRow.addWidget(QtWidgets.QLabel("Hang (s)"))
        gateRow.addWidget(self.gateHangEdit)
        self.generalLayout.addLayout(gateRow)

        # Start / Stop buttons
        btnRow = QtWidgets.QHBoxLayout()
        self.startBtn = QtWidgets.QPushButton("Start")
        self.stopBtn = QtWidgets.QPushButton("Stop")
        self.stopBtn.setEnabled(False)
        self.startBtn.setToolTip("Start DF360 engine with the selected settings")
        self.stopBtn.setToolTip("Stop the running DF360 engine")
        btnRow.addWidget(self.startBtn)
        btnRow.addWidget(self.stopBtn)
        self.generalLayout.addLayout(btnRow)

        self.startBtn.clicked.connect(self.start_df360)
        self.stopBtn.clicked.connect(self.stop_df360)

    def _init_iq_and_profiles_ui(self):
        # IQ row
        self.iqDirLabel = QtWidgets.QLabel("IQ directory")
        self.iqDirEdit = QtWidgets.QLineEdit(str(safe_iq_root()))
        self.iqDirBtn = QtWidgets.QPushButton("…")
        self.iqDirLabel.setToolTip("Where raw IQ/large artifacts should be written")
        self.iqDirEdit.setToolTip("Choose a folder for IQ capture")
        self.iqDirBtn.setToolTip("Pick an IQ folder (writable)")

        iqRow = QtWidgets.QHBoxLayout()
        iqRow.addWidget(self.iqDirLabel)
        iqRow.addWidget(self.iqDirEdit, 1)
        iqRow.addWidget(self.iqDirBtn)
        self.generalLayout.addLayout(iqRow)

        def pick_iq_dir():
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose IQ directory", self.iqDirEdit.text())
            if d:
                self.iqDirEdit.setText(d)

        self.iqDirBtn.clicked.connect(pick_iq_dir)

        # Profiles row
        self.profileBox = QtWidgets.QComboBox()
        self.profileSave = QtWidgets.QPushButton("Save as…")
        self.profileDel = QtWidgets.QPushButton("Delete")
        self.profileBox.setToolTip("Choose a saved preset")
        self.profileSave.setToolTip("Save current fields as a new profile")
        self.profileDel.setToolTip("Delete the selected profile")

        profRow = QtWidgets.QHBoxLayout()
        profRow.addWidget(QtWidgets.QLabel("Profile"))
        profRow.addWidget(self.profileBox, 1)
        profRow.addWidget(self.profileSave)
        profRow.addWidget(self.profileDel)
        self.generalLayout.addLayout(profRow)

        store = ProfilesStore()

        def refresh_profiles():
            self.profileBox.blockSignals(True)
            self.profileBox.clear()
            self.profileBox.addItems(store.list_names())
            self.profileBox.blockSignals(False)

        refresh_profiles()

        def apply_profile():
            name = self.profileBox.currentText().strip()
            if not name:
                return
            store.load(name).apply_to_main(self)

        self.profileBox.currentIndexChanged.connect(apply_profile)

        def save_current_as_profile():
            name, ok = QtWidgets.QInputDialog.getText(self, "Save Profile", "Profile name:")
            if not ok or not name.strip():
                return
            try:
                store.save(name.strip(), SessionProfile.from_main(self))
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Profiles", f"Failed to save: {e}")
                return
            refresh_profiles()
            self.profileBox.setCurrentText(name.strip())

        self.profileSave.clicked.connect(save_current_as_profile)

        def delete_selected_profile():
            name = self.profileBox.currentText().strip()
            if not name:
                return
            store.delete(name)
            refresh_profiles()

        self.profileDel.clicked.connect(delete_selected_profile)

    def _init_sdr_ui(self):
        # Top row
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("SDR"))
        self.sdrSourceBox = QtWidgets.QComboBox()
        self.sdrSourceBox.addItems(["Auto", "RTL-SDR (USB)", "RTL-TCP", "SoapySDR"])
        self.enumBtn = QtWidgets.QPushButton("List SDRs")
        self.testBtn = QtWidgets.QPushButton("Test")
        self.enumBtn.setToolTip("Enumerate devices (Soapy only)")
        self.testBtn.setToolTip("Connectivity test for the selected source")
        row.addWidget(self.sdrSourceBox)
        row.addStretch(1)
        row.addWidget(self.enumBtn)
        row.addWidget(self.testBtn)
        self.sdrLayout.addLayout(row)

        # RTL-TCP row
        tcpRow = QtWidgets.QHBoxLayout()
        tcpRow.addWidget(QtWidgets.QLabel("RTL-TCP"))
        self.rtlHost = QtWidgets.QLineEdit("127.0.0.1")
        self.rtlPort = QtWidgets.QLineEdit("1234")
        self.rtlHost.setPlaceholderText("host")
        self.rtlPort.setPlaceholderText("port")
        self.rtlHost.setToolTip("Hostname or IP of rtl_tcp server")
        self.rtlPort.setToolTip("Port of rtl_tcp server")
        tcpRow.addWidget(self.rtlHost)
        tcpRow.addWidget(self.rtlPort)
        tcpRow.addStretch(1)
        self.sdrLayout.addLayout(tcpRow)

        # Device list
        self.sdrList = QtWidgets.QListWidget()
        self.sdrList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sdrList.setToolTip("SoapySDR device arguments (JSON)")
        self.sdrLayout.addWidget(self.sdrList)

        # Wiring
        self.enumBtn.clicked.connect(self._on_enumerate_sdrs)
        self.testBtn.clicked.connect(self._on_test_sdr)

    # ----- DF360 launcher -----
    def _choose_df360_path(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select DF360.py", "", "Python (*.py);;All Files (*)")
        if path:
            self.df360PathEdit.setText(path)

    def _extend_launch_with_extras(self, args: List[str]) -> List[str]:
        df360_path = self.df360PathEdit.text().strip()
        chosen_iq = (self.iqDirEdit.text().strip() if hasattr(self, "iqDirEdit") else "") or str(safe_iq_root())
        chosen_out = (self.outDirEdit.text().strip() if hasattr(self, "outDirEdit") else "") or str(safe_logs_root())

        # --iq-dir support detection; fallback to env var
        if self._help.supports_flag(df360_path, "--iq-dir"):
            args += ["--iq-dir", chosen_iq]
        else:
            os.environ["DF360_IQ_DIR"] = chosen_iq

        # --outdir support detection; fallback to env var
        if self._help.supports_flag(df360_path, "--outdir"):
            args += ["--outdir", chosen_out]
        else:
            os.environ["DF360_OUTDIR"] = chosen_out

        # SDR hints (non-breaking)
        src = self.sdrSourceBox.currentText() if hasattr(self, "sdrSourceBox") else ""
        if src.startswith("RTL-TCP"):
            args += ["--rtl-tcp-host", self.rtlHost.text().strip(), "--rtl-tcp-port", self.rtlPort.text().strip()]
        elif src.startswith("Soapy"):
            if self.sdrList.currentItem():
                args += ["--soapy-args", self.sdrList.currentItem().text()]

        return args

    def start_df360(self):
        if self.proc is not None:
            QtWidgets.QMessageBox.warning(self, "DF360", "Process already running.")
            return

        # Resolve DF360.py to an absolute path so working-directory changes don't break it
        df360_raw = self.df360PathEdit.text().strip() or "DF360.py"
        try:
            pth = Path(df360_raw)
            if not pth.is_absolute():
                base = getattr(self, "_app_start_dir", Path.cwd())
                candidate = (base / df360_raw)
                if candidate.exists():
                    pth = candidate
            df360_path = str(pth.expanduser().resolve(strict=False))
        except Exception:
            df360_path = df360_raw

        args = [
            sys.executable, df360_path,
            "--noninteractive",
            "--center-mhz", self.centerFreqEdit.text(),
            "--sample-rate", self.sampleRateEdit.text(),
            "--gain", self.gainEdit.text(),
            "--interval", self.intervalEdit.text(),
            "--ip", self.ipEdit.text(),
            "--port", self.portEdit.text(),
            "--trip", self.tripNameEdit.text(),
        ]

        # Gate flags (only if supported by DF360.py)
        gate_supported = self._help.supports_flag(df360_path, "--gate-logging") and \
                         self._help.supports_flag(df360_path, "--snr-threshold-db") and \
                         self._help.supports_flag(df360_path, "--hang-s")
        if self.gateCheck.isChecked() and gate_supported:
            args += ["--gate-logging", "--snr-threshold-db", self.gateThreshEdit.text(), "--hang-s", self.gateHangEdit.text()]
        else:
            if self.gateCheck.isChecked() and not gate_supported:
                self.logsEdit.appendPlainText("Note: DF360.py does not support --gate-logging in this build; running without gating.\n")
            if self._help.supports_flag(df360_path, "--no-gate-logging"):
                args += ["--no-gate-logging"]

        # IQ dir + SDR hints + outdir
        args = self._extend_launch_with_extras(args)

        # Prepare output dir and map csv discovery
        outdir = Path(self.outDirEdit.text().strip() or safe_logs_root())
        outdir.mkdir(parents=True, exist_ok=True)

        # Suppress noisy pkg_resources warnings from some drivers
        os.environ.setdefault("PYTHONWARNINGS", "ignore:pkg_resources is deprecated as an API:UserWarning")

        # Spawn QProcess
        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(args[0])
        self.proc.setArguments(args[1:])
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Force DF360's working directory to the chosen Output Dir.
        # This guarantees that engines writing relative paths drop CSVs here.
        outdir = Path(self.outDirEdit.text().strip() or safe_logs_root())
        self.proc.setWorkingDirectory(str(outdir))
        self.logsEdit.appendPlainText(f"Working dir set to: {outdir}\n")

        self.proc.readyReadStandardOutput.connect(self._on_df360_output)
        self.proc.finished.connect(self._on_df360_finished)
        self.proc.start()

        self.procState.setText("Running")
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.logsEdit.appendPlainText(f"Launching: {' '.join(args)}\n")

        # Start charts timer
        self.chart_timer.start()

    def stop_df360(self):
        if self.proc is None:
            return
        self.proc.kill()
        self.proc = None
        self.procState.setText("Idle")
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.logsEdit.appendPlainText("Stopped.\n")
        self.chart_timer.stop()

    def _on_df360_output(self):
        if not self.proc:
            return
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.logsEdit.appendPlainText(text.rstrip())

    def _on_df360_finished(self):
        self.logsEdit.appendPlainText("Process finished.\n")
        self.proc = None
        self.procState.setText("Idle")
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.chart_timer.stop()

    # ----- Charts logic -----
    def _charts_tick(self):
        if not MPL_OK:
            return
        outdir = Path(self.outDirEdit.text().strip() or safe_logs_root())
        csv_path = find_latest_csv(outdir, self.tripNameEdit.text().strip())
        if not csv_path:
            self.chartStatus.setText("No CSV found in output dir.")
            return
        self.current_csv = csv_path
        lines = tail_lines(csv_path, 400)
        rows, idx = parse_csv_rows(lines)
        if not rows:
            self.chartStatus.setText(f"Waiting for data: {csv_path.name}")
            return

        # Column detection
        def pick(row: dict, names: List[str], default=None):
            for nm in names:
                if nm in row:
                    return row.get(nm, default)
            return default

        snrs, rssis, noises = [], [], []
        last_lat, last_lon = None, None
        for r in rows:
            snr = to_float(pick(r, ["snr", "snr_db", "snr(dB)", "snr_dbfs"]), None)
            rssi = to_float(pick(r, ["rssi","rssi_db","rssi_dbm"]), None)
            noise = to_float(pick(r, ["noise","noise_db","noise_floor_db"]), None)
            if snr is not None: snrs.append(snr)
            if rssi is not None: rssis.append(rssi)
            if noise is not None: noises.append(noise)
            lat = to_float(pick(r, ["lat", "latitude"]), None)
            lon = to_float(pick(r, ["lon", "lng", "longitude"]), None)
            if lat is not None and lon is not None:
                last_lat, last_lon = lat, lon

        self.chartCanvas.update_series(snrs, rssis, noises)
        if last_lat is not None:
            self.chartStatus.setText(f"Watching: {csv_path}  |  Last fix: {last_lat:.5f}, {last_lon:.5f}")
        else:
            self.chartStatus.setText(f"Watching: {csv_path}")

        # —— Dashboard metrics feed ——
        try:
            last = rows[-1]
            # Ingest latest row for centroid math
            self.metrics.ingest_row(last)

            # Current fix for bearing → centroid
            cur_fix = None
            if last_lat is not None and last_lon is not None:
                cur_fix = (last_lat, last_lon)

            # Frequency for dashboard
            freq_hz = None
            fm = to_float(last.get("freq_mhz"))
            if fm is not None:
                freq_hz = fm * 1e6
            else:
                ui_mhz = to_float(self.centerFreqEdit.text())
                if ui_mhz is not None:
                    freq_hz = ui_mhz * 1e6

            snap = self.metrics.snapshot(cur_fix, freq_hz)

            # Manual Yagi overrides bearing if enabled
            yagi_on = getattr(self, 'manualYagiCheck', None) and self.manualYagiCheck.isChecked()
            yagi_brg = float(self.manualYagiSpin.value()) if yagi_on else None

            # Stability + speed checks (last ~5 rows)
            speed_vals = []
            bstd_vals = []
            for rr in rows[-5:]:
                sp = to_float(rr.get('speed_kph'))
                if sp is not None: speed_vals.append(sp)
                bs = to_float(rr.get('bearing_std_deg'))
                if bs is not None: bstd_vals.append(bs)
            speed_bad = (len(speed_vals)>0 and any(s>10.0 for s in speed_vals))
            stability_bad = (len(bstd_vals)>=3 and all(b>15.0 for b in bstd_vals[-3:]))
            msg = []
            if speed_bad: msg.append('Slow for tight LOB (speed>10kph)')
            if stability_bad: msg.append('Hold steady (bearing std>15°)')
            self.stabilityLabel.setText(' | '.join(msg))

            # Update cards & dial (respect manual and gates)
            self.card_freq.set_value(snap.freq_mhz)
            self.card_snr.set_value(snap.snr_db)
            disp_bearing = None
            if yagi_on:
                disp_bearing = yagi_brg
            elif snap.bearing_deg is not None and not (speed_bad or stability_bad):
                disp_bearing = snap.bearing_deg
            if disp_bearing is not None:
                self.card_bear.set_value(disp_bearing, _cardinal16(disp_bearing))
                self.bearingDial.setBearing(disp_bearing)
            else:
                self.card_bear.set_value(None)
                self.bearingDial.setBearing(None)
        except Exception as _e:
            # Don't let dashboard issues break charts
            pass

    # ----- Map logic -----
    def _refresh_map(self):
        try:
            import folium  # lazy import
            FOLIUM = True
        except Exception:
            FOLIUM = False
        if not FOLIUM:
            self.mapStatus.setText("folium not installed. Install with: pip install folium")
            return
        outdir = Path(self.outDirEdit.text().strip() or safe_logs_root())
        csv_path = self.current_csv or find_latest_csv(outdir, self.tripNameEdit.text().strip())
        if not csv_path:
            self.mapStatus.setText("No CSV file found to map.")
            return
        lines_txt = tail_lines(csv_path, 1000)
        rows, idx = parse_csv_rows(lines_txt)
        if not rows:
            self.mapStatus.setText("CSV has no rows yet.")
            return

        # Extract last points
        def pick(row: dict, names: List[str], default=None):
            for nm in names:
                if nm in row:
                    return row.get(nm, default)
            return default

        pts = []
        for r in rows:
            lat = to_float(pick(r, ["lat","latitude"]))
            lon = to_float(pick(r, ["lon","lng","longitude"]))
            brg = to_float(pick(r, ["bearing_deg","bearing","deg","azimuth"]))
            bstd = to_float(r.get("bearing_std_deg"))
            snr = to_float(pick(r, ["snr","snr_db","snr(dB)","snr_dbfs"]))
            valid = int(pick(r, ["bearing_valid"], 1))
            if (lat is None) or (lon is None):
                continue
            pts.append((lat,lon,brg,bstd,snr,valid,r))

        if not pts:
            self.mapStatus.setText("No lat/lon points yet.")
            return

        clat, clon = pts[-1][0], pts[-1][1]
        import folium
        m = folium.Map(location=[clat,clon], zoom_start=13, control_scale=True)

        def color_for_snr(s):
            if s is None: return "gray"
            if s >= 20: return "green"
            if s >= 12: return "orange"
            return "red"

        def _wedge_points(lat0, lon0, brg, spread_deg, length_km, steps=12):
            pts = []
            try:
                import numpy as np
                angles = np.linspace(brg - spread_deg, brg + spread_deg, steps)
            except Exception:
                angles = [brg - spread_deg + (2*spread_deg)*(i/max(1,steps-1)) for i in range(steps)]
            for a in angles:
                lat2, lon2 = project_bearing(lat0, lon0, float(a), length_km)
                pts.append([lat2, lon2])
            return pts

        use_manual = getattr(self, 'manualYagiCheck', None) and self.manualYagiCheck.isChecked()
        manual_brg = float(self.manualYagiSpin.value()) if use_manual else None

        for (lat,lon,brg,bstd,snr,valid,row) in pts[-500:]:
            draw_brg = manual_brg if use_manual else brg
            if draw_brg is None:
                continue
            spread = 8.0 if use_manual else max(5.0, min(25.0, bstd if bstd is not None else 15.0))
            length_km = max(0.05, min(0.3, (snr or 0) * 0.01))
            lat2, lon2 = project_bearing(lat, lon, draw_brg, length_km)

            if valid:
                folium.PolyLine([[lat,lon],[lat2,lon2]], weight=2, color=color_for_snr(snr), opacity=0.8).add_to(m)
                poly = [[lat,lon]] + _wedge_points(lat,lon,draw_brg,spread,length_km) + [[lat,lon]]
                folium.Polygon(poly, color=color_for_snr(snr), fill=True, fill_opacity=0.15, weight=1).add_to(m)
            else:
                folium.PolyLine([[lat,lon],[lat2,lon2]], weight=1, color="gray", opacity=0.4).add_to(m)

        # Triangulated hypothesis marker
        t_lat = None; t_lon = None
        for r in rows[::-1]:
            tlat = to_float(r.get('triangulated_lat'))
            tlon = to_float(r.get('triangulated_lon'))
            if (tlat is not None) and (tlon is not None):
                t_lat, t_lon = tlat, tlon
                break
        if (t_lat is not None) and (t_lon is not None):
            folium.CircleMarker([t_lat, t_lon], radius=7, color='red', fill=True, fill_opacity=0.9, popup='Target Hypothesis').add_to(m)

        outdir.mkdir(parents=True, exist_ok=True)
        map_path = outdir / f"map_{self.tripNameEdit.text().strip() or 'session'}.html"
        m.save(str(map_path))
        self.mapStatus.setText(f"Map generated: {map_path} (from {csv_path})")
        if self.webView is not None:
            self.webView.setUrl(QtCore.QUrl.fromLocalFile(str(map_path)))
        self._last_map_path = map_path

    def _open_map_external(self):
        p = getattr(self, "_last_map_path", None)
        if not p:
            # try to generate quickly
            self._refresh_map()
            p = getattr(self, "_last_map_path", None)
        if p and Path(p).exists():
            webbrowser.open(str(p))

    # ----- SDR handlers -----
    def _on_enumerate_sdrs(self):
        self.sdrList.clear()
        if self.sdrSourceBox.currentText().startswith("Soapy"):
            res = DeviceTools.enumerate_soapy()
            if isinstance(res, str):
                self.sdrList.addItem(res)
            else:
                for r in res:
                    self.sdrList.addItem(json.dumps(r, separators=(",", ":")))
        else:
            self.sdrList.addItem("Enumeration not required for this source.")

    def _on_test_sdr(self):
        src = self.sdrSourceBox.currentText()
        if src.startswith("RTL-TCP"):
            ok, msg = DeviceTools.test_rtl_tcp(self.rtlHost.text().strip(), int(self.rtlPort.text()))
            QtWidgets.QMessageBox.information(self, "RTL-TCP" if ok else "RTL-TCP Error", msg)
        elif src.startswith("Soapy"):
            item = self.sdrList.currentItem()
            if not item:
                QtWidgets.QMessageBox.warning(self, "SoapySDR", "Select a device from the list first.")
                return
            try:
                dev_args = json.loads(item.text())
            except Exception:
                QtWidgets.QMessageBox.warning(self, "SoapySDR", "Invalid device args (JSON).")
                return
            ok, msg = DeviceTools.test_soapy_device(dev_args)
            QtWidgets.QMessageBox.information(self, "SoapySDR" if ok else "SoapySDR Error", msg)
        else:
            QtWidgets.QMessageBox.information(self, "SDR", "For RTL-SDR (USB), plug the dongle and press Start; DF360 opens it.")

    # ----- Helpers -----
    def _get_out_dir(self):
        try:
            return Path(self.outDirEdit.text().strip() or safe_logs_root())
        except Exception:
            return safe_logs_root()


# ---------- Main ----------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("DF360")
    app.setApplicationName("DF360")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
