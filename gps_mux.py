
# gps_mux.py — GPS multiplexer with TCP/UDP/Serial/gpsd + manual fallback
# Fixes in this build:
#  • Split on real newlines ('\n' / b'\n') and actually parse each line.
#  • Initialize debug ring BEFORE reading cache to avoid first-run errors.
#  • Robust cache loader: maps legacy fields (e.g., speed_kph) and ignores unknowns.
#  • Sane rotation: if stuck on MANUAL, periodically re-probe other providers (manual_retry_s).
#
# Public API:
#   mux = GPSMux(cfg_dict)
#   fix, state, age = mux.get_last_fix()
#
# Config keys (all optional):
#   - source: 'auto' (default) or one of: 'tcp','udp','serial','gpsd','manual'
#   - order: list of sources to try when source='auto'
#   - prefer_talkers: tuple like ('GN','GP','GL','GA')
#   - tcp_host, tcp_port (defaults: 127.0.0.1, 8080)
#   - udp_port (default: 10110)
#   - ser_port, ser_baud (defaults: 'COM3', 9600)
#   - gpsd_host, gpsd_port (defaults: 127.0.0.1, 2947)
#   - manual_lat, manual_lon, manual_alt_m
#   - last_fix_cache (default: 'df_last_fix.json')
#   - fix_timeout_s (default: 15), stale_age_s (default: 30)
#   - manual_retry_s (default: 20)  # how often to leave MANUAL and re-probe others
#   - min_hdop (float), manual_fallback (bool), allow_no_fix (bool)
#
from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Deque, List, Optional, Tuple
from collections import deque

# Optional deps
try:
    import serial  # pyserial
except Exception:  # pragma: no cover
    serial = None

try:
    import pynmea2
except Exception:  # pragma: no cover
    pynmea2 = None


def _utc_now_s() -> float:
    return time.time()


@dataclass
class Fix:
    lat: float
    lon: float
    alt_m: float = 0.0
    hdop: float = 0.0
    sats: int = 0
    talker: str = ""
    fix_type: str = "NONE"  # NONE | 2D | 3D | MANUAL
    ts: float = 0.0
    speed_mps: float = 0.0
    course_deg: float = 0.0


# -----------------------------
# NMEA parsing helpers
# -----------------------------

def _parse_nmea(line: str, last: Optional[Fix]) -> Optional[Fix]:
    """Parse an NMEA sentence. Prefer pynmea2; fallback to tiny GGA parser."""
    now = _utc_now_s()
    line = line.strip()
    if not line or not line.startswith("$"):
        return None

    talker = ""
    if len(line) > 6 and line[1:3].isalpha():
        talker = line[1:3]

    if pynmea2 is not None:
        try:
            msg = pynmea2.parse(line, check=True)
            tag = getattr(msg, "sentence_type", "").upper()

            if tag == "GGA":
                lat = float(msg.latitude) if getattr(msg, "latitude", "") else (last.lat if last else 0.0)
                lon = float(msg.longitude) if getattr(msg, "longitude", "") else (last.lon if last else 0.0)
                alt = float(getattr(msg, "altitude", 0.0) or 0.0)
                hd  = float(getattr(msg, "horizontal_dil", 0.0) or 0.0)
                sats = int(getattr(msg, "num_sats", 0) or 0)
                q = getattr(msg, "gps_qual", 0) or 0  # 0=no,1=GPS,2=DGPS,4/5 RTK
                typ = "3D" if q in (2,4,5) else ("2D" if q == 1 else "NONE")
                return Fix(lat, lon, alt_m=alt, hdop=hd, sats=sats, talker=talker, fix_type=typ, ts=now)

            if tag == "RMC":
                lat = float(msg.latitude) if getattr(msg, "latitude", "") else (last.lat if last else 0.0)
                lon = float(msg.longitude) if getattr(msg, "longitude", "") else (last.lon if last else 0.0)
                sp_kts = float(getattr(msg, "spd_over_grnd", 0.0) or 0.0)
                crs = float(getattr(msg, "true_course", 0.0) or 0.0)
                typ = "2D" if (lat or lon) else "NONE"
                return Fix(lat, lon, speed_mps=sp_kts * 0.514444, course_deg=crs, talker=talker, fix_type=typ, ts=now)
        except Exception:
            pass  # fall through

    # Minimal GGA fallback
    try:
        if line[3:6] == "GGA":
            parts = line.split(",")
            lat_raw, ns, lon_raw, ew = parts[2], parts[3], parts[4], parts[5]
            lat = _nmea_to_deg(lat_raw, ns) if lat_raw else (last.lat if last else 0.0)
            lon = _nmea_to_deg(lon_raw, ew) if lon_raw else (last.lon if last else 0.0)
            alt = float(parts[9]) if len(parts) > 9 and parts[9] else 0.0
            hd  = float(parts[8]) if len(parts) > 8 and parts[8] else 0.0
            sats = int(parts[7]) if len(parts) > 7 and parts[7] else 0
            q   = int(parts[6]) if len(parts) > 6 and parts[6] else 0
            typ = "3D" if q in (2,4,5) else ("2D" if q == 1 else "NONE")
            return Fix(lat, lon, alt_m=alt, hdop=hd, sats=sats, talker=talker, fix_type=typ, ts=now)
    except Exception:
        return None

    return None


def _nmea_to_deg(raw: str, hemi: str) -> float:
    if not raw or "." not in raw:
        return 0.0
    i = raw.index(".")
    head = raw[:i]
    deg = float(head[:-2]) if len(head) > 2 else 0.0
    mins = float(head[-2:] + raw[i:])
    val = deg + mins / 60.0
    if hemi in ("S","W"):
        val = -val
    return val


# -----------------------------
# Provider base + NMEA stream mixin
# -----------------------------

class _BaseProvider:
    def __init__(self, debug_cb: Optional[Callable[[str], None]] = None):
        self._debug = debug_cb or (lambda s: None)
        self._th: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._last: Optional[Fix] = None

    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop_evt.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def is_alive(self) -> bool:
        return bool(self._th and self._th.is_alive())

    def stop(self):
        self._stop_evt.set()
        if self._th:
            self._th.join(timeout=1.0)

    def last(self) -> Optional[Fix]:
        with self._lock:
            return self._last

    def _set(self, fx: Fix):
        with self._lock:
            self._last = fx

    def _loop(self):
        raise NotImplementedError


class _NMEAStreamProvider(_BaseProvider):
    def __init__(self, prefer_talkers=("GN","GP","GL","GA"), debug_cb=None):
        super().__init__(debug_cb=debug_cb)
        self.pref = tuple(prefer_talkers)

    def _on_line(self, line: str):
        self._debug(line)
        fx = _parse_nmea(line, self.last())
        if fx and (not self.pref or fx.talker in self.pref):
            self._set(fx)


# -----------------------------
# Concrete providers
# -----------------------------

class _TcpProvider(_NMEAStreamProvider):
    def __init__(self, host: str, port: int, prefer_talkers, debug_cb):
        super().__init__(prefer_talkers=prefer_talkers, debug_cb=debug_cb)
        self.host, self.port = host, port

    def _loop(self):
        while not self._stop_evt.is_set():
            s = None
            try:
                s = socket.create_connection((self.host, self.port), timeout=3.0)
                s.settimeout(1.0)
                self._debug(f"[tcp] connected {self.host}:{self.port}")
                buf = ""
                while not self._stop_evt.is_set():
                    try:
                        chunk = s.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        time.sleep(0.1)
                        continue
                    buf += chunk.decode("ascii", "ignore")
                    while "\n" in buf:
                        # split on real newline, parse each line
                        line, buf = buf.split("\n", 1)
                        line = line.rstrip("\r")
                        if line:
                            self._on_line(line)
            except Exception as e:
                self._debug(f"[tcp] reconnect: {e!s}")
                time.sleep(0.7)
            finally:
                try:
                    if s:
                        s.close()
                except Exception:
                    pass


class _UdpProvider(_NMEAStreamProvider):
    def __init__(self, port: int, prefer_talkers, debug_cb):
        super().__init__(prefer_talkers=prefer_talkers, debug_cb=debug_cb)
        self.port = port

    def _loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", self.port))
            s.settimeout(1.0)
            self._debug(f"[udp] listening 0.0.0.0:{self.port}")
            buf = ""
            while not self._stop_evt.is_set():
                try:
                    data, _ = s.recvfrom(4096)
                except socket.timeout:
                    continue
                if not data:
                    continue
                buf += data.decode("ascii", "ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if line:
                        self._on_line(line)
        finally:
            try:
                s.close()
            except Exception:
                pass


class _SerialProvider(_NMEAStreamProvider):
    def __init__(self, ser_port: str, ser_baud: int, prefer_talkers, debug_cb):
        super().__init__(prefer_talkers=prefer_talkers, debug_cb=debug_cb)
        self.ser_port, self.ser_baud = ser_port, ser_baud

    def _loop(self):
        if serial is None:
            self._debug("[serial] pyserial not installed")
            return
        while not self._stop_evt.is_set():
            try:
                ser = serial.Serial(self.ser_port, self.ser_baud, timeout=1)
                self._debug(f"[serial] open {self.ser_port}@{self.ser_baud}")
                buf = ""
                while not self._stop_evt.is_set():
                    data = ser.read(512)
                    if not data:
                        continue
                    buf += data.decode("ascii", "ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.rstrip("\r")
                        if line:
                            self._on_line(line)
            except Exception as e:
                self._debug(f"[serial] reconnect: {e!s}")
                time.sleep(0.7)


class _GpsdProvider(_BaseProvider):
    def __init__(self, host: str, port: int, debug_cb):
        super().__init__(debug_cb=debug_cb)
        self.host, self.port = host, port

    def _loop(self):
        while not self._stop_evt.is_set():
            s = None
            try:
                s = socket.create_connection((self.host, self.port), timeout=3.0)
                s.settimeout(1.0)
                self._debug(f"[gpsd] connected {self.host}:{self.port}")
                s.sendall(b'?WATCH={"enable":true,"json":true}\n')
                buf = b""
                while not self._stop_evt.is_set():
                    try:
                        chunk = s.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line.decode("utf-8", "ignore"))
                        except Exception:
                            continue
                        if obj.get("class") == "TPV":
                            lat = obj.get("lat"); lon = obj.get("lon")
                            if lat is None or lon is None:
                                continue
                            alt = float(obj.get("alt", 0.0) or 0.0)
                            spd = float(obj.get("speed", 0.0) or 0.0)
                            crs = float(obj.get("track", 0.0) or 0.0)
                            fx = Fix(float(lat), float(lon), alt_m=alt, speed_mps=spd, course_deg=crs,
                                     talker="GP", fix_type="2D", ts=_utc_now_s())
                            self._set(fx)
            except Exception as e:
                self._debug(f"[gpsd] reconnect: {e!s}")
                time.sleep(0.7)
            finally:
                try:
                    if s:
                        s.close()
                except Exception:
                    pass


# -----------------------------
# GPS multiplexer
# -----------------------------

class GPSMux:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = "NO_SOURCE"
        self._prov: Optional[_BaseProvider] = None
        self.provider_name = "None"

        # Debug ring FIRST so cache errors can be logged safely
        self._debug_lines: Deque[str] = deque(maxlen=300)

        self.cache_path = cfg.get("last_fix_cache", "df_last_fix.json")
        self._cached: Optional[Fix] = self._load_cache()
        self._last_cache_write = 0.0

        self._fix_timeout_s = float(cfg.get("fix_timeout_s", 15.0))
        self._stale_age_s = float(cfg.get("stale_age_s", 30.0))
        self._manual_retry_s = float(cfg.get("manual_retry_s", 20.0))
        self._min_hdop = float(cfg.get("min_hdop", 0.0) or 0.0)
        self._allow_no_fix = bool(cfg.get("allow_no_fix", True))
        self._manual_fallback = bool(cfg.get("manual_fallback", True))

        # Provider order
        self._order = (
            cfg.get("order") or ["tcp","udp","gpsd","serial","manual"]
        ) if (cfg.get("source", "auto") == "auto") else [cfg.get("source")]
        self._prefer_talkers = tuple(cfg.get("prefer_talkers", ("GN","GP","GL","GA")))

        self._stack: List[Tuple[str, Callable[[], _BaseProvider]]] = []
        self._build_stack()
        self._idx = -1
        self._prov_started_s = 0.0
        self._choose_and_start()

    # ---- Debug helpers
    def _push_debug(self, s: str):
        self._debug_lines.appendleft(f"{time.strftime('%H:%M:%S')} {s}")

    def get_debug_lines(self, n: int = 50) -> List[str]:
        return list(list(self._debug_lines)[:n])

    # ---- Build providers stack based on cfg
    def _build_stack(self):
        def add(name: str, factory: Callable[[], _BaseProvider]):
            self._stack.append((name, factory))

        for name in self._order:
            n = (name or "").lower()
            if n == "tcp":
                add("tcp", lambda: _TcpProvider(self.cfg.get("tcp_host","127.0.0.1"),
                                                int(self.cfg.get("tcp_port",8080)),
                                                self._prefer_talkers, self._push_debug))
            elif n == "udp":
                add("udp", lambda: _UdpProvider(int(self.cfg.get("udp_port",10110)),
                                                self._prefer_talkers, self._push_debug))
            elif n == "serial":
                add("serial", lambda: _SerialProvider(self.cfg.get("ser_port","COM3"),
                                                      int(self.cfg.get("ser_baud",9600)),
                                                      self._prefer_talkers, self._push_debug))
            elif n == "gpsd":
                add("gpsd", lambda: _GpsdProvider(self.cfg.get("gpsd_host","127.0.0.1"),
                                                  int(self.cfg.get("gpsd_port",2947)),
                                                  self._push_debug))
            elif n == "manual":
                add("manual", lambda: _ManualProvider(float(self.cfg.get("manual_lat",0.0)),
                                                      float(self.cfg.get("manual_lon",0.0)),
                                                      float(self.cfg.get("manual_alt_m",0.0)),
                                                      self._push_debug))

    def _choose_and_start(self):
        if not self._stack:
            self.state = "NO_SOURCE"
            return
        try:
            if self._prov:
                self._prov.stop()
        except Exception:
            pass

        self._idx = (self._idx + 1) % len(self._stack)
        name, factory = self._stack[self._idx]
        try:
            self._prov = factory()
            self.provider_name = type(self._prov).__name__
            self._prov_started_s = _utc_now_s()
            self._push_debug(f"[mux] starting {self.provider_name}")
            self._prov.start()
            self.state = "CONNECTING"
        except Exception as e:
            self._push_debug(f"[mux] provider start failed: {e!s}")
            self._prov = None
            self.state = "NO_SOURCE"

    def _rotate_provider(self):
        self._push_debug("[mux] rotating provider")
        self._choose_and_start()

    # ---- Cache
    def _load_cache(self) -> Optional[Fix]:
        try:
            if not os.path.exists(self.cache_path):
                return None
            with open(self.cache_path, "r", encoding="utf-8") as f:
                d = json.load(f)

            out = {}
            for k, v in (d or {}).items():
                if k == "speed_kph":
                    out["speed_mps"] = float(v) / 3.6
                elif k in ("speed_knots","spd_knots"):
                    out["speed_mps"] = float(v) * 0.514444
                elif k == "speed_mph":
                    out["speed_mps"] = float(v) * 0.44704
                elif k in ("speed","speed_mps"):
                    out["speed_mps"] = float(v)
                elif k in ("course","heading","course_deg"):
                    out["course_deg"] = float(v)
                elif k == "ts" and isinstance(v, str):
                    # ISO string from older builds
                    try:
                        out["ts"] = datetime.fromisoformat(v).timestamp()
                    except Exception:
                        continue
                else:
                    out[k] = v

            valid = set(Fix.__dataclass_fields__.keys())
            filtered = {k: out[k] for k in out.keys() if k in valid}
            return Fix(**filtered)
        except Exception as e:
            try:
                self._push_debug(f"[cache] load failed: {e!s}")
            except Exception:
                pass
            return None

    def _save_cache(self, fx: Fix, now: float):
        try:
            if now - getattr(self, "_last_cache_write", 0.0) < 2.0:
                return
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(asdict(fx), f)
            self._last_cache_write = now
        except Exception as e:
            self._push_debug(f"[cache] save failed: {e!s}")

    # ---- Public API
    def get_last_fix(self):
        now = _utc_now_s()
        fx = self._prov.last() if self._prov else None

        # HDOP filter
        if fx and self._min_hdop > 0.0 and fx.hdop and fx.hdop > self._min_hdop:
            fx = None

        # Rotate only if truly stale or provider dead, OR if we're on MANUAL too long
        if (
            (self._prov and not self._prov.is_alive())
            or (fx and (now - fx.ts) > self._fix_timeout_s)
            or (fx is None and (now - self._prov_started_s) > self._fix_timeout_s)
            or (fx and fx.fix_type == "MANUAL" and (now - self._prov_started_s) > self._manual_retry_s)
        ):
            self._rotate_provider()
            fx = self._prov.last() if self._prov else None

        if fx:
            age = max(0.0, now - fx.ts)
            if fx.fix_type == "MANUAL":
                self.state = "MANUAL"
            elif fx.fix_type == "3D":
                self.state = "FIX_3D"
            elif fx.fix_type == "2D":
                self.state = "FIX_2D"
            else:
                self.state = "NO_FIX"

            if fx.fix_type in ("2D","3D"):
                self._save_cache(fx, now)

            if age > self._stale_age_s:
                self.state = "STALE"

            return fx, self.state, age

        # No provider fix — return cached if not too old
        if self._cached:
            age = max(0.0, now - self._cached.ts)
            if age <= self._stale_age_s:
                self.state = "STALE"
                return self._cached, self.state, age

        # Manual fallback
        if self._manual_fallback and ("manual" in [n for n, _ in self._stack]):
            lat = float(self.cfg.get("manual_lat", 0.0))
            lon = float(self.cfg.get("manual_lon", 0.0))
            mf = Fix(lat, lon, fix_type="MANUAL", ts=now)
            self.state = "MANUAL"
            return mf, self.state, 0.0

        self.state = "NO_FIX"
        return None, self.state, 1e9


# -----------------------------
# Helper: Manual provider (used in stack)
# -----------------------------

class _ManualProvider(_BaseProvider):
    def __init__(self, lat: float, lon: float, alt_m: float, debug_cb):
        super().__init__(debug_cb=debug_cb)
        self.fx = Fix(lat, lon, alt_m=alt_m, fix_type="MANUAL", ts=_utc_now_s())

    def _loop(self):
        while not self._stop_evt.is_set():
            self.fx.ts = _utc_now_s()
            self._set(self.fx)
            time.sleep(1.0)
