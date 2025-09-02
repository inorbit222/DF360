# sdr_backend.py — Multi-SDR abstraction for DF360/DF3
# Supports: RTL-SDR (pyrtlsdr), RTL-TCP (RtlSdrTcpClient), and SoapySDR (Airspy, HackRF, Lime, SDRplay, ...)
# Drop this file next to DF360.py, then: from sdr_backend import SDRBackend, SDRInitError

from __future__ import annotations
import numpy as np

class SDRInitError(RuntimeError):
    pass

class SDRBackend:
    """Thin wrapper so the app doesn't care which SDR is underneath."""
    def __init__(self, kind: str = "auto", device: str | None = None):
        self.kind = (kind or "auto").lower()
        self.device_str = device or ""
        self.dev = None
        self._soapy = None
        self._soapy_dir = None
        self._soapy_stream = None
        self._fs = None

    # ---------- Discovery / opening ----------
    def open(self):
        if self.kind == "auto":
            # Try RTL first, then Soapy
            try:
                return self._open_rtl()
            except Exception:
                pass
            try:
                return self._open_soapy()
            except Exception as e:
                raise SDRInitError(f"No supported SDR found (RTL and Soapy failed): {e!r}")
        elif self.kind == "rtl":
            return self._open_rtl()
        elif self.kind in ("rtltcp", "rtl_tcp", "rtl-tcp"):
            return self._open_rtltcp()
        elif self.kind == "soapy":
            return self._open_soapy()
        else:
            raise SDRInitError(f"Unknown SDR kind: {self.kind}")

    def _open_rtl(self):
        try:
            from rtlsdr import RtlSdr
        except Exception as e:
            raise SDRInitError(f"pyrtlsdr not available: {e}")
        self.dev = RtlSdr()
        return self

    def _open_rtltcp(self):
        try:
            from rtlsdr import RtlSdrTcpClient
        except Exception as e:
            raise SDRInitError(f"RTL-TCP backend requires pyrtlsdr with RtlSdrTcpClient: {e}")
        # device_str can be "host:port"
        host, port = "127.0.0.1", 1234
        if self.device_str and ":" in self.device_str:
            host, port_s = self.device_str.split(":", 1)
            try:
                port = int(port_s)
            except Exception:
                pass
        self.dev = RtlSdrTcpClient(hostname=host, port=port)
        return self

    def _open_soapy(self):
        try:
            import SoapySDR  # type: ignore
            from SoapySDR import SOAPY_SDR_RX
        except Exception as e:
            raise SDRInitError(f"SoapySDR not available: {e}")
        args = self.device_str or ""
        try:
            self._soapy = SoapySDR.Device(args)
        except Exception as e:
            raise SDRInitError(f"Could not open Soapy device '{args}': {e}")
        self._soapy_dir = SOAPY_SDR_RX
        try:
            self._soapy_stream = self._soapy.setupStream(self._soapy_dir, "CF32")
            self._soapy.activateStream(self._soapy_stream)
        except Exception as e:
            raise SDRInitError(f"Soapy stream setup failed: {e}")
        self.dev = self._soapy
        return self

    # ---------- Configuration ----------
    def configure(self, center_mhz: float, sample_rate: float, gain: str | float = "auto",
                  bandwidth_hz: float | None = None, ppm: float | None = None):
        self._fs = float(sample_rate)
        if self._is_rtl_like():
            g = 'auto' if str(gain).lower() == 'auto' else float(gain)
            self.dev.center_freq = float(center_mhz) * 1e6
            self.dev.sample_rate = float(sample_rate)
            self.dev.gain = g
            if ppm is not None:
                try:
                    self.dev.freq_correction = int(ppm)
                except Exception:
                    pass
        elif self._is_soapy():
            self.dev.setFrequency(self._soapy_dir, 0, float(center_mhz) * 1e6)
            self.dev.setSampleRate(self._soapy_dir, 0, float(sample_rate))
            if bandwidth_hz:
                try:
                    self.dev.setBandwidth(self._soapy_dir, 0, float(bandwidth_hz))
                except Exception:
                    pass
            try:
                if str(gain).lower() == 'auto':
                    try:
                        self.dev.setGainMode(self._soapy_dir, 0, True)
                    except Exception:
                        pass
                else:
                    try:
                        self.dev.setGainMode(self._soapy_dir, 0, False)
                    except Exception:
                        pass
                    self.dev.setGain(self._soapy_dir, 0, float(gain))
            except Exception:
                pass
            if ppm is not None:
                try:
                    self.dev.setFrequencyCorrection(self._soapy_dir, 0, float(ppm))
                except Exception:
                    pass
        else:
            raise SDRInitError("Device not opened or unknown kind for configure()")

    # ---------- Reading samples ----------
    def read_samples(self, n: int) -> np.ndarray:
        if self._is_rtl_like():
            data = self.dev.read_samples(n)  # returns complex64
            return np.asarray(data, dtype=np.complex64)
        elif self._is_soapy():
            import numpy as _np
            to_get = int(n)
            out = _np.empty(to_get, dtype=_np.complex64)
            got = 0
            reads = 0
            while got < to_get and reads < 1000:
                sr = self.dev.readStream(self._soapy_stream, [out[got:]], to_get - got)
                if isinstance(sr, tuple):
                    ret = sr[0]
                else:
                    ret = sr.ret
                if ret and ret > 0:
                    got += int(ret)
                reads += 1
            if got < to_get:
                out[got:] = 0
            return out
        else:
            raise SDRInitError("Device not opened or unknown kind for read_samples()")

    # ---------- Cleanup ----------
    def close(self):
        try:
            if self._is_rtl_like():
                self.dev.close()
            elif self._is_soapy():
                try:
                    self.dev.deactivateStream(self._soapy_stream)
                except Exception:
                    pass
                try:
                    self.dev.closeStream(self._soapy_stream)
                except Exception:
                    pass
        finally:
            self.dev = None
            self._soapy_stream = None

    # ---------- Helpers ----------
    def _is_soapy(self) -> bool:
        return self._soapy_stream is not None

    def _is_rtl_like(self) -> bool:
        # Both RTL-USB and RTL-TCP via pyrtlsdr look the same to us for reading
        try:
            from rtlsdr import RtlSdr  # noqa: F401
            return hasattr(self.dev, 'read_samples')
        except Exception:
            return False
