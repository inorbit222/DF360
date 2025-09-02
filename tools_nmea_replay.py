#!/usr/bin/env python3
import socket, time, sys

NMEA = [
    # GGA (fix)
    "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47",
    # RMC (course/speed)
    "$GPRMC,123520,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A",
    "$GPRMC,123521,A,4807.050,N,01131.020,E,030.0,085.0,230394,003.1,W*68",
    "$GPRMC,123522,A,4807.070,N,01131.040,E,028.0,086.0,230394,003.1,W*6D",
    "$GPGGA,123523,4807.090,N,01131.060,E,2,10,0.8,545.6,M,46.9,M,,*4A",
]

def main(host="127.0.0.1", port=10110, period=0.5):
    addr = (host, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Broadcasting {len(NMEA)} NMEA sentences to udp://{host}:{port} every {period}s")
    i = 0
    try:
        while True:
            line = (NMEA[i % len(NMEA)] + "\n").encode("ascii")
            sock.sendto(line, addr)
            i += 1
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 10110
    period = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    main(host, port, period)
