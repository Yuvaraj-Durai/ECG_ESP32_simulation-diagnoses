"""
send_ecg.py - stream a simulated ECG to the ESP32 over USB serial (alternative to the browser page).

    pip install numpy pyserial
    python send_ecg.py --port COM5 --dx afib --age 6                # Windows
    python send_ecg.py --port /dev/ttyUSB0 --dx stemi --age 58 --height 175 --weight 92
    python send_ecg.py --dump test.bin --dx vt                      # no board: write 20 s of frames to a file

The bytes on the wire are identical to what ecg-simulator-esp32.html sends.
"""
import argparse, struct, sys, threading, time
import numpy as np
import ecg_sim as E


def crc8(data, crc=0):
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def pack(ftype, payload):
    body = bytes([ftype, len(payload)]) + payload
    return b'\xA5\x5A' + body + bytes([crc8(body)])


def to_frames(x_mv, chunk=25):
    q = np.clip(np.round(x_mv * 1000.0), -32768, 32767).astype('<i2')
    return [pack(1, q[i:i + chunk].tobytes()) for i in range(0, len(q) - chunk + 1, chunk)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', help='serial port, e.g. COM5 or /dev/ttyUSB0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--dx', default='nsr', choices=E.DX_IDS)
    ap.add_argument('--age', type=float, default=35.0, help='years (0.5 = six months)')
    ap.add_argument('--height', type=float, help='cm (default: typical for age)')
    ap.add_argument('--weight', type=float, help='kg (default: typical for age)')
    ap.add_argument('--no-noise', action='store_true')
    ap.add_argument('--dump', help='write frames to this file instead of a serial port')
    a = ap.parse_args()

    th, tw = E.typical_for(a.age)
    h = a.height or th
    w = a.weight or tw
    dx_idx = E.DX_IDS.index(a.dx)
    cfg = pack(2, struct.pack('<fffB', a.age, h, w, dx_idx))
    print('%s | age %.2f y, %.0f cm, %.1f kg' % (E.DX_NAMES[dx_idx], a.age, h, w))

    rng = np.random.default_rng()

    def block(seconds, t_start):
        x, _, _ = E.simulate(a.dx, a.age, h, w, dur=seconds, fs=250, rng=rng,
                             noise=not a.no_noise, t_start=t_start)
        return to_frames(x)

    if a.dump:
        with open(a.dump, 'wb') as f:
            f.write(cfg)
            for fr in block(20, 100.0):
                f.write(fr)
        print('wrote', a.dump)
        return

    if not a.port:
        sys.exit('give --port (or --dump FILE)')
    import serial
    ser = serial.Serial(a.port, a.baud, timeout=0)
    ser.dtr = False
    ser.rts = False
    print('waiting for the board to boot ...')
    time.sleep(2.0)

    def reader():
        buf = b''
        while True:
            buf += ser.read(256)
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                line = line.decode(errors='ignore').strip()
                if line.startswith('P,'):
                    p = line.split(',')
                    try:
                        print('ESP32 predicts: %-26s %3s %%   %s ms   HR %s' %
                              (E.DX_NAMES[int(p[1])], p[2], p[3], p[4] if len(p) > 4 else '?'))
                    except (ValueError, IndexError):
                        pass
            time.sleep(0.02)
    threading.Thread(target=reader, daemon=True).start()

    ser.write(cfg)
    t_sim = 100.0
    next_cfg = time.perf_counter() + 2.0
    t0 = time.perf_counter()
    sent = 0                                    # frames sent so far (each = 0.1 s of signal)
    try:
        while True:
            frames = block(30, t_sim)           # 30 s at a time
            t_sim += 30.0
            for fr in frames:
                due = t0 + sent * 0.1           # real-time pacing
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                ser.write(fr)
                sent += 1
                if time.perf_counter() > next_cfg:
                    ser.write(cfg)
                    next_cfg += 2.0
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
