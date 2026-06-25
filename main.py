import csv
import queue
import signal
import threading
import time
from collections import deque

import aubio
import numpy as np
import sounddevice as sd
import soxr
from ai_edge_litert.interpreter import Interpreter
from rpi_hardware_pwm import HardwarePWM

# ---------------------------------------------------------------------------
# GPIO / Hardware PWM
# ---------------------------------------------------------------------------
GPIO_PIN      = 18      # must be 18 (PWM ch0) or 19 (PWM ch1) with dtoverlay=pwm
GPIO_PWM_HZ   = 1000    # PWM carrier frequency in Hz
GPIO_PWM_DUTY = 50.0    # duty cycle % when "on"

# HardwarePWM channel: GPIO 18/12 = channel 0, GPIO 19/13 = channel 1
_pwm = HardwarePWM(pwm_channel=0, hz=GPIO_PWM_HZ, chip=0)
_pwm.start(0)   # initialise with 0% duty (off)
_gpio_state = 0


def gpio_cleanup():
    try:
        _pwm.stop()
    except Exception:
        pass

def _signal_handler(sig, frame):
    gpio_cleanup()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGHUP,  _signal_handler)

# ---------------------------------------------------------------------------
# YAMNet model
# ---------------------------------------------------------------------------
interpreter = Interpreter('lite-model_yamnet_classification_tflite_1.tflite')
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('yamnet_classes.csv', newline='') as f:
    reader = csv.DictReader(f)
    rows   = list(reader)
    name_col       = 'display_name' if 'display_name' in reader.fieldnames else reader.fieldnames[-1]
    YAMNET_CLASSES = [r[name_col] for r in rows]

print(f"Loaded {len(YAMNET_CLASSES)} YAMNet classes from CSV")

CLAP_CLASS_IDX = next(
    (i for i, n in enumerate(YAMNET_CLASSES) if 'clap' in n.lower()), 62
)
print(f"Clap class: [{CLAP_CLASS_IDX}] {YAMNET_CLASSES[CLAP_CLASS_IDX]}")

# ---------------------------------------------------------------------------
# Audio config
# ---------------------------------------------------------------------------
DEVICE_RATE   = 48000
YAMNET_RATE   = 16000
YAMNET_WINDOW = 15600
DEVICE_WINDOW = int(YAMNET_WINDOW * DEVICE_RATE / YAMNET_RATE)
HOP           = 512

# ---------------------------------------------------------------------------
# Detection config
# ---------------------------------------------------------------------------
DESCRIPTOR_GATE        = 20000
DOUBLE_CLAP_MIN        = 0.1   # minimum seconds between two claps
DOUBLE_CLAP_WINDOW     = 0.5   # maximum seconds between two claps
COOLDOWN               = 0
YAMNET_TOP_N           = 5
DEBUG                  = True
CAPTURE_DELAY          = 0.15

# Watchdog: exit if no audio callback for this many seconds (systemd restarts)
WATCHDOG_TIMEOUT       = 120

# Aubio reset: rebuild detector every N seconds to prevent HFC state drift
AUBIO_RESET_SEC        = 3600

# ---------------------------------------------------------------------------
# Heuristic: composite clap score
# ---------------------------------------------------------------------------
HEURISTIC_THRESHOLD = 0.015

def _find_class(fragment):
    return next(
        (i for i, n in enumerate(YAMNET_CLASSES) if fragment.lower() in n.lower()), None
    )

HEURISTIC_POSITIVE = [
    (_find_class('clapping'),        2.0),
    (_find_class('hands'),           1.5),
    (_find_class('slap'),            1.2),
    (_find_class('finger snapping'), 0.5),
    (_find_class('chop'),            0.4),
    (_find_class('tap'),             0.3),
    (_find_class('cap gun'),         0.3),
    (_find_class('percussion'),      0.3),
    (_find_class('gunshot'),         0.2),
    (_find_class('explosion'),       0.2),
    (_find_class('burst'),           0.2),
]
HEURISTIC_POSITIVE = [(i, w) for i, w in HEURISTIC_POSITIVE if i is not None]
_POSITIVE_SUM = sum(w for _, w in HEURISTIC_POSITIVE)

HEURISTIC_NEGATIVE = [
    (_find_class('speech'),        1.2),
    (_find_class('vehicle horn'),  0.3),
    (_find_class('typewriter'),    0.6),
    (_find_class('typing'),        0.6),
]
HEURISTIC_NEGATIVE = [(i, w) for i, w in HEURISTIC_NEGATIVE if i is not None]

def clap_heuristic_score(scores):
    pos     = sum(scores[i] * w for i, w in HEURISTIC_POSITIVE)
    neg     = sum(scores[i] * w for i, w in HEURISTIC_NEGATIVE)
    penalty = max(0.0, neg - pos) * 0.5
    return float(np.clip((pos - penalty) / _POSITIVE_SUM, 0.0, 1.0))

# ---------------------------------------------------------------------------
# Aubio onset detector
# ---------------------------------------------------------------------------
_aubio_lock = threading.Lock()

def _make_onset():
    o = aubio.onset("hfc", buf_size=1024, hop_size=HOP, samplerate=DEVICE_RATE)
    o.set_threshold(0.5)
    o.set_minioi_ms(100)
    o.set_silence(-20)
    return o

onset_detector = _make_onset()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
audio_buffer      = deque(maxlen=DEVICE_WINDOW)
clap_times        = deque(maxlen=10)
onset_queue       = queue.Queue(maxsize=20)
buffer_lock       = threading.Lock()
last_double_time  = 0
_callback_count   = 0
_last_callback_ts = time.time()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dbg(msg):
    if DEBUG:
        print(f"[{time.time():.3f}] {msg}")

def trigger():
    global _gpio_state
    _gpio_state ^= 1
    if _gpio_state:
        _pwm.change_duty_cycle(GPIO_PWM_DUTY)
        state_str = f"PWM {GPIO_PWM_DUTY}%"
    else:
        _pwm.change_duty_cycle(0)
        state_str = "OFF"
    print(f">>> Double clap detected! GPIO {GPIO_PIN} → {state_str} <<<")

# ---------------------------------------------------------------------------
# Watchdog thread
# ---------------------------------------------------------------------------
def watchdog():
    dbg(f"watchdog | started (timeout={WATCHDOG_TIMEOUT}s)")
    time.sleep(WATCHDOG_TIMEOUT)
    while True:
        time.sleep(WATCHDOG_TIMEOUT)
        age = time.time() - _last_callback_ts
        if age > WATCHDOG_TIMEOUT:
            print(f"[{time.time():.3f}] watchdog | DEAD — no callback for {age:.0f}s, triggering restart")
            gpio_cleanup()
            raise SystemExit(1)
        else:
            dbg(f"watchdog | ok — last callback {age:.1f}s ago, total calls={_callback_count}")

threading.Thread(target=watchdog, daemon=True).start()

# ---------------------------------------------------------------------------
# YAMNet worker thread
# ---------------------------------------------------------------------------
def yamnet_worker():
    global last_double_time
    dbg("yamnet_worker | started")

    while True:
        onset_time = onset_queue.get()
        dbg(f"yamnet_worker | picked up onset, queue size={onset_queue.qsize()}")

        elapsed   = time.time() - onset_time
        remaining = CAPTURE_DELAY - elapsed
        if remaining > 0:
            time.sleep(remaining)

        with buffer_lock:
            if len(audio_buffer) < DEVICE_WINDOW:
                dbg(f"yamnet_worker | buffer too small ({len(audio_buffer)}/{DEVICE_WINDOW}), skipping")
                continue
            audio_snapshot = np.array(audio_buffer, dtype=np.float32)

        resampled = soxr.resample(audio_snapshot, DEVICE_RATE, YAMNET_RATE)
        interpreter.set_tensor(input_details[0]['index'], resampled)
        interpreter.invoke()
        scores     = interpreter.get_tensor(output_details[0]['index'])[0]
        clap_score = scores[CLAP_CLASS_IDX]
        composite  = clap_heuristic_score(scores)

        if DEBUG:
            top_idx = np.argsort(scores)[::-1][:YAMNET_TOP_N]
            top_str = "  |  ".join(
                f"{'*** ' if i == CLAP_CLASS_IDX else ''}"
                f"{YAMNET_CLASSES[i] if i < len(YAMNET_CLASSES) else f'idx{i}'}={scores[i]:.3f}"
                f"{' ***' if i == CLAP_CLASS_IDX else ''}"
                for i in top_idx
            )
            dbg(f"yamnet_worker | clap={clap_score:.4f}  composite={composite:.4f}  top-{YAMNET_TOP_N}: {top_str}")

        if composite < HEURISTIC_THRESHOLD:
            dbg(f"yamnet_worker | rejected (composite={composite:.4f} < {HEURISTIC_THRESHOLD})")
            continue

        now    = onset_time
        recent = [t for t in clap_times if DOUBLE_CLAP_MIN < now - t <= DOUBLE_CLAP_WINDOW]
        clap_times.append(now)
        dbg(f"yamnet_worker | clap confirmed, {len(recent)} prior clap(s) in window "
            f"({DOUBLE_CLAP_MIN}s–{DOUBLE_CLAP_WINDOW}s)")

        if recent and (now - last_double_time) > COOLDOWN:
            last_double_time = now
            clap_times.clear()
            trigger()
        elif not recent:
            dbg("yamnet_worker | first clap (or too close/far), waiting for second")
        else:
            dbg(f"yamnet_worker | in cooldown, {COOLDOWN - (now - last_double_time):.2f}s remaining")

# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    global _callback_count, _last_callback_ts, onset_detector

    _callback_count   += 1
    _last_callback_ts  = time.time()

    # heartbeat every ~54s
    if _callback_count % 5000 == 0:
        dbg(f"audio_callback | heartbeat (calls={_callback_count}, aubio_reset_in="
            f"{AUBIO_RESET_SEC - (_callback_count * HOP // DEVICE_RATE) % AUBIO_RESET_SEC}s)")

    # periodic aubio reset to prevent HFC state drift
    aubio_reset_calls = AUBIO_RESET_SEC * DEVICE_RATE // HOP
    if _callback_count % aubio_reset_calls == 0:
        with _aubio_lock:
            onset_detector = _make_onset()
        dbg(f"audio_callback | aubio reset at call {_callback_count} "
            f"(uptime={(aubio_reset_calls * HOP / DEVICE_RATE * (_callback_count // aubio_reset_calls)) / 3600:.1f}h)")

    if status:
        print(f"[{time.time():.3f}] audio_callback | WARNING status={status}")

    audio = indata[:, 0].astype(np.float32)
    with buffer_lock:
        audio_buffer.extend(audio)

    with _aubio_lock:
        is_onset   = onset_detector(audio)
        descriptor = onset_detector.get_descriptor()

    if descriptor > 50000:
        dbg(f"audio_callback | descriptor={descriptor:.0f} is_onset={bool(is_onset)}")

    if is_onset:
        if descriptor > DESCRIPTOR_GATE:
            try:
                onset_queue.put_nowait(time.time())
                dbg(f"audio_callback | strong onset passed gate (descriptor={descriptor:.0f})")
            except queue.Full:
                dbg("audio_callback | onset queue full, dropping")
        else:
            dbg(f"audio_callback | onset below gate, ignoring (descriptor={descriptor:.0f})")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
threading.Thread(target=yamnet_worker, daemon=True).start()

print("Listening for double claps...")
print(f"  GPIO pin           : BCM {GPIO_PIN} (hardware PWM, toggle)")
print(f"  GPIO PWM           : {GPIO_PWM_HZ} Hz, {GPIO_PWM_DUTY}% duty when on")
print(f"  descriptor gate    : {DESCRIPTOR_GATE}")
print(f"  heuristic threshold: {HEURISTIC_THRESHOLD}")
print(f"  double clap window : {DOUBLE_CLAP_MIN}s – {DOUBLE_CLAP_WINDOW}s")
print(f"  capture delay      : {CAPTURE_DELAY}s")
print(f"  cooldown           : {COOLDOWN}s")
print(f"  watchdog timeout   : {WATCHDOG_TIMEOUT}s")
print(f"  aubio reset        : every {AUBIO_RESET_SEC}s ({AUBIO_RESET_SEC//3600}h)")
print(f"  debug              : {DEBUG}")
print()

try:
    with sd.InputStream(callback=audio_callback, samplerate=DEVICE_RATE,
                        channels=1, blocksize=HOP, device=0):
        threading.Event().wait()
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    gpio_cleanup()
