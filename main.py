import csv
import signal
import threading
import time
from collections import deque

import aubio
import lgpio
import numpy as np
import sounddevice as sd
import soxr
from ai_edge_litert.interpreter import Interpreter

# ---------------------------------------------------------------------------
# GPIO
# ---------------------------------------------------------------------------
GPIO_PIN      = 17
GPIO_PWM_HZ   = 1000   # PWM carrier frequency in Hz
GPIO_PWM_DUTY = 50.0   # duty cycle % when "on"

_gpio_handle = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(_gpio_handle, GPIO_PIN, lFlags=0, level=0)
_gpio_state  = 0

def gpio_cleanup():
    try:
        lgpio.tx_pwm(_gpio_handle, GPIO_PIN, 0, 0)
    except Exception:
        pass
    try:
        lgpio.gpio_write(_gpio_handle, GPIO_PIN, 0)   # actively drive low before releasing
    except Exception:
        pass
    try:
        lgpio.gpio_claim_input(_gpio_handle, GPIO_PIN, lFlags=lgpio.SET_PULL_DOWN)
    except Exception:
        pass
    try:
        lgpio.gpiochip_close(_gpio_handle)
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
DESCRIPTOR_GATE    = 20000
DOUBLE_CLAP_WINDOW = 0.3
COOLDOWN           = 0
YAMNET_TOP_N       = 5
DEBUG              = True
CAPTURE_DELAY      = 0.15

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
    """Composite clap likelihood in [0, 1]."""
    pos     = sum(scores[i] * w for i, w in HEURISTIC_POSITIVE)
    neg     = sum(scores[i] * w for i, w in HEURISTIC_NEGATIVE)
    penalty = max(0.0, neg - pos) * 0.5
    return float(np.clip((pos - penalty) / _POSITIVE_SUM, 0.0, 1.0))

# ---------------------------------------------------------------------------
# Aubio onset detector
# ---------------------------------------------------------------------------
onset_detector = aubio.onset("hfc", buf_size=1024, hop_size=HOP, samplerate=DEVICE_RATE)
onset_detector.set_threshold(0.5)
onset_detector.set_minioi_ms(100)
onset_detector.set_silence(-20)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
audio_buffer     = deque(maxlen=DEVICE_WINDOW)
clap_times       = deque(maxlen=10)
onset_queue      = deque(maxlen=20)
buffer_lock      = threading.Lock()
last_double_time = 0

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
        lgpio.tx_pwm(_gpio_handle, GPIO_PIN, GPIO_PWM_HZ, GPIO_PWM_DUTY)
        state_str = f"PWM {GPIO_PWM_DUTY}%"
    else:
        lgpio.tx_pwm(_gpio_handle, GPIO_PIN, 0, 0)   # stop PWM
        lgpio.gpio_write(_gpio_handle, GPIO_PIN, 0)   # actively drive low
        state_str = "OFF"
    print(f">>> Double clap detected! GPIO {GPIO_PIN} → {state_str} <<<")

# ---------------------------------------------------------------------------
# YAMNet worker thread
# ---------------------------------------------------------------------------
def yamnet_worker():
    global last_double_time
    dbg("yamnet_worker started")

    while True:
        if not onset_queue:
            time.sleep(0.01)
            continue

        onset_time = onset_queue.popleft()
        dbg(f"yamnet_worker | picked up onset, queue size={len(onset_queue)}")

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
        recent = [t for t in clap_times if 0 < now - t <= DOUBLE_CLAP_WINDOW]
        clap_times.append(now)
        dbg(f"yamnet_worker | clap confirmed, {len(recent)} prior clap(s) in window")

        if recent and (now - last_double_time) > COOLDOWN:
            last_double_time = now
            clap_times.clear()
            trigger()
        elif not recent:
            dbg("yamnet_worker | first clap, waiting for second")
        else:
            dbg(f"yamnet_worker | in cooldown, {COOLDOWN - (now - last_double_time):.2f}s remaining")

# ---------------------------------------------------------------------------
# Audio callback
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[warning] {status}")
    audio = indata[:, 0].astype(np.float32)
    with buffer_lock:
        audio_buffer.extend(audio)
    is_onset   = onset_detector(audio)
    descriptor = onset_detector.get_descriptor()
    if is_onset:
        if descriptor > DESCRIPTOR_GATE:
            onset_queue.append(time.time())
            dbg(f"audio_callback | strong onset passed gate (descriptor={descriptor:.0f})")
        else:
            dbg(f"audio_callback | onset below gate, ignoring (descriptor={descriptor:.0f})")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
threading.Thread(target=yamnet_worker, daemon=True).start()

print("Listening for double claps...")
print(f"  GPIO pin           : BCM {GPIO_PIN} (toggle)")
print(f"  GPIO PWM           : {GPIO_PWM_HZ} Hz, {GPIO_PWM_DUTY}% duty when on")
print(f"  descriptor gate    : {DESCRIPTOR_GATE}")
print(f"  heuristic threshold: {HEURISTIC_THRESHOLD}")
print(f"  double clap window : {DOUBLE_CLAP_WINDOW}s")
print(f"  capture delay      : {CAPTURE_DELAY}s")
print(f"  cooldown           : {COOLDOWN}s")
print(f"  debug              : {DEBUG}")
print()

try:
    with sd.InputStream(callback=audio_callback, samplerate=DEVICE_RATE,
                        channels=1, blocksize=HOP, device=0):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    gpio_cleanup()
