import numpy as np
import sounddevice as sd
import soxr
import aubio
import time
import threading
from collections import deque
from ai_edge_litert.interpreter import Interpreter

# --- YAMNet setup ---
interpreter = Interpreter('lite-model_yamnet_classification_tflite_1.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Config ---
DEVICE_RATE = 48000
YAMNET_RATE = 16000
YAMNET_WINDOW = 15600
DEVICE_WINDOW = int(YAMNET_WINDOW * DEVICE_RATE / YAMNET_RATE)
HOP = 512
DESCRIPTOR_GATE = 20000
YAMNET_THRESHOLD = 0.025
DOUBLE_CLAP_WINDOW = 2.5
COOLDOWN = 1.0

# --- Aubio onset detector ---
onset = aubio.onset("hfc", buf_size=1024, hop_size=HOP, samplerate=DEVICE_RATE)
onset.set_threshold(0.9)
onset.set_minioi_ms(100)
onset.set_silence(-20)

# --- Shared state ---
audio_buffer = deque(maxlen=DEVICE_WINDOW)
clap_times = deque(maxlen=10)
onset_queue = deque(maxlen=20)
buffer_lock = threading.Lock()
last_double_time = 0


def trigger():
    print(">>> Double clap detected! <<<")


def yamnet_worker():
    global last_double_time
    while True:
        if not onset_queue:
            time.sleep(0.01)
            continue

        onset_time = onset_queue.popleft()

        with buffer_lock:
            if len(audio_buffer) < DEVICE_WINDOW:
                continue
            audio_snapshot = np.array(audio_buffer, dtype=np.float32)

        resampled = soxr.resample(audio_snapshot, DEVICE_RATE, YAMNET_RATE)
        interpreter.set_tensor(input_details[0]['index'], resampled)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])

        if scores[0][58] < YAMNET_THRESHOLD:
            continue

        now = onset_time
        clap_times.append(now)
        recent = [t for t in clap_times if 0 < now - t < DOUBLE_CLAP_WINDOW]

        if recent and (now - last_double_time) > COOLDOWN:
            last_double_time = now
            clap_times.clear()
            trigger()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[warning] {status}")

    audio = indata[:, 0].astype(np.float32)

    with buffer_lock:
        audio_buffer.extend(audio)

    if onset(audio) and onset.get_descriptor() > DESCRIPTOR_GATE:
        onset_queue.append(time.time())


worker = threading.Thread(target=yamnet_worker, daemon=True)
worker.start()

print("Listening for double claps...")
with sd.InputStream(callback=audio_callback, samplerate=DEVICE_RATE,
                    channels=1, blocksize=HOP, device=0):
    while True:
        sd.sleep(100)
