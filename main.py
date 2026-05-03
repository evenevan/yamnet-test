import numpy as np
import sounddevice as sd
import soxr
import aubio
import time
import threading
from collections import deque
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter('lite-model_yamnet_classification_tflite_1.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

DEVICE_RATE = 48000
YAMNET_RATE = 16000
YAMNET_WINDOW = 15600
DEVICE_WINDOW = int(YAMNET_WINDOW * DEVICE_RATE / YAMNET_RATE)  # ~46800 samples
HOP = 512
DOUBLE_CLAP_WINDOW = 2.5
COOLDOWN = 1.0
DESCRIPTOR_GATE = 20000  # only pass strong transients to YAMNet

onset = aubio.onset("hfc", buf_size=1024, hop_size=HOP, samplerate=DEVICE_RATE)
onset.set_threshold(0.9)
onset.set_minioi_ms(100)
onset.set_silence(-20)

audio_buffer = deque(maxlen=DEVICE_WINDOW)
clap_times = deque(maxlen=10)
last_double_time = 0
onset_queue = deque(maxlen=20)
buffer_lock = threading.Lock()

def yamnet_worker():
    global last_double_time
    print("[yamnet_worker] thread started")
    while True:
        if onset_queue:
            onset_time = onset_queue.popleft()
            print(f"[yamnet_worker] picked up onset at {onset_time:.3f}, queue size now={len(onset_queue)}")

            with buffer_lock:
                buf_len = len(audio_buffer)
                if buf_len < DEVICE_WINDOW:
                    print(f"[yamnet_worker] buffer too small ({buf_len}/{DEVICE_WINDOW}), skipping")
                    continue
                audio_snapshot = np.array(audio_buffer, dtype=np.float32)

            print(f"[yamnet_worker] resampling {len(audio_snapshot)} samples...")
            resampled = soxr.resample(audio_snapshot, DEVICE_RATE, YAMNET_RATE)
            print(f"[yamnet_worker] running inference...")
            interpreter.set_tensor(input_details[0]['index'], resampled)
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])
            clap_score = scores[0][58]
            print(f"[yamnet_worker] YAMNet score={clap_score:.3f}")

            if clap_score >= 0.025:
                now = onset_time
                clap_times.append(now)
                recent = [t for t in clap_times if 0 < now - t < DOUBLE_CLAP_WINDOW]
                print(f"[yamnet_worker] clap confirmed, recent claps={recent}")

                if recent and (now - last_double_time) > COOLDOWN:
                    print(">>> Double clap detected! <<<")
                    last_double_time = now
                    clap_times.clear()  # reset so rapid follow-up claps don't re-trigger
                    # trigger your MOSFET here
            else:
                print(f"[yamnet_worker] onset rejected by YAMNet (score={clap_score:.3f})")
        else:
            time.sleep(0.01)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio_callback] status: {status}")

    audio = indata[:, 0].astype(np.float32)

    with buffer_lock:
        audio_buffer.extend(audio)

    is_onset = onset(audio)
    descriptor = onset.get_descriptor()

    if is_onset and descriptor > DESCRIPTOR_GATE:
        t = time.time()
        onset_queue.append(t)
        print(f"[audio_callback] strong onset at {t:.3f} descriptor={descriptor:.0f}")
    elif is_onset:
        print(f"[audio_callback] onset below gate, ignoring (descriptor={descriptor:.0f})")

worker = threading.Thread(target=yamnet_worker, daemon=True)
worker.start()

print("Listening...")
with sd.InputStream(callback=audio_callback, samplerate=DEVICE_RATE,
                    channels=1, blocksize=HOP, device=0):
    while True:
        sd.sleep(100)
