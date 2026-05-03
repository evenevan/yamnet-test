import numpy as np
import sounddevice as sd
import soxr
from ai_edge_litert.interpreter import Interpreter

# Load model
interpreter = Interpreter('lite-model_yamnet_classification_tflite_1.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

DEVICE_RATE = 48000       # Blue Yeti's native rate
YAMNET_RATE = 16000       # YAMNet requires this
WINDOW_SIZE = 15600       # samples at 16000Hz (~0.975s)
DEVICE_WINDOW = int(WINDOW_SIZE * DEVICE_RATE / YAMNET_RATE)  # 46800 samples at 48000Hz

def audio_callback(indata, frames, time, status):
    audio = indata[:, 0].astype(np.float32)
    resampled = soxr.resample(audio, DEVICE_RATE, YAMNET_RATE)
    interpreter.set_tensor(input_details[0]['index'], resampled)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])
    clap_score = scores[0][58]
    print(f"clap score: {clap_score:.3f}")
    if clap_score > 0.5:
        print("Clap detected!")
        # trigger your MOSFET here

print("Listening...")
with sd.InputStream(callback=audio_callback, samplerate=DEVICE_RATE,
                    channels=1, blocksize=DEVICE_WINDOW):
    while True:
        sd.sleep(1)
