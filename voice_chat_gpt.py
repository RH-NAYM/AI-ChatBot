import os
import sounddevice as sd
import numpy as np
import wave
import pyttsx3
from faster_whisper import WhisperModel
from collections import deque
import time

# For LLaMA local inference
try:
    from llama_cpp import Llama
except ImportError:
    print("Installing llama-cpp-python...")
    os.system("pip install llama-cpp-python")
    from llama_cpp import Llama

# ------------------------------
# 🎤 Audio Settings
# ------------------------------
MIC_DEVICE_INDEX = 10  # Change to your mic index
CHANNELS = 1
SAMPLE_RATE = 44100
AUDIO_QUEUE = deque()
FILENAME = "input.wav"
SILENCE_THRESHOLD = 500  # Adjust as needed
SILENCE_DURATION = 5  # seconds to stop recording after silence

device_info = sd.query_devices(MIC_DEVICE_INDEX, 'input')
SAMPLE_RATE = int(device_info['default_samplerate'])
print(f"🎤 Using device: {device_info['name']} ({MIC_DEVICE_INDEX}) at {SAMPLE_RATE} Hz")

# ------------------------------
# 🧠 Initialize Faster Whisper
# ------------------------------
whisper_model = WhisperModel("medium.en", device="cuda")  # Or "cpu"

def record_audio(filename=FILENAME, silence_duration=SILENCE_DURATION):
    print("🎙️ Speak now... (5s silence will stop recording)")
    buffer = []
    start_time = time.time()
    silence_start = None

    def callback(indata, frames, time_info, status):
        nonlocal silence_start
        buffer.append(indata.copy())
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
        else:
            silence_start = None

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype='int16', device=MIC_DEVICE_INDEX,
                        callback=callback):
        while True:
            sd.sleep(100)
            if silence_start and (time.time() - silence_start) > silence_duration:
                break

    audio = np.concatenate(buffer)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    return filename

def transcribe_audio(filename):
    print("🔍 Transcribing...")
    segments, _ = whisper_model.transcribe(filename)
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"🗣️ You said: {text}")
    return text

# ------------------------------
# 🤖 LLaMA Local Response
# ------------------------------
MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading LLaMA 2 7B model locally...")
    os.makedirs("models", exist_ok=True)
    # Replace with correct GGUF URL
    os.system(f"wget -O {MODEL_PATH} https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf")

llm = Llama(model_path=MODEL_PATH, n_threads=8)

def get_ai_response(prompt):
    print("🤖 Thinking locally with LLaMA...")
    response = llm(prompt, max_tokens=256)
    reply = response['choices'][0]['text'].strip()
    print(f"🤖 AI: {reply}")
    return reply

# ------------------------------
# 🔊 Text-to-Speech
# ------------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 180)

def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# ------------------------------
# 🔁 Main Voice Chat Loop
# ------------------------------
def voice_chat_loop():
    print("🧠 Voice Chat Ready (say 'exit' to quit)\n")
    while True:
        try:
            audio_file = record_audio()
            user_text = transcribe_audio(audio_file)
            if not user_text:
                continue
            if "exit" in user_text.lower():
                print("👋 Goodbye!")
                break

            ai_text = get_ai_response(user_text)
            speak_text(ai_text)

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"⚠ Error: {e}")

# ------------------------------
if __name__ == "__main__":
    voice_chat_loop()
