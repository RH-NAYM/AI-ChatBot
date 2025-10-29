import os
import sounddevice as sd
import numpy as np
import wave
import pyttsx3
from faster_whisper import WhisperModel
from openai import OpenAI
from collections import deque
# -----------------------------------------------------------
# 🔐 SET YOUR OPENAI API KEY HERE
# -----------------------------------------------------------
# ✅ REPLACE with your actual key (keep it private!)
OPENAI_API_KEY = "sk-my api key"

# Set it to environment variable (so OpenAI client can read it)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI()

# ------------------------------
# 🎤 Audio Settings
# ------------------------------
MIC_DEVICE_INDEX = 10  # ALC897 Analog mic
audio_queue = deque()

# Query device info for correct sample rate
device_info = sd.query_devices(MIC_DEVICE_INDEX, 'input')
SAMPLE_RATE = int(device_info['default_samplerate'])
CHANNELS = 1
DURATION = 6  # seconds per recording
FILENAME = "input.wav"

print(f"🎤 Using device: {device_info['name']} ({MIC_DEVICE_INDEX}) at {SAMPLE_RATE} Hz")

# ------------------------------
# 🎙️ Record Audio
# ------------------------------
def record_audio(filename=FILENAME, duration=DURATION):
    print("🎙️ Speak now...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', device=MIC_DEVICE_INDEX)
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return filename

# ------------------------------
# 🧠 Speech-to-Text (Faster Whisper)
# ------------------------------
def transcribe_audio(filename):
    print("🔍 Transcribing...")
    model = WhisperModel("medium", device="cuda")  # Use "cpu" if no GPU
    segments, _ = model.transcribe(filename)
    text = " ".join([seg.text for seg in segments]).strip()
    print(f"🗣️ You said: {text}")
    return text

# ------------------------------
# 🤖 GPT Response
# ------------------------------
def get_ai_response(prompt):
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OpenAI API Key.")
        return "No API key configured."

    print("🤖 Thinking...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    reply = response.choices[0].message.content.strip()
    print(f"🤖 AI: {reply}")
    return reply

# ------------------------------
# 🔊 Text-to-Speech
# ------------------------------
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

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